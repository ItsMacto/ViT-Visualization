import math
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageDraw
from transformers import ViTModel, ViTImageProcessor
import matplotlib.cm as cm

class ViTAttentionExplorer:
    """
    Encapsulates ViT model loading, attention computation, and interactive visualization.
    """
    def __init__(self):
        # Load pre-trained ViT and processor
        self.proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            output_attentions=True,
            attn_implementation="eager",
            add_pooling_layer=False
        ).eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

    def attention_rollout(self, attns_tensor: torch.Tensor, discard_ratio: float = 0.8) -> torch.Tensor:
        """
        Computes attention rollout across all layers, optionally dropping the lowest attentions.

        Args:
            attns_tensor: Tensor of shape (layers, batch, heads, N, N)
            discard_ratio: Fraction of lowest attention entries to zero out per row
        Returns:
            rollout: Tensor of shape (N, N) with aggregated attention flows
        """
        # Remove batch dimension
        attns = attns_tensor.squeeze(1)  # now (layers, heads, N, N)
        device = attns.device
        N = attns.size(-1)
        rollout = torch.eye(N, device=device)
        for layer in attns:
            # fuse heads by simple mean
            A = layer.mean(dim=0)  # (N, N)
            # incorporate residual
            A = A + torch.eye(N, device=device)
            # normalize rows
            A = A / A.sum(dim=-1, keepdim=True)
            # drop lowest attentions per row (excluding class token at index 0)
            flat = A.view(N, -1)
            k = int(flat.size(1) * discard_ratio)
            if k > 0:
                idx = flat.topk(k, dim=1, largest=False).indices
                mask = torch.zeros_like(flat)
                mask.scatter_(1, idx, 1)
                flat = flat.masked_fill(mask.bool(), 0)
                A = flat.view(N, N)
            # propagate
            rollout = A @ rollout
        return rollout

    def prepare_data(self, image_np):
        """
        Preprocesses image and runs the model once to compute all attentions and rollout map.
        """
        img_224 = Image.fromarray(image_np).convert("RGB").resize((224, 224))
        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = self.proc(img_224, return_tensors="pt").to(device)
            out = self.model(**inputs)

        # compute patch grid size
        patches = out.last_hidden_state.size(1) - 1
        side = int(math.sqrt(patches))

        # stack attentions: (layers, batch, heads, N, N)
        all_attns_torch = torch.stack(out.attentions)
        # rollout map (N, N)
        rollout_map = self.attention_rollout(all_attns_torch)

        # to numpy for Gradio state
        return (
            np.array(img_224),            # base image
            all_attns_torch.cpu().numpy(),# raw attentions
            rollout_map.cpu().numpy(),     # rollout map
            side,                          # number of patches per side
            np.array(img_224),            # placeholder for initial viz
            np.array(img_224)             # placeholder for initial viz
        )

    def get_head_idx(self, head_str):
        if head_str == "Average":
            return None
        try:
            return int(head_str.split()[1]) - 1
        except:
            return None

    def on_pointer(self, evt: gr.SelectData, base_img, all_attns_np, rollout_map_np, side, viz_type, head_str, alpha):
        """
        Handles click events on the image: computes and returns a blended heatmap + title.
        """
        if base_img is None:
            return np.zeros((224,224,3),dtype=np.uint8), "Upload an image first."

        x, y = evt.index
        patch_size = 224 / side
        col = int(x // patch_size)
        row = int(y // patch_size)
        if not (0 <= row < side and 0 <= col < side):
            return base_img, "Click inside the image."
        patch_idx = row * side + col

        # select attention map
        if viz_type == "Rollout":
            R = torch.from_numpy(rollout_map_np)
            heat = R[1:, patch_idx+1].reshape(side, side)
            title = f"Global Rollout"
        else:
            layer_idx = int(viz_type.split()[1]) - 1
            head_idx = self.get_head_idx(head_str)
            A_layer = torch.from_numpy(all_attns_np)[layer_idx, 0]
            if head_idx is None:
                A_sel = A_layer.mean(dim=0)
                title = f"Layer {layer_idx+1}, Average"
            else:
                A_sel = A_layer[head_idx]
                title = f"Layer {layer_idx+1}, Head {head_idx+1}"
            heat = A_sel[1:, patch_idx+1].reshape(side, side)

        # normalize & resize
        heat = heat.numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        heat_img = Image.fromarray((heat*255).astype(np.uint8)).resize((224,224))
        heat_arr = np.array(heat_img) / 255.0
        cmap = cm.get_cmap('viridis')
        heat_rgb = (cmap(heat_arr)[...,:3] * 255).astype(np.uint8)

        blended = ((1-alpha)*base_img + alpha*heat_rgb).astype(np.uint8)
        return blended, title

# --- Gradio UI ---
explorer = ViTAttentionExplorer()
with gr.Blocks(title="ViT Attention Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ViT Attention Explorer")
    base_state = gr.State()
    attn_state = gr.State()
    rollout_state = gr.State()
    side_state = gr.State()

    with gr.Row():
        img_in = gr.Image(type="numpy", label="Input Image", height=448, width=448)
        img_out = gr.Image(interactive=False, label="Attention Map", height=448, width=448)
        out_title = gr.Textbox(label="Info", interactive=False)

    with gr.Row():
        viz_type = gr.Dropdown(choices=["Rollout"]+[f"Layer {i+1}" for i in range(explorer.num_layers)], value="Rollout", label="Visualization")
        head_choice = gr.Dropdown(choices=[f"Head {i+1}" for i in range(explorer.num_heads)]+["Average"], value="Average", label="Head", visible=False)
        alpha_slider = gr.Slider(0,1,0.6,0.05, label="Opacity")

    viz_type.change(lambda v: gr.update(visible=(v!="Rollout")), viz_type, head_choice)
    img_in.upload(explorer.prepare_data, img_in, [base_state, attn_state, rollout_state, side_state, img_out, img_in])
    img_in.select(explorer.on_pointer, [base_state, attn_state, rollout_state, side_state, viz_type, head_choice, alpha_slider], [img_out, out_title])

    demo.launch(share=True)
