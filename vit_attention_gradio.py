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
    A class to encapsulate the ViT model, processing, and visualization logic.
    """
    def __init__(self):
        """
        Initializes the model and image processor.
        """
        self.proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            output_attentions=True,
            attn_implementation="eager", # Use eager for easier attention map access
            add_pooling_layer=False
        ).eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

    def attention_rollout(self, attns_tensor):
        """
        Computes attention rollout to aggregate attention scores across all layers.

        This method, based on the paper "Quantifying Attention Flow in Transformers",
        multiplies the attention matrices of all layers to get a single attention
        map that represents the global flow of information from input to output.

        Args:
            attns_tensor: A tensor of shape (layer, batch, head, T, T)
                          containing the attention maps from the model.

        Returns:
            A single aggregated attention map (rollout) of shape (T, T).
        """
        device = attns_tensor.device
        T = attns_tensor.size(-1)
        rollout = torch.eye(T, device=device)

        for A in attns_tensor:
            A_heads_avg = A.squeeze(0).mean(dim=0)
            A_residual = A_heads_avg + torch.eye(T, device=device)
            A_normalized = A_residual / A_residual.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(A_normalized, rollout)

        return rollout


    def prepare_data(self, image_np):
        """
        Called on image upload. Preprocesses the image, runs the model, and
        pre-computes both the attention rollout and the per-layer attention maps.
        """
        img_224 = Image.fromarray(image_np).convert("RGB").resize((224, 224))
        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = self.proc(img_224, return_tensors="pt").to(device)
            out = self.model(**inputs)
        
        patches = out.last_hidden_state.size(1) - 1
        side = int(math.sqrt(patches))
        
        all_attns_torch = torch.stack(out.attentions)
        
        rollout_map = self.attention_rollout(all_attns_torch)
        all_attns_np = all_attns_torch.cpu().numpy()
        
        return (
            np.array(img_224), 
            all_attns_np, 
            rollout_map.cpu().numpy(),
            side, 
            np.array(img_224), 
            np.array(img_224)
        )

    def get_head_idx(self, head_str):
        if not isinstance(head_str, str) or head_str == "Average":
            return self.num_heads
        try:
            return int(head_str.split(" ")[1]) - 1
        except (IndexError, ValueError):
            return self.num_heads

    def on_pointer(self, evt: gr.SelectData, base_img, all_attns_np, rollout_map_np, side, viz_type, head_str, alpha):
        if side is None or base_img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8), "Please upload an image first."

        x, y = evt.index
        patch_size = 224 // side
        row, col = int(y // patch_size), int(x // patch_size)
        patch_idx = row * side + col

        if not (0 <= row < side and 0 <= col < side):
            return base_img, "Pointer is outside the image."
        
        heatmap_tensor = None
        title = ""

        if viz_type == "Rollout":
            if rollout_map_np is None: return base_img, "Rollout data not ready."
            R = torch.from_numpy(rollout_map_np)
            heatmap_tensor = R[1:, patch_idx + 1].reshape(side, side)
            title = "Global Attention (Rollout)"
        else:
            if all_attns_np is None: return base_img, "Attention data not ready."
            layer_idx = int(viz_type.split(" ")[1]) - 1
            head_idx = self.get_head_idx(head_str)
            
            all_attns = torch.from_numpy(all_attns_np)
            attn_map_layer = all_attns[layer_idx, 0]

            if head_idx == self.num_heads:
                attn_map_single = attn_map_layer.mean(dim=0)
                title = f"Layer {layer_idx+1}, Average of all Heads"
            else:
                attn_map_single = attn_map_layer[head_idx]
                title = f"Layer {layer_idx+1}, Head {head_idx+1}"
            
            heatmap_tensor = attn_map_single[1:, patch_idx + 1].reshape(side, side)

        heatmap = F.interpolate(
            heatmap_tensor.unsqueeze(0).unsqueeze(0), 
            size=(224, 224), 
            mode="bilinear", 
            align_corners=False
        )[0, 0].numpy()
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
        cmap = cm.get_cmap('viridis')
        heatmap_rgb = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
        
        blended_image = ((1 - alpha) * base_img + alpha * heatmap_rgb).astype(np.uint8)
        
        # The red box drawing logic has been removed from here.
        
        return blended_image, title

# --- Gradio UI ---
explorer = ViTAttentionExplorer()

with gr.Blocks(title="ViT Attention Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ViT Attention Explorer")
    gr.Markdown("Upload an image, then hover over it. Use the controls to switch between a global **Rollout** view "
                "and inspecting the attention from specific **Layers** and **Heads**.")

    base_state = gr.State()
    attn_state = gr.State()
    rollout_state = gr.State()
    side_state = gr.State()

    with gr.Row():
        img_in = gr.Image(label="Input Image", type="numpy", height=448, width=448)
        with gr.Column():
            img_out = gr.Image(label="Attention Heatmap", height=448, width=448, interactive=False)
            out_title = gr.Textbox(label="Visualization Info", interactive=False)

    with gr.Row():
        viz_type_dropdown = gr.Dropdown(
            choices=["Rollout"] + [f"Layer {i+1}" for i in range(explorer.num_layers)],
            value="Rollout",
            label="Visualization Type"
        )
        head_dropdown = gr.Dropdown(
            choices=[f"Head {i+1}" for i in range(explorer.num_heads)] + ["Average"],
            value="Average",
            label="Attention Head",
            visible=False
        )
        alpha_slider = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.6, label="Heatmap Opacity")
    
    def update_head_visibility(viz_type):
        return gr.update(visible=(viz_type != "Rollout"))

    viz_type_dropdown.change(
        fn=update_head_visibility,
        inputs=viz_type_dropdown,
        outputs=head_dropdown,
        api_name=False
    )

    img_in.upload(
        fn=explorer.prepare_data,
        inputs=img_in,
        outputs=[base_state, attn_state, rollout_state, side_state, img_out, img_in],
        api_name=False
    )

    img_in.select(
        fn=explorer.on_pointer,
        inputs=[base_state, attn_state, rollout_state, side_state, viz_type_dropdown, head_dropdown, alpha_slider],
        outputs=[img_out, out_title],
        api_name=False
    )

demo.launch(share=True)
