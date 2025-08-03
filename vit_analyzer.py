# vit_analyzer.py

import math
import torch
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from typing import Dict, Any

from config import MODEL_NAME, IMAGE_SIZE

class ViTAttentionAnalyzer:
    """Encapsulates ViT model loading, inference, and attention computation."""

    def __init__(self, model_name: str = MODEL_NAME):
        """Initializes the model and processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(
            model_name,
            output_attentions=True,
            add_pooling_layer=False,
            attn_implementation="eager"
        ).to(self.device).eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Preprocesses an image and runs the model to compute attentions and rollout map.

        Args:
            image: A PIL Image object.

        Returns:
            A dictionary containing the processed image, attention matrices, rollout map,
            and grid size.
        """
        processed_image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        with torch.no_grad():
            inputs = self.processor(processed_image, return_tensors="pt").to(self.device)
            model_outputs = self.model(**inputs)

        # Stack attention matrices from all layers
        attention_matrices = torch.stack(model_outputs.attentions)  # (layers, batch, heads, N, N)

        # The grid size is the square root of the number of patches (tokens - 1 for CLS)
        num_patches = model_outputs.last_hidden_state.size(1) - 1
        grid_size = int(math.sqrt(num_patches))

        # Compute attention rollout
        rollout_map = self.attention_rollout(attention_matrices)

        return {
            "processed_image": processed_image,
            "attention_matrices": attention_matrices.cpu(),
            "rollout_map": rollout_map.cpu(),
            "grid_size": grid_size
        }

    def attention_rollout(self, attention_matrices: torch.Tensor, discard_ratio: float = 0.9) -> torch.Tensor:
        """
        Computes attention rollout across all layers, providing a global attention map.
        This method aggregates attention flow from the input to the output of the model.

        Args:
            attention_matrices: Tensor of shape (layers, batch, heads, N, N).
            discard_ratio: Fraction of lowest attention entries to zero out per row.
                           Helps remove noise from the visualization.

        Returns:
            rollout_map: Tensor of shape (N, N) with aggregated attention flows.
        """
        # (layers, batch, heads, N, N) -> (layers, heads, N, N)
        attns = attention_matrices.squeeze(1)
        num_tokens = attns.size(-1)

        # Initialize the attention flow with an identity matrix.
        # This represents the initial state where each token attends only to itself.
        attention_flow = torch.eye(num_tokens, device=self.device)

        # Sequentially multiply the attention maps of each layer.
        for layer_attention in attns:
            # 1. Average attention across all heads in the current layer.
            avg_attention_matrix = layer_attention.mean(dim=0)

            # 2. Add identity to account for residual connections.
            # This ensures that information flowing through the skip connections is
            # included in the attention flow.
            residual_attention = avg_attention_matrix + torch.eye(num_tokens, device=self.device)

            # 3. Normalize the matrix so that each row sums to 1.
            # This step is crucial for stable propagation of attention values.
            normalized_attention = residual_attention / residual_attention.sum(dim=-1, keepdim=True)

            # --- Optional: Pruning weak attentions to reduce noise ---
            # Flatten to easily find the top-k smallest values.
            flat_attention = normalized_attention.view(num_tokens, -1)
            num_to_discard = int(flat_attention.size(1) * discard_ratio)

            if num_to_discard > 0:
                # Find the indices of the k-smallest attention values for each token.
                indices_to_discard = flat_attention.topk(num_to_discard, dim=1, largest=False).indices
                # Create a mask and set the discarded values to 0.
                mask = torch.zeros_like(flat_attention)
                mask.scatter_(1, indices_to_discard, 1)
                flat_attention.masked_fill_(mask.bool(), 0)
                # Reshape back to the original matrix shape.
                normalized_attention = flat_attention.view(num_tokens, num_tokens)

            # 4. Multiply the current layer's attention with the accumulated flow.
            # This "rolls out" the attention one layer deeper.
            attention_flow = normalized_attention @ attention_flow

        return attention_flow
