# visualization.py

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

def create_heatmap(
    attention_map: torch.Tensor,
    base_image: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Generates and blends a heatmap over a base image.

    Args:
        attention_map: A 2D torch.Tensor representing the attention heatmap.
        base_image: The original image as a NumPy array.
        alpha: The opacity of the heatmap overlay.

    Returns:
        The blended image as a NumPy array.
    """
    if not isinstance(attention_map, torch.Tensor) or attention_map.ndim != 2:
        raise ValueError("attention_map must be a 2D torch.Tensor.")

    # Normalize the heatmap to the range [0, 1]
    heat_np = attention_map.cpu().numpy()
    heat_np = (heat_np - heat_np.min()) / (heat_np.max() - heat_np.min() + 1e-9)

    # Resize and apply a colormap
    heat_img = Image.fromarray((heat_np * 255).astype(np.uint8)).resize(
        (base_image.shape[1], base_image.shape[0]), resample=Image.BICUBIC
    )
    cmap = cm.get_cmap('viridis')
    heat_rgb = (cmap(np.array(heat_img) / 255.0)[..., :3] * 255).astype(np.uint8)

    # Blend the heatmap with the base image
    blended_image = ((1 - alpha) * base_image + alpha * heat_rgb).astype(np.uint8)
    return blended_image
