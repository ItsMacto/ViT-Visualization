# app.py

import gradio as gr
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any

from vit_analyzer import ViTAttentionAnalyzer
from visualization import create_heatmap
from config import IMAGE_SIZE

# --- Constants & Initialization ---
TITLE = "Interactive Vision Transformer (ViT) Attention Explorer"
DESCRIPTION = """
Upload an image to explore its attention maps.
1.  **Visualize CLS Token**: See what the model's primary classification token focuses on across the image.
2.  **Click on Image**: Click any patch on the image to see which *other* patches attend **TO** it.
Use the dropdowns to select a specific layer/head or the global "Rollout" view.
"""
analyzer = ViTAttentionAnalyzer()

# --- Helper Functions ---
def get_head_index(head_str: str) -> int | None:
    if head_str == "Average":
        return None
    try:
        # e.g., "Head 5" -> 4
        return int(head_str.split()[1]) - 1
    except (ValueError, IndexError):
        return None

# --- Gradio UI & Event Handlers ---
def build_ui():
    """Builds the Gradio interface."""
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"## {TITLE}")
        gr.Markdown(DESCRIPTION)
        gr.Markdown("""
### Understanding Attention Rollout
The **Rollout** visualization aggregates attention across all 12 layers to show global attention flow. For each layer, we average attention across all heads, add residual connections to account for skip connections, and multiply with the accumulated flow from previous layers. We also prune the bottom 90% of attention values to reduce noise and highlight dominant patterns. The result is a global map showing which input patches most influence the model's final classification.
        """)

        # Store analysis results to avoid re-computing on every interaction
        analysis_state = gr.State(value=None)

        with gr.Row():
            input_image = gr.Image(type="pil", label="Input Image", height=448, width=448)
            output_heatmap = gr.Image(interactive=False, label="Attention Heatmap", height=448, width=448)

        with gr.Row():
            info_textbox = gr.Textbox(label="Info", interactive=False, scale=3)
            cls_button = gr.Button("Visualize CLS Token", variant="primary", scale=1)

        with gr.Row():
            viz_type_dropdown = gr.Dropdown(
                choices=["Rollout"] + [f"Layer {i+1}" for i in range(analyzer.num_layers)],
                value="Rollout",
                label="Visualization Type"
            )
            head_choice_dropdown = gr.Dropdown(
                choices=["Average"] + [f"Head {i+1}" for i in range(analyzer.num_heads)],
                value="Average",
                label="Attention Head",
                visible=False # Initially hidden
            )
            opacity_slider = gr.Slider(0, 1, value=0.6, step=0.05, label="Heatmap Opacity")

        # --- Event Logic ---
        def on_upload(image: Image.Image) -> Dict[str, Any]:
            """Handles image upload: runs analysis and updates UI."""
            analysis_result = analyzer.analyze_image(image)
            # Display the processed image in both boxes initially
            return {
                analysis_state: analysis_result,
                input_image: analysis_result["processed_image"],
                output_heatmap: analysis_result["processed_image"],
                info_textbox: "Analysis complete. Click the image or 'Visualize CLS Token'."
            }

        def on_select_patch(
            evt: gr.SelectData, state: Dict[str, Any], viz_type: str, head_str: str, alpha: float
        ):
            """Handles click events on the input image to show attention TO a patch."""
            if not state:
                return state["processed_image"], "Please upload an image first."

            grid_size = state["grid_size"]
            patch_size = IMAGE_SIZE / grid_size
            col, row = int(evt.index[0] // patch_size), int(evt.index[1] // patch_size)
            patch_index = row * grid_size + col + 1  # +1 to account for CLS token

            if viz_type == "Rollout":
                attention_map = state["rollout_map"]
                # Select attention from all patches TO the selected patch (column)
                heatmap_data = attention_map[1:, patch_index].reshape(grid_size, grid_size)
                title = f"Global Rollout TO Patch ({row}, {col})"
            else:
                layer_index = int(viz_type.split()[1]) - 1
                head_index = get_head_index(head_str)
                layer_attns = state["attention_matrices"][layer_index, 0] # Squeeze batch dim

                if head_index is None: # Average head
                    attention_map = layer_attns.mean(dim=0)
                    title = f"Layer {layer_index+1}, Avg Head TO Patch ({row}, {col})"
                else: # Specific head
                    attention_map = layer_attns[head_index]
                    title = f"Layer {layer_index+1}, Head {head_index+1} TO Patch ({row}, {col})"
                heatmap_data = attention_map[1:, patch_index].reshape(grid_size, grid_size)

            blended_image = create_heatmap(heatmap_data, np.array(state["processed_image"]), alpha)
            return blended_image, title

        def on_visualize_cls(state: Dict, viz_type: str, head_str: str, alpha: float):
            """Handles button click to show attention FROM the CLS token."""
            if not state:
                return state["processed_image"], "Please upload an image first."

            grid_size = state["grid_size"]
            if viz_type == "Rollout":
                attention_map = state["rollout_map"]
                # Select attention FROM CLS token to all patches (row 0, cols 1:)
                heatmap_data = attention_map[0, 1:].reshape(grid_size, grid_size)
                title = "Global Rollout from CLS Token"
            else:
                layer_index = int(viz_type.split()[1]) - 1
                head_index = get_head_index(head_str)
                layer_attns = state["attention_matrices"][layer_index, 0] # Squeeze batch dim

                if head_index is None: # Average head
                    attention_map = layer_attns.mean(dim=0)
                    title = f"Layer {layer_index+1}, Avg Head from CLS Token"
                else: # Specific head
                    attention_map = layer_attns[head_index]
                    title = f"Layer {layer_index+1}, Head {head_index+1} from CLS Token"
                heatmap_data = attention_map[0, 1:].reshape(grid_size, grid_size)

            blended_image = create_heatmap(heatmap_data, np.array(state["processed_image"]), alpha)
            return blended_image, title

        # --- Component Wiring ---
        input_image.upload(
            on_upload,
            inputs=[input_image],
            outputs=[analysis_state, input_image, output_heatmap, info_textbox]
        )
        input_image.select(
            on_select_patch,
            inputs=[analysis_state, viz_type_dropdown, head_choice_dropdown, opacity_slider],
            outputs=[output_heatmap, info_textbox]
        )
        cls_button.click(
            on_visualize_cls,
            inputs=[analysis_state, viz_type_dropdown, head_choice_dropdown, opacity_slider],
            outputs=[output_heatmap, info_textbox]
        )
        # Show/hide the head dropdown based on whether "Rollout" is selected
        viz_type_dropdown.change(
            lambda viz_type: gr.update(visible=(viz_type != "Rollout")),
            inputs=[viz_type_dropdown],
            outputs=[head_choice_dropdown]
        )

    return demo

if __name__ == "__main__":
    app_ui = build_ui()
    app_ui.launch()
