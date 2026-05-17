"""
Gradio web application for spine segmentation and scoliosis diagnosis.
Provides an interactive interface for uploading radiographs and viewing results.
"""

import gradio as gr
import numpy as np
import cv2

from spine_segmentation.config import (
    APP_HOST,
    APP_PORT,
    DEFAULT_BINARY_MODEL,
    DEFAULT_MULTICLASS_MODEL,
    MEDICAL_DISCLAIMER,
    MODEL_CONFIGS,
)
from spine_segmentation.data.transforms import get_inference_transforms, resize_with_padding
from spine_segmentation.deployment.inference import SpineSegmentationPipeline
from spine_segmentation.deployment.weights import ensure_weights
from spine_segmentation.evaluation.explainability import generate_confidence_map, generate_gradcam


def create_app(
    binary_checkpoint: str = None,
    multiclass_checkpoint: str = None,
    binary_model_name: str = "unet_resnet50",
    multiclass_model_name: str = "unet_resnet50",
) -> gr.Blocks:
    """
    Create the Gradio application.

    Args:
        binary_checkpoint: Path to binary model checkpoint
        multiclass_checkpoint: Path to multiclass model checkpoint
        binary_model_name: Name of the binary model architecture
        multiclass_model_name: Name of the multiclass model architecture

    Returns:
        Gradio Blocks application
    """
    # Initialize pipeline
    pipeline = None
    if binary_checkpoint or multiclass_checkpoint:
        pipeline = SpineSegmentationPipeline(
            binary_checkpoint=binary_checkpoint,
            multiclass_checkpoint=multiclass_checkpoint,
            binary_model_name=binary_model_name,
            multiclass_model_name=multiclass_model_name,
        )

    def predict(input_image):
        """Process uploaded radiograph."""
        if input_image is None:
            return None, None, None, "Please upload a radiograph image."

        if pipeline is None:
            return None, None, None, "No model loaded. Please provide checkpoint paths."

        # Convert to RGB if needed
        if input_image.ndim == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        elif input_image.shape[2] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

        # Run prediction
        results = pipeline.predict(input_image)

        # Prepare outputs
        binary_overlay = results.get("binary_overlay")
        multiclass_overlay = results.get("multiclass_overlay")
        cobb_vis = results.get("cobb_visualization")

        # Build results text
        text_lines = ["=== RESULTS ===\n"]

        # Binary Cobb angle
        cobb_binary = results.get("cobb_binary")
        if cobb_binary and cobb_binary.get("success"):
            text_lines.append(f"Cobb Angle (Binary Method): {cobb_binary['cobb_angle_deg']:.1f} degrees")
        elif cobb_binary:
            text_lines.append(f"Cobb Angle (Binary): Error - {cobb_binary.get('error', 'unknown')}")

        # Multiclass Cobb angle
        cobb_multi = results.get("cobb_multiclass")
        if cobb_multi and cobb_multi.get("success"):
            text_lines.append(f"Cobb Angle (Multiclass Method): {cobb_multi['cobb_angle_deg']:.1f} degrees")
            text_lines.append(f"  Upper end vertebra: {cobb_multi.get('upper_end_vertebra', 'N/A')}")
            text_lines.append(f"  Lower end vertebra: {cobb_multi.get('lower_end_vertebra', 'N/A')}")
        elif cobb_multi:
            text_lines.append(f"Cobb Angle (Multiclass): Error - {cobb_multi.get('error', 'unknown')}")

        # Vertebrae detected
        vertebrae = results.get("vertebrae_detected", [])
        if vertebrae:
            text_lines.append(f"\nVertebrae detected ({len(vertebrae)}):")
            text_lines.append(f"  {', '.join(vertebrae)}")

        # Scoliosis assessment
        if cobb_multi and cobb_multi.get("success"):
            angle = cobb_multi["cobb_angle_deg"]
            if angle < 10:
                assessment = "Normal (< 10 degrees)"
            elif angle < 25:
                assessment = "Mild scoliosis (10-25 degrees)"
            elif angle < 40:
                assessment = "Moderate scoliosis (25-40 degrees)"
            else:
                assessment = "Severe scoliosis (> 40 degrees)"
            text_lines.append(f"\nAssessment: {assessment}")

        results_text = "\n".join(text_lines)

        # Generate explainability panel
        explainability_img = None
        if pipeline is not None:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from pytorch_grad_cam.utils.image import show_cam_on_image

                # Use whichever model is available
                model_for_explain = None
                model_name_explain = ""
                if pipeline.binary_model is not None:
                    model_for_explain = pipeline.binary_model
                    model_name_explain = binary_model_name
                elif pipeline.multiclass_model is not None:
                    model_for_explain = pipeline.multiclass_model
                    model_name_explain = multiclass_model_name

                if model_for_explain is not None:
                    img_resized, _, _ = resize_with_padding(
                        input_image, np.zeros(input_image.shape[:2], dtype=np.uint8), 512
                    )
                    transforms = get_inference_transforms()
                    aug = transforms(image=img_resized)
                    input_t = aug["image"].unsqueeze(0).to(pipeline.device)

                    # Grad-CAM
                    gradcam = generate_gradcam(model_for_explain, input_t, model_name_explain)
                    img_float = img_resized.astype(np.float32) / 255.0
                    cam_overlay = show_cam_on_image(img_float, gradcam, use_rgb=True)

                    # Confidence map
                    task = "binary" if pipeline.binary_model is not None else "multiclass"
                    confidence = generate_confidence_map(model_for_explain, input_t, task)
                    conf_colored = plt.cm.RdYlGn(confidence)[:, :, :3]
                    conf_colored = (conf_colored * 255).astype(np.uint8)

                    # Combine side by side
                    explainability_img = np.concatenate([cam_overlay, conf_colored], axis=1)
            except Exception as e:
                print(f"Explainability error: {e}")

        return binary_overlay, multiclass_overlay, cobb_vis, explainability_img, results_text

    # Build Gradio interface
    with gr.Blocks(
        title="Spine Segmentation for Scoliosis Diagnosis",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Spine Segmentation for Scoliosis Diagnosis
            ### Automatic vertebral segmentation and Cobb angle measurement from X-ray radiographs

            Upload a spinal X-ray radiograph to get:
            - **Binary segmentation** of the spinal column
            - **Multiclass segmentation** of individual vertebrae
            - **Automated Cobb angle** measurement using two methods
            - **Explainability** — Grad-CAM and confidence maps (why the model decided)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Spinal X-ray Radiograph",
                    type="numpy",
                    height=500,
                )
                predict_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Binary Segmentation"):
                        binary_output = gr.Image(label="Spine Segmentation", height=400)
                    with gr.TabItem("Vertebrae Segmentation"):
                        multi_output = gr.Image(label="Individual Vertebrae", height=400)
                    with gr.TabItem("Cobb Angle"):
                        cobb_output = gr.Image(label="Cobb Angle Measurement", height=400)
                    with gr.TabItem("Explainability"):
                        explain_output = gr.Image(label="Grad-CAM (left) | Confidence Map (right)", height=400)
                        gr.Markdown(
                            """
                            **Grad-CAM** (izquierda): Regiones que influyeron en la decision del modelo.
                            Zonas calidas (rojo) = alta influencia.

                            **Mapa de Confianza** (derecha): Certeza del modelo por pixel.
                            Verde = alta confianza | Rojo = baja confianza (el medico debe revisar).

                            *Este sistema es una herramienta de apoyo. NO reemplaza el criterio del especialista.*
                            """
                        )

                results_text = gr.Textbox(
                    label="Diagnosis Results",
                    lines=10,
                    interactive=False,
                )

        predict_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[binary_output, multi_output, cobb_output, explain_output, results_text],
        )

        gr.Markdown(
            f"""
            ---
            **Aviso medico / Medical disclaimer:** {MEDICAL_DISCLAIMER}

            *MaIA - Master's in Artificial Intelligence - Universidad de los Andes*
            """
        )

    return app


def main():
    """Launch the Gradio application.

    Configuration source priority: CLI args > env vars (config.py defaults) > hardcoded fallbacks.
    Args exist for local dev; the container relies entirely on env vars.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Spine Segmentation Web App")
    parser.add_argument("--binary-checkpoint", type=str, default=None,
                       help="Override the binary .pth path (default: from HF Hub / cache)")
    parser.add_argument("--multiclass-checkpoint", type=str, default=None,
                       help="Override the multiclass .pth path (default: from HF Hub / cache)")
    parser.add_argument("--binary-model", type=str, default=DEFAULT_BINARY_MODEL,
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--multiclass-model", type=str, default=DEFAULT_MULTICLASS_MODEL,
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--host", type=str, default=APP_HOST)
    parser.add_argument("--port", type=int, default=APP_PORT)
    parser.add_argument("--share", action="store_true",
                       help="Create a public Gradio share link (dev only)")
    args = parser.parse_args()

    # Resolve checkpoints: local cache -> HF Hub -> None
    # Unless explicitly overridden via CLI.
    if args.binary_checkpoint is None or args.multiclass_checkpoint is None:
        resolved = ensure_weights()
        if args.binary_checkpoint is None and resolved["binary"] is not None:
            args.binary_checkpoint = str(resolved["binary"])
        if args.multiclass_checkpoint is None and resolved["multiclass"] is not None:
            args.multiclass_checkpoint = str(resolved["multiclass"])

    if args.binary_checkpoint:
        print(f"[app] binary checkpoint: {args.binary_checkpoint}")
    if args.multiclass_checkpoint:
        print(f"[app] multiclass checkpoint: {args.multiclass_checkpoint}")
    if not args.binary_checkpoint and not args.multiclass_checkpoint:
        print(
            "[app] WARNING: no checkpoints available. The UI will start but "
            "predictions will fail until weights are provided "
            "(set HF_REPO_ID or pass --*-checkpoint)."
        )

    app = create_app(
        binary_checkpoint=args.binary_checkpoint,
        multiclass_checkpoint=args.multiclass_checkpoint,
        binary_model_name=args.binary_model,
        multiclass_model_name=args.multiclass_model,
    )

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
