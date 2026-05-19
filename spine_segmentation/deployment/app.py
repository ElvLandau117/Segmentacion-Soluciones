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


def rotate_image_for_analysis(image: np.ndarray, deg: float) -> np.ndarray:
    """Rotate `image` by `deg` degrees about its center for analysis (Ciclo 5.5).

    Pure function at module level so tests can exercise it without spinning
    up a Gradio app. The Gradio `predict()` closure calls this with the
    slider value before delegating to `pipeline.predict()`.

    Args:
        image: (H, W, 3) RGB uint8. May be None — returned unchanged.
        deg: rotation in degrees. Positive = counter-clockwise in display
            space (cv2 convention). Sub-degree magnitudes (|deg| < 0.5)
            are treated as 0 so the slider sitting at 0 doesn't pay for a
            warp call that produces a numerically-different array.

    Returns:
        The rotated image with the same shape as input. Corners that fall
        outside the original canvas are cut; the canvas does NOT expand.
        Border pixels are filled by `cv2.BORDER_REPLICATE` so a rotated
        radiograph does not introduce artificial black bands that the
        binary segmenter could confuse with the chest cavity.
    """
    if image is None:
        return image
    if abs(float(deg)) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(deg), 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def build_results_text(
    cobb_binary: dict | None,
    cobb_multiclass: dict | None,
    vertebrae_detected: list | None = None,
    coverage_info: dict | None = None,
    orientation_info: dict | None = None,
) -> str:
    """Render the multi-line Diagnosis Results panel with multi-curve support.

    Pure function (no I/O, no Gradio deps) so it can be unit-tested without
    spinning up an app or loading a model. The Gradio predict() closure
    feeds the output directly into the results Textbox.

    When cobb_binary carries a non-empty "curves" list (Ciclo 5.2), the panel
    lists every detected curve (principal, secondary, ...) with its direction
    and the vertebrae closest to its inflection points. When "curves" is
    missing or empty, the layout degrades to the single-curve Ciclo 5 format.

    When `coverage_info` is supplied and reports partial coverage (Ciclo 5.3
    fix F), a COVERAGE block is emitted and the Assessment switches to
    "inconclusive" when the Cobb reading is ~0 — that way a partial mask
    cannot be misread as "Normal".

    Layout (sections omitted when the corresponding data is missing):

      === COBB ANGLE - Curvas detectadas ===
      Curva principal:    XX.X deg  (Tn - Lm, convexidad <right|left>)
      Curva secundaria:   YY.Y deg  (Tn - Lm, convexidad <opposite>)
      [Curva N:            ZZ.Z deg  ...  — only if >2]

      Cross-check binary vs multiclass principal:
        Binary principal:    XX.X deg
        Multiclass:          MM.M deg  (anatomical reference, illustration only)
        Concordancia: <High agreement | Review | Significant discrepancy>

      === COVERAGE ===                              # Ciclo 5.3, only when partial
      Binary mask covers: C6 - T10  (12 of ~22 vertebrae, ~55%)
      WARNING: Lower spine (T11-L5) NOT segmented — Cobb angle may be misleading.

      === ASSESSMENT (curva mayor) ===
      <Normal/Mild/Moderate/Severe | Inconclusive — insufficient coverage>
      Numero total de curvas detectadas: N (S-curve / triple-curve / ...)

      === VERTEBRAE DETECTED (N) ===
      C7, T1, ...
    """
    binary_ok = bool(cobb_binary and cobb_binary.get("success"))
    multi_ok = bool(cobb_multiclass and cobb_multiclass.get("success"))
    curves = cobb_binary.get("curves") if binary_ok else None

    lines: list[str] = []

    # -------------------------------------------------------------------- COBB
    if curves:
        lines.append("=== COBB ANGLE - Curvas detectadas ===")
        labels = ["Curva principal:   ", "Curva secundaria:  "]
        for i, c in enumerate(curves):
            label = labels[i] if i < len(labels) else f"Curva {i + 1}:         "
            up = c.get("upper_vertebra") or "?"
            lo = c.get("lower_vertebra") or "?"
            direction = c.get("direction", "unknown")
            conv = {
                "right": "convexidad derecha",
                "left": "convexidad izquierda",
                "neutral": "sin convexidad clara",
            }.get(direction, "direccion desconocida")
            lines.append(
                f"{label} {c['cobb_angle_deg']:5.1f} deg  ({up} - {lo}, {conv})"
            )
    elif binary_ok:
        # Binary succeeded but no curves passed the noise floor (straight spine).
        lines.append("=== COBB ANGLE ===")
        lines.append(
            f"Binary method:     {cobb_binary['cobb_angle_deg']:5.1f} deg  "
            "(no clinically meaningful curves above the noise floor)"
        )
    elif cobb_binary:
        lines.append("=== COBB ANGLE ===")
        lines.append(
            f"Binary method:     ERROR - {cobb_binary.get('error', 'unknown')}"
        )

    # ------------------------------------------------ Cross-check vs multiclass
    if binary_ok and multi_ok:
        # Compare the binary principal angle (curves[0] when multi-curve detection
        # ran, else the back-compat single cobb_angle_deg) against the multiclass.
        binary_principal = (
            curves[0]["cobb_angle_deg"] if curves
            else cobb_binary["cobb_angle_deg"]
        )
        multi_deg = cobb_multiclass["cobb_angle_deg"]
        diff = abs(binary_principal - multi_deg)
        if diff <= 5.0:
            concordance = "High agreement - both methods coincide"
        elif diff <= 15.0:
            concordance = "Review recommended - methods differ slightly"
        else:
            concordance = "Significant discrepancy - specialist judgment required"
        upper = cobb_multiclass.get("upper_end_vertebra", "N/A")
        lower = cobb_multiclass.get("lower_end_vertebra", "N/A")
        lines.append("\n=== CROSS-CHECK binary vs multiclass ===")
        lines.append(f"    Binary principal:    {binary_principal:5.1f} deg")
        lines.append(
            f"    Multiclass:          {multi_deg:5.1f} deg  "
            f"(Upper={upper}, Lower={lower}; illustration / anatomical reference only)"
        )
        lines.append(f"    CONCORDANCIA: {concordance}  (diff = {diff:.1f} deg)")
    elif multi_ok and not binary_ok:
        # Binary failed entirely; multiclass is the only signal available.
        upper = cobb_multiclass.get("upper_end_vertebra", "N/A")
        lower = cobb_multiclass.get("lower_end_vertebra", "N/A")
        lines.append(
            f"Multiclass method (fallback): {cobb_multiclass['cobb_angle_deg']:5.1f} deg  "
            f"(Upper={upper}, Lower={lower})"
        )
    elif cobb_multiclass and not multi_ok:
        lines.append(
            f"Multiclass method: ERROR - {cobb_multiclass.get('error', 'unknown')}"
        )

    # ---------------------------------------------------------------- COVERAGE
    # Ciclo 5.3 fix F: when the binary mask covered only part of the spine,
    # surface that explicitly so a "0 deg" reading is not mistaken for a
    # healthy spine. Block is emitted only when coverage_info marks the
    # segmentation as partial; full coverage stays silent (less UI noise).
    coverage_is_partial = bool(
        coverage_info
        and coverage_info.get("success")
        and coverage_info.get("is_partial")
    )
    if coverage_is_partial:
        upper_v = coverage_info.get("upper_vertebra")
        lower_v = coverage_info.get("lower_vertebra")
        n_v = coverage_info.get("n_vertebrae", 0)
        n_exp = coverage_info.get("n_expected", 22)
        ratio_pct = (coverage_info.get("coverage_ratio") or 0.0) * 100.0
        lines.append("\n=== COVERAGE ===")
        if upper_v and lower_v:
            lines.append(
                f"Binary mask covers: {upper_v} - {lower_v}  "
                f"({n_v} of ~{n_exp} vertebrae, ~{ratio_pct:.0f}%)"
            )
        else:
            lines.append(
                f"Binary mask covers ~{ratio_pct:.0f}% of image height "
                "(no multiclass vertebrae to name the range)"
            )
        below = coverage_info.get("vertebrae_below_range") or []
        above = coverage_info.get("vertebrae_above_range") or []
        if below:
            rng = f"{below[0]}-{below[-1]}" if len(below) > 1 else below[0]
            lines.append(
                f"WARNING: Lower spine ({rng}) NOT segmented — "
                "Cobb angle may be misleading."
            )
        elif above:
            rng = f"{above[0]}-{above[-1]}" if len(above) > 1 else above[0]
            lines.append(
                f"WARNING: Upper spine ({rng}) NOT segmented — "
                "Cobb angle may be misleading."
            )
        else:
            lines.append(
                "WARNING: Partial spine coverage — Cobb angle may be misleading."
            )

    # ----------------------------------------------------- ROTATION WARNING
    # Ciclo 5.4 fix G: when the spine skeleton's principal axis deviates more
    # than TILT_THRESHOLD_DEG from vertical, the binary Cobb method (which
    # fits x = f(y)) reports rotation as scoliosis. Emit a warning so the
    # clinician knows to trust the multiclass measurement (which is per-
    # vertebra and largely rotation-invariant) instead. The binary number
    # is left in place — confirmed UX decision: minimum-change, maximum-info.
    is_tilted = bool(
        orientation_info
        and orientation_info.get("success")
        and orientation_info.get("is_tilted")
    )
    if is_tilted:
        tilt_abs = orientation_info.get("tilt_abs_deg", 0.0)
        threshold = orientation_info.get("threshold_deg", 12.0)
        lines.append("\n=== ROTATION WARNING ===")
        lines.append(
            f"Image appears tilted {tilt_abs:.1f} deg from vertical "
            f"(threshold {threshold:.0f} deg)."
        )
        lines.append(
            "The binary Cobb method fits x = f(y) and may report rotation "
            "as scoliosis."
        )
        lines.append(
            "Re-capture with the patient straight, or trust the multiclass "
            "measurement (per-vertebra, rotation-invariant)."
        )

    # -------------------------------------------------------------- ASSESSMENT
    # Severity uses the LARGEST curve detected by the binary method. The
    # binary method is the source of truth on our data (MAE 23 deg, r=0.66,
    # vs multiclass MAE 26-45 deg with negative correlation in the worst case).
    assessment_source = None
    if curves:
        assessment_source = ("Binary principal", curves[0]["cobb_angle_deg"])
    elif binary_ok:
        assessment_source = ("Binary", cobb_binary["cobb_angle_deg"])
    elif multi_ok:
        assessment_source = ("Multiclass fallback", cobb_multiclass["cobb_angle_deg"])

    if assessment_source is not None:
        source_label, angle = assessment_source
        # Ciclo 5.3 fix F: if coverage is partial AND the angle is ~0, the
        # binary likely missed enough spine that "Normal" would be misleading.
        # Flag it as inconclusive so the user knows to investigate instead.
        if coverage_is_partial and angle < 1.0:
            assessment = (
                "Inconclusive - insufficient binary coverage to compute Cobb. "
                "Review segmentation before interpreting."
            )
        elif angle < 10:
            assessment = "Normal (< 10 degrees)"
        elif angle < 25:
            assessment = "Mild scoliosis (10-25 degrees)"
        elif angle < 40:
            assessment = "Moderate scoliosis (25-40 degrees)"
        else:
            assessment = "Severe scoliosis (> 40 degrees)"
        lines.append(f"\n=== ASSESSMENT (based on {source_label}) ===")
        lines.append(assessment)

        if curves and len(curves) > 1:
            shape = {
                2: "doble curva (S-shape)",
                3: "triple curva",
            }.get(len(curves), f"{len(curves)} curvas")
            lines.append(f"Numero total de curvas detectadas: {len(curves)} ({shape})")

    # ---------------------------------------------------- Vertebrae list
    if vertebrae_detected:
        lines.append(f"\n=== VERTEBRAE DETECTED ({len(vertebrae_detected)}) ===")
        lines.append(", ".join(vertebrae_detected))

    return "\n".join(lines)


def create_app(
    binary_checkpoint: str = None,
    multiclass_checkpoint: str = None,
    binary_model_name: str = None,
    multiclass_model_name: str = None,
) -> gr.Blocks:
    """
    Create the Gradio application.

    Args:
        binary_checkpoint: Path to binary model checkpoint
        multiclass_checkpoint: Path to multiclass model checkpoint
        binary_model_name: Name of the binary architecture (defaults to DEFAULT_BINARY_MODEL)
        multiclass_model_name: Name of the multiclass architecture (defaults to DEFAULT_MULTICLASS_MODEL)

    Returns:
        Gradio Blocks application
    """
    # Sentinel-resolve: callers can leave the names as None and we use the
    # env-driven defaults from config. Hardcoding "unet_resnet50" here was a
    # latent bug — the deploy serves DeepLabV3+ multiclass, which has a
    # different decoder, so the checkpoint failed to load.
    binary_model_name = binary_model_name or DEFAULT_BINARY_MODEL
    multiclass_model_name = multiclass_model_name or DEFAULT_MULTICLASS_MODEL

    # Initialize pipeline
    pipeline = None
    if binary_checkpoint or multiclass_checkpoint:
        pipeline = SpineSegmentationPipeline(
            binary_checkpoint=binary_checkpoint,
            multiclass_checkpoint=multiclass_checkpoint,
            binary_model_name=binary_model_name,
            multiclass_model_name=multiclass_model_name,
        )

    def predict(input_image, rotation_deg=0.0):
        """Process uploaded radiograph, optionally rotated by `rotation_deg`.

        Ciclo 5.5: the UI slider lets the clinician straighten a tilted
        radiograph before analysis. The rotation is applied here, in the
        UI layer, BEFORE the segmentation pipeline runs — so the pipeline
        sees an image whose spine is as vertical as the clinician thinks
        is best. The rotation warning from Ciclo 5.4 stays useful as a
        post-hoc check ("you rotated, but it's still tilted X deg").
        """
        if input_image is None:
            return None, None, None, None, "Please upload a radiograph image."

        if pipeline is None:
            return None, None, None, None, "No model loaded. Please provide checkpoint paths."

        # Convert to RGB if needed
        if input_image.ndim == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        elif input_image.shape[2] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

        # Apply manual rotation (Ciclo 5.5) BEFORE handing to the pipeline.
        input_image = rotate_image_for_analysis(input_image, rotation_deg)

        # Run prediction
        results = pipeline.predict(input_image)

        # Prepare outputs
        binary_overlay = results.get("binary_overlay")
        multiclass_overlay = results.get("multiclass_overlay")
        cobb_vis = results.get("cobb_visualization")

        results_text = build_results_text(
            cobb_binary=results.get("cobb_binary"),
            cobb_multiclass=results.get("cobb_multiclass"),
            vertebrae_detected=results.get("vertebrae_detected"),
            coverage_info=results.get("coverage_info"),
            orientation_info=results.get("orientation_info"),
        )

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
                # Ciclo 5.5: manual rotation. The clinician decides if the
                # uploaded radiograph needs to be straightened before the
                # binary Cobb pipeline (which fits x = f(y), assuming a
                # vertical spine) sees it. Positive angle = counter-clockwise
                # in display space (cv2 convention).
                rotation_slider = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    value=0,
                    step=1,
                    label="Rotate image (degrees). Negative = clockwise.",
                )
                with gr.Row():
                    btn_rot_minus_90 = gr.Button("↺ -90°", size="sm")
                    btn_rot_minus_5 = gr.Button("↺ -5°", size="sm")
                    btn_rot_reset = gr.Button("Reset", size="sm")
                    btn_rot_plus_5 = gr.Button("↻ +5°", size="sm")
                    btn_rot_plus_90 = gr.Button("↻ +90°", size="sm")
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

        # Quick-rotation buttons: each adjusts the slider value by a fixed
        # delta. Reset jumps to 0. Values are clamped to (-180, 180) by
        # Gradio's slider bounds.
        def _adjust_rotation(current, delta):
            return float(max(-180.0, min(180.0, (current or 0.0) + delta)))

        btn_rot_minus_90.click(
            fn=lambda v: _adjust_rotation(v, -90),
            inputs=[rotation_slider], outputs=[rotation_slider],
        )
        btn_rot_minus_5.click(
            fn=lambda v: _adjust_rotation(v, -5),
            inputs=[rotation_slider], outputs=[rotation_slider],
        )
        btn_rot_reset.click(fn=lambda: 0.0, outputs=[rotation_slider])
        btn_rot_plus_5.click(
            fn=lambda v: _adjust_rotation(v, 5),
            inputs=[rotation_slider], outputs=[rotation_slider],
        )
        btn_rot_plus_90.click(
            fn=lambda v: _adjust_rotation(v, 90),
            inputs=[rotation_slider], outputs=[rotation_slider],
        )

        predict_btn.click(
            fn=predict,
            inputs=[input_image, rotation_slider],
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
