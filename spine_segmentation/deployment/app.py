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
from spine_segmentation.deployment.i18n import (
    DEFAULT_LANG,
    header_markdown,
    label_to_lang,
    t,
)
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


def _draw_colorbar_strip(
    image: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    cmap_name: str,
    lang: str,
) -> None:
    """Paint a small vertical colorbar inside `image` at (x, y, width, height).

    Top = high value (color at cmap(1.0)), bottom = low (cmap(0.0)). Labels
    "Alta/High" and "Baja/Low" anchored at the strip's top and bottom edges.
    Mutates `image` in place.
    """
    import matplotlib.pyplot as plt

    cmap_obj = plt.get_cmap(cmap_name)
    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)
    colors = (cmap_obj(gradient)[:, :3] * 255).astype(np.uint8)  # (height, 3)
    strip = np.broadcast_to(colors[:, None, :], (height, width, 3))
    image[y:y + height, x:x + width] = strip
    # Thin border so the strip is readable against any background.
    cv2.rectangle(image, (x - 1, y - 1), (x + width, y + height), (200, 200, 200), 1)
    # Labels — short so they fit in the narrow margin we reserve.
    label_high = t("explain_colorbar_high", lang)
    label_low = t("explain_colorbar_low", lang)
    cv2.putText(image, label_high, (x - 2, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(image, label_low, (x - 2, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)


def annotate_explainability_panel(
    cam_overlay: np.ndarray,
    conf_overlay: np.ndarray,
    language_label: str = "Español",
) -> np.ndarray:
    """Compose the side-by-side explainability figure with titles + colorbars.

    Ciclo 5.8 (fix P): the bare side-by-side panel from Ciclo 5 was ambiguous
    (no titles, no scale). This wraps each subpanel in a header strip with
    the title and a thin vertical colorbar on the right edge showing the
    intensity scale. Bilingual via i18n.

    Args:
        cam_overlay: (H, W, 3) Grad-CAM heatmap blended with the radiograph.
        conf_overlay: (H, W, 3) confidence map blended with the radiograph.
        language_label: 'Español' or 'English' (from the UI radio).

    Returns:
        (H + header, 2*W + 2*colorbar_margin, 3) uint8 panel ready for Gradio.
    """
    lang = label_to_lang(language_label)
    h, w = cam_overlay.shape[:2]
    header_h = 32           # height of the title strip
    bar_w = 18              # width of the colorbar
    margin = 38             # space to the right of each panel for bar + labels
    panel_w = w + margin

    def build_subpanel(image: np.ndarray, title: str, cmap_name: str) -> np.ndarray:
        sub = np.zeros((h + header_h, panel_w, 3), dtype=np.uint8)
        sub[:header_h] = (30, 30, 30)        # dark title strip
        sub[header_h:, :w] = image
        cv2.putText(sub, title, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
        _draw_colorbar_strip(
            sub, x=w + 8, y=header_h + 14,
            width=bar_w, height=h - 28,
            cmap_name=cmap_name, lang=lang,
        )
        return sub

    left = build_subpanel(cam_overlay, t("explain_title_gradcam", lang), "jet")
    right = build_subpanel(conf_overlay, t("explain_title_confidence", lang), "RdYlGn")
    return np.concatenate([left, right], axis=1)


def preview_rotation_for_display(original: np.ndarray, deg: float) -> np.ndarray:
    """Rotate the stashed ORIGINAL image for live preview (Ciclo 5.6).

    Called by the Gradio slider.change handler so the clinician sees the
    rotation result before committing 10s of CPU to Analyze. Operates on
    the original (stashed in gr.State at upload time), NOT on whatever is
    currently displayed — that prevents accumulated double-rotations as
    the slider drags through intermediate values.

    Args:
        original: the image as uploaded by the user; None when nothing
            has been uploaded yet.
        deg: current slider value in degrees.

    Returns:
        Rotated image, or None when `original` is None (slider moved
        before any upload). The Gradio Image component accepts None and
        clears the display, which is the right thing here.
    """
    if original is None:
        return None
    return rotate_image_for_analysis(original, deg)


def build_results_text(
    cobb_binary: dict | None,
    cobb_multiclass: dict | None,
    vertebrae_detected: list | None = None,
    coverage_info: dict | None = None,
    orientation_info: dict | None = None,
    language: str = DEFAULT_LANG,
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
    lang = language  # short alias used in t(...) calls below

    lines: list[str] = []

    # -------------------------------------------------------------------- COBB
    if curves:
        lines.append(t("cobb_block_header_curves", lang))
        for i, c in enumerate(curves):
            if i == 0:
                label = t("principal_label", lang)
            elif i == 1:
                label = t("secondary_label", lang)
            else:
                label = t("curve_n_label", lang).format(n=i + 1)
            up = c.get("upper_vertebra") or "?"
            lo = c.get("lower_vertebra") or "?"
            direction = c.get("direction", "unknown")
            conv = {
                "right": t("convex_right", lang),
                "left": t("convex_left", lang),
                "neutral": t("convex_neutral", lang),
            }.get(direction, t("convex_unknown", lang))
            lines.append(
                f"{label} {c['cobb_angle_deg']:5.1f} deg  ({up} - {lo}, {conv})"
            )
    elif binary_ok:
        # Binary succeeded but no curves passed the noise floor (straight spine).
        lines.append(t("cobb_block_header_simple", lang))
        lines.append(
            f"{t('binary_method', lang)} {cobb_binary['cobb_angle_deg']:5.1f} deg  "
            f"{t('binary_no_curves', lang)}"
        )
    elif cobb_binary:
        lines.append(t("cobb_block_header_simple", lang))
        lines.append(
            f"{t('binary_error', lang)} {cobb_binary.get('error', 'unknown')}"
        )

    # ----- Multiclass fallback only (Ciclo 5.7: cross-check block removed) -----
    # The CROSS-CHECK binary vs multiclass block confused users (binary 4 deg vs
    # multi 90 deg looks like a contradiction without context), and Elvis asked
    # to keep the multiclass strictly as a backstage helper for label transfer
    # and green-box drawing. The ONLY user-visible multiclass output left is
    # the fallback line below, which fires only when the binary method failed
    # entirely — in that case the multiclass is genuinely the best signal.
    if multi_ok and not binary_ok:
        upper = cobb_multiclass.get("upper_end_vertebra", "N/A")
        lower = cobb_multiclass.get("lower_end_vertebra", "N/A")
        lines.append(
            f"{t('multi_fallback_prefix', lang)} "
            f"{cobb_multiclass['cobb_angle_deg']:5.1f} deg  "
            f"(Upper={upper}, Lower={lower})"
        )
    elif cobb_multiclass and not multi_ok:
        lines.append(
            f"{t('multi_error_prefix', lang)} {cobb_multiclass.get('error', 'unknown')}"
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
        lines.append("\n" + t("coverage_header", lang))
        if upper_v and lower_v:
            lines.append(
                f"{t('coverage_covers', lang)} {upper_v} - {lower_v}  "
                f"({n_v} {t('coverage_of', lang)} ~{n_exp} "
                f"{t('coverage_vertebrae_word', lang)}, ~{ratio_pct:.0f}%)"
            )
        else:
            lines.append(
                t("coverage_no_names", lang).format(pct=f"{ratio_pct:.0f}")
            )
        below = coverage_info.get("vertebrae_below_range") or []
        above = coverage_info.get("vertebrae_above_range") or []
        if below:
            rng = f"{below[0]}-{below[-1]}" if len(below) > 1 else below[0]
            lines.append(
                f"{t('warning_lower_spine_prefix', lang)} ({rng}) "
                f"{t('warning_not_segmented_suffix', lang)}"
            )
        elif above:
            rng = f"{above[0]}-{above[-1]}" if len(above) > 1 else above[0]
            lines.append(
                f"{t('warning_upper_spine_prefix', lang)} ({rng}) "
                f"{t('warning_not_segmented_suffix', lang)}"
            )
        else:
            lines.append(t("warning_partial_generic", lang))

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
        lines.append("\n" + t("rotation_header", lang))
        lines.append(
            t("rotation_line_1", lang).format(tilt=tilt_abs, threshold=threshold)
        )
        lines.append(t("rotation_line_2", lang))
        lines.append(t("rotation_line_3", lang))

    # -------------------------------------------------------------- ASSESSMENT
    # Severity uses the LARGEST curve detected by the binary method. The
    # binary method is the source of truth on our data (MAE 23 deg, r=0.66,
    # vs multiclass MAE 26-45 deg with negative correlation in the worst case).
    assessment_source_key = None  # which header to use
    angle = None
    if curves:
        assessment_source_key = "assessment_header_principal"
        angle = curves[0]["cobb_angle_deg"]
    elif binary_ok:
        assessment_source_key = "assessment_header_binary"
        angle = cobb_binary["cobb_angle_deg"]
    elif multi_ok:
        assessment_source_key = "assessment_header_fallback"
        angle = cobb_multiclass["cobb_angle_deg"]

    if assessment_source_key is not None:
        # Ciclo 5.3 fix F: if coverage is partial AND the angle is ~0, the
        # binary likely missed enough spine that "Normal" would be misleading.
        # Flag it as inconclusive so the user knows to investigate instead.
        if coverage_is_partial and angle < 1.0:
            assessment = t("assessment_inconclusive", lang)
        elif angle < 10:
            assessment = t("assessment_normal", lang)
        elif angle < 25:
            assessment = t("assessment_mild", lang)
        elif angle < 40:
            assessment = t("assessment_moderate", lang)
        else:
            assessment = t("assessment_severe", lang)
        lines.append("\n" + t(assessment_source_key, lang))
        lines.append(assessment)

        if curves and len(curves) > 1:
            n = len(curves)
            shape = {
                2: t("shape_double", lang),
                3: t("shape_triple", lang),
            }.get(n, t("shape_n_curves", lang).format(n=n))
            lines.append(
                f"{t('curves_total_prefix', lang)} {n} ({shape})"
            )

    # ---------------------------------------------------- Vertebrae list
    if vertebrae_detected:
        lines.append(
            f"\n{t('vertebrae_header_prefix', lang)} "
            f"({len(vertebrae_detected)}) ==="
        )
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

    def predict(input_image, language_label="Español"):
        """Process the (already-rotated) radiograph shown in the UI.

        Ciclo 5.6: the rotation is applied LIVE in the UI via the slider
        + 5 quick buttons. By the time the user clicks Analyze, the
        displayed `input_image` is already rotated, so predict() just
        delegates to the pipeline.

        Ciclo 5.7: `language_label` carries the user's selection from
        the EN/ES radio ("Español" or "English"); we resolve it to a
        lang code and pass it to build_results_text.
        """
        lang = label_to_lang(language_label)
        if input_image is None:
            return None, None, None, None, t("no_image", lang)

        if pipeline is None:
            return None, None, None, None, t("no_model", lang)

        # Convert to RGB if needed
        if input_image.ndim == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        elif input_image.shape[2] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

        # Run prediction (image is already rotated by the UI preview pipeline)
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
            language=lang,
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

                    # Ciclo 5.8: pass the predicted spine mask so Grad-CAM and
                    # Confidence Map only paint INSIDE the detected column.
                    # Off-spine pixels stay as the grayscale radiograph, which
                    # is the natural baseline the clinician already saw.
                    pred_mask = results.get("binary_mask")
                    if pred_mask is not None and pred_mask.shape != img_resized.shape[:2]:
                        # Predict masks are 512x512; img_resized is also 512x512,
                        # so this is usually a no-op. Defensive only.
                        pred_mask = cv2.resize(
                            pred_mask.astype(np.uint8),
                            (img_resized.shape[1], img_resized.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    # Grad-CAM (masked + percentile-clipped inside the helper)
                    gradcam = generate_gradcam(
                        model_for_explain, input_t, model_name_explain,
                        prediction_mask=pred_mask,
                    )
                    img_float = img_resized.astype(np.float32) / 255.0
                    cam_overlay = show_cam_on_image(img_float, gradcam, use_rgb=True)

                    # Confidence map (also masked by the predicted spine)
                    task = "binary" if pipeline.binary_model is not None else "multiclass"
                    confidence = generate_confidence_map(
                        model_for_explain, input_t, task,
                        prediction_mask=pred_mask,
                    )
                    conf_colored = plt.cm.RdYlGn(confidence)[:, :, :3]
                    conf_colored = (conf_colored * 255).astype(np.uint8)

                    # Blend the confidence colormap with the original radiograph
                    # outside the spine — without this, the masked-to-zero region
                    # paints saturated red (the RdYlGn 0-value), which looks like
                    # "low confidence everywhere" instead of "not evaluated here".
                    if pred_mask is not None:
                        outside = (pred_mask == 0)[..., None]
                        bg = img_resized.astype(np.uint8)
                        conf_colored = np.where(outside, bg, conf_colored).astype(np.uint8)
                        # Same trick for the CAM overlay: outside the spine,
                        # restore the original image so the user keeps anatomical
                        # context.
                        cam_overlay = np.where(outside, bg, cam_overlay).astype(np.uint8)

                    explainability_img = annotate_explainability_panel(
                        cam_overlay, conf_colored, language_label,
                    )
            except Exception as e:
                print(f"Explainability error: {e}")

        return binary_overlay, multiclass_overlay, cobb_vis, explainability_img, results_text

    # Build Gradio interface
    with gr.Blocks(
        title="Spine Segmentation for Scoliosis Diagnosis",
        theme=gr.themes.Soft(),
    ) as app:
        # Ciclo 5.7: language radio drives both the header markdown and the
        # diagnosis text. Default = Español (project audience: U. Andes,
        # Colombia). The bilingual label ("Idioma / Language") makes the
        # toggle discoverable for English speakers landing on the page.
        language_radio = gr.Radio(
            choices=["Español", "English"],
            value="Español",
            label="Idioma / Language",
            interactive=True,
        )
        # The Markdown header is a tracked component so language_radio.change
        # can rewrite its content with the translated intro.
        header_md = gr.Markdown(header_markdown(DEFAULT_LANG))

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Spinal X-ray Radiograph (drag the slider to rotate)",
                    type="numpy",
                    height=500,
                )
                # Ciclo 5.6: gr.State stashes the ORIGINAL upload so the
                # slider can rotate a fresh copy each time, instead of
                # accumulating rotations on whatever happens to be shown.
                original_image_state = gr.State(value=None)

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

        # ----- Ciclo 5.6 live preview wiring -----
        # The original-image state preserves what the user uploaded so the
        # slider can rotate a fresh copy each frame (no accumulated double-
        # rotation). Three event paths drive the preview:
        #   (a) input_image.upload  -> stash original + reset slider to 0
        #   (b) rotation_slider.change -> rotate stashed original -> display
        #   (c) quick rotation buttons -> update slider + display in one go

        def _on_upload(img):
            """Stash the upload, reset slider to 0 so the new image shows
            unrotated. Returns (state, slider) — UPLOAD does not fire when
            the preview writes back to input_image, so this only runs for
            actual user uploads (no loop)."""
            return img, 0.0

        input_image.upload(
            fn=_on_upload,
            inputs=[input_image],
            outputs=[original_image_state, rotation_slider],
        )

        rotation_slider.change(
            fn=preview_rotation_for_display,
            inputs=[original_image_state, rotation_slider],
            outputs=[input_image],
        )

        def _rotate_by(current_slider, original, delta):
            """Quick-button handler: compute the new slider value, rotate
            the original by it, return BOTH so the slider widget and the
            displayed image update in one Gradio round-trip. More robust
            than relying on the slider.change event to fire after a
            programmatic value update."""
            new_slider = float(max(-180.0, min(180.0, (current_slider or 0.0) + delta)))
            rotated = preview_rotation_for_display(original, new_slider)
            return new_slider, rotated

        btn_rot_minus_90.click(
            fn=lambda s, o: _rotate_by(s, o, -90),
            inputs=[rotation_slider, original_image_state],
            outputs=[rotation_slider, input_image],
        )
        btn_rot_minus_5.click(
            fn=lambda s, o: _rotate_by(s, o, -5),
            inputs=[rotation_slider, original_image_state],
            outputs=[rotation_slider, input_image],
        )
        btn_rot_reset.click(
            fn=lambda o: (0.0, o),  # slider=0, display the un-rotated original
            inputs=[original_image_state],
            outputs=[rotation_slider, input_image],
        )
        btn_rot_plus_5.click(
            fn=lambda s, o: _rotate_by(s, o, 5),
            inputs=[rotation_slider, original_image_state],
            outputs=[rotation_slider, input_image],
        )
        btn_rot_plus_90.click(
            fn=lambda s, o: _rotate_by(s, o, 90),
            inputs=[rotation_slider, original_image_state],
            outputs=[rotation_slider, input_image],
        )

        predict_btn.click(
            fn=predict,
            # input_image is already-rotated by the preview pipeline;
            # language_radio is the EN/ES selector for the diagnosis report.
            inputs=[input_image, language_radio],
            outputs=[binary_output, multi_output, cobb_output, explain_output, results_text],
        )

        # Ciclo 5.7: language toggle updates the header markdown live. The
        # diagnosis report itself is updated the next time the user presses
        # Analyze — re-running the model just to retranslate fixed text would
        # be wasteful (10s for ~200 bytes of string difference).
        language_radio.change(
            fn=lambda lbl: header_markdown(label_to_lang(lbl)),
            inputs=[language_radio],
            outputs=[header_md],
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
