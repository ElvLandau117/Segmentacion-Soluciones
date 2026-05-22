"""
test_app_smoke.py — verify the Gradio app can be constructed.

We DON'T launch a server — just build the Blocks object. That alone
catches a huge class of bugs (broken imports, missing components,
wrong event wiring, type errors in markdown templates).
"""

from __future__ import annotations

import gradio as gr


def test_app_builds_without_checkpoints():
    """create_app() must return a gr.Blocks even when no checkpoints exist.
    The UI shows up; predictions fail gracefully with a user-visible message."""
    from spine_segmentation.deployment.app import create_app

    app = create_app(
        binary_checkpoint=None,
        multiclass_checkpoint=None,
    )

    assert isinstance(app, gr.Blocks)
    assert app.title is not None


def test_app_main_shim_imports():
    """The app/main.py shim must expose main() exactly as the package does."""
    import app.main as shim
    from spine_segmentation.deployment.app import main as real_main
    assert shim.main is real_main


def test_disclaimer_visible_in_app_markdown():
    """Regulatory requirement: the medical disclaimer must appear in the UI."""
    from spine_segmentation.deployment.app import create_app
    from spine_segmentation.config import MEDICAL_DISCLAIMER

    app = create_app(binary_checkpoint=None, multiclass_checkpoint=None)

    # Walk the block tree and collect every markdown string
    markdown_blocks = [
        getattr(child, "value", "") or ""
        for child in app.blocks.values()
        if isinstance(child, gr.Markdown)
    ]
    combined = "\n".join(markdown_blocks)
    assert MEDICAL_DISCLAIMER in combined, (
        "MEDICAL_DISCLAIMER not found in any Markdown block of the app"
    )


# ----------------------------------------------------------------------------
# build_results_text — dual-Cobb panel rendering (Ciclo 5)
# ----------------------------------------------------------------------------

def test_build_results_text_uses_binary_for_assessment():
    """When both methods succeed, the Assessment severity must come from the
    binary Cobb (more robust on our data: MAE 23 deg vs 26-45 deg multiclass).
    Pinning this regression-prevents the old behavior where Assessment came
    from the noisy multiclass and false-labelled scoliosis cases as Normal.
    Ciclo 5.7: force language='en' so the assertions match English strings."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 30.0},
        cobb_multiclass={
            "success": True, "cobb_angle_deg": 5.0,
            "upper_end_vertebra": "T6", "lower_end_vertebra": "T12",
        },
        language="en",
    )

    assert "ASSESSMENT (based on Binary)" in text
    # 30 deg falls in Moderate (25-40), not Normal (<10). Without the binary
    # source, the assessment would say Normal (because multi is 5.0).
    assert "Moderate" in text
    assert "Normal" not in text


def test_build_results_text_no_longer_emits_cross_check_block():
    """Ciclo 5.7 regression pin: when both binary and multiclass succeed, the
    diagnosis text must NOT contain the old CROSS-CHECK / CONCORDANCIA block.
    Elvis asked to remove it — the multiclass cobb (often 90 deg degenerate
    on our data) looked contradictory next to the binary cobb and confused
    users. Multiclass is now strictly a backstage helper for label transfer."""
    from spine_segmentation.deployment.app import build_results_text

    # Case where 5.2-5.6 would have shown CONCORDANCIA: both methods succeed.
    # Force language='en' so we can assert exact English strings; the i18n
    # path is exercised by the dedicated EN/ES tests above.
    text_high = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 20.0},
        cobb_multiclass={"success": True, "cobb_angle_deg": 21.5,
                         "upper_end_vertebra": "T5", "lower_end_vertebra": "T11"},
        language="en",
    )
    assert "CONCORDANCIA" not in text_high
    assert "CROSS-CHECK" not in text_high
    assert "Multiclass:" not in text_high
    # But the binary Cobb and the assessment still appear.
    assert "20.0" in text_high or "20" in text_high
    assert "ASSESSMENT" in text_high

    # Significant-discrepancy case: same expectation — no block at all.
    text_disc = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 10.0},
        cobb_multiclass={"success": True, "cobb_angle_deg": 35.0,
                         "upper_end_vertebra": "T5", "lower_end_vertebra": "T11"},
        language="en",
    )
    assert "CONCORDANCIA" not in text_disc
    assert "Significant discrepancy" not in text_disc


def test_build_results_text_falls_back_to_multiclass_when_binary_fails():
    """If the binary method failed (e.g. empty mask), the Assessment must use
    multiclass as a labelled fallback so the user still gets a severity hint.
    No CONCORDANCIA block in this case (only one Cobb value)."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": False, "error": "Empty mask after cleaning"},
        cobb_multiclass={"success": True, "cobb_angle_deg": 28.0,
                         "upper_end_vertebra": "T6", "lower_end_vertebra": "L1"},
        language="en",
    )

    assert "ASSESSMENT (based on Multiclass fallback)" in text
    assert "Moderate" in text
    assert "CONCORDANCIA" not in text


def test_draw_cobb_angle_visualization_modifies_image():
    """The visualization must actually draw on the image (overlays + lines +
    text). A no-op would defeat the purpose of the cycle 5 work."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import draw_cobb_angle_visualization

    # Construct a synthetic multiclass mask with two distinct vertebra regions:
    # class 7 (=C7) at the top, class 12 (=T6) at the bottom.
    h, w = 256, 128
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:60, 30:90] = 7    # upper vertebra
    mask[180:200, 30:90] = 12  # lower vertebra

    image = np.full((h, w, 3), 128, dtype=np.uint8)  # uniform gray
    cobb_result = {
        "success": True,
        "cobb_angle_deg": 12.5,
        "upper_end_vertebra": "C7",
        "lower_end_vertebra": "T6",
    }
    # Ciclo 5.1: pass the full dict (signature changed from `cobb_binary_deg: float`
    # to `cobb_binary_result: dict` so the viz can overlay spline + inflection points)
    cobb_binary_result = {"success": True, "cobb_angle_deg": 10.0}

    vis = draw_cobb_angle_visualization(
        image=image,
        multiclass_mask=mask,
        cobb_multiclass_result=cobb_result,
        cobb_binary_result=cobb_binary_result,
    )

    assert vis.shape == image.shape
    assert vis.dtype == np.uint8
    # Must have modified pixels (overlay/lines/text). Compare to original.
    assert not np.array_equal(vis, image)


# ----------------------------------------------------------------------------
# Cobb visualization helpers — Ciclo 5.1
# ----------------------------------------------------------------------------

def test_line_intersection_handles_parallel_and_crossing():
    """The helper must return None for parallel lines and a finite point for
    crossing lines, so the orchestrator can decide between drawing the arc
    or falling back to the speedometer."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import _line_intersection

    # Two horizontal lines at different y -> parallel -> None
    assert _line_intersection((0.0, 0.0), (1.0, 0.0), (0.0, 5.0), (1.0, 0.0)) is None

    # Crossing at (1, 0): horizontal y=0 vs vertical x=1
    pt = _line_intersection((0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (0.0, 1.0))
    assert pt is not None
    assert abs(pt[0] - 1.0) < 1e-6
    assert abs(pt[1] - 0.0) < 1e-6


def test_endplate_vectors_are_perpendicular_unit_vectors():
    """For a horizontal vertebra (orientation = pi/2) the tangent must be
    horizontal and the perpendicular vertical (in image coords, y-down).
    Both must be unit vectors and orthogonal."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import _endplate_vectors

    tan, perp = _endplate_vectors(np.pi / 2)
    # tan ~= (1, 0), perp ~= (0, -1)
    assert abs(tan[0] - 1.0) < 1e-6
    assert abs(tan[1] - 0.0) < 1e-6
    assert abs(perp[0] - 0.0) < 1e-6
    assert abs(perp[1] - (-1.0)) < 1e-6
    # Orthogonality
    dot = tan[0] * perp[0] + tan[1] * perp[1]
    assert abs(dot) < 1e-6


def test_speedometer_draws_inside_the_image():
    """The bottom-left mini gauge must touch pixels in the lower portion of
    the image. This is the visual fallback for tiny Cobb angles (<8 deg) where
    the in-frame arc would be unreadable, so it has to be actually rendered."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import _draw_speedometer

    h, w = 400, 300
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_speedometer(vis, angle_deg=3.5)
    # Upper half should still be untouched
    assert vis[:h // 2, :, :].sum() == 0
    # Lower half should have non-zero pixels (the gauge)
    assert vis[h // 2:, :, :].sum() > 0


def test_binary_overlay_renders_spline_and_inflection_points():
    """When the binary result carries a fitted spline and inflection points,
    the overlay must actually paint them so the user can trace how the binary
    angle was computed (educational value + credibility)."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import _draw_binary_overlay

    h, w = 200, 100
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    spline_x = list(np.linspace(50, 50, 50))  # vertical spline at x=50
    spline_y = list(np.linspace(20, 180, 50))
    cobb_binary = {
        "success": True,
        "spline_x": spline_x,
        "spline_y": spline_y,
        "inflection_points": [(50.0, 60.0), (50.0, 140.0)],
    }
    _draw_binary_overlay(vis, cobb_binary)

    # Spline polyline should leave white-ish pixels in the central column
    central = vis[:, 48:53, :]
    assert central.sum() > 0
    # Inflection points are filled yellow circles (R=255, G=255, B=0) — the
    # function uses RGB conventions because the upstream image comes from
    # Gradio as RGB. Two circles of radius 5 -> ~150 yellow pixels.
    yellow = (vis[..., 0] > 200) & (vis[..., 1] > 200) & (vis[..., 2] < 30)
    assert yellow.sum() > 30

    # Skip-path: a failed binary result must not raise.
    _draw_binary_overlay(vis, None)
    _draw_binary_overlay(vis, {"success": False, "error": "x"})


# ----------------------------------------------------------------------------
# Explainability polish — Ciclo 5.8 (fixes O + P + Q + R)
# ----------------------------------------------------------------------------

def test_annotate_explainability_panel_adds_titles_and_colorbars():
    """The side-by-side panel must (a) be roughly 2x wider than each input
    panel (two subpanels), (b) be taller than the input (header strip), and
    (c) contain readable title pixels (white-ish text on dark header)."""
    import numpy as np
    from spine_segmentation.deployment.app import annotate_explainability_panel

    h, w = 200, 200
    cam = np.full((h, w, 3), 100, dtype=np.uint8)
    conf = np.full((h, w, 3), 50, dtype=np.uint8)

    out = annotate_explainability_panel(cam, conf, language_label="Español")

    # Two panels side by side; each gains a header strip on top.
    assert out.shape[0] > h, "panel should be taller (header strip added)"
    assert out.shape[1] > 2 * w, "panel should be ~2x width plus colorbar margins"
    # The top 32 px is the dark header. Below 32 we should see the input pixels.
    assert out[5, 5].tolist() == [30, 30, 30] or out[5, 5].sum() < 100, \
        "top strip should be dark"
    # White-ish title text somewhere in the top strip.
    top_strip = out[:32]
    bright = (top_strip > 200).all(axis=2)
    assert bright.sum() > 20, "no bright title pixels found in the header strip"


def test_annotate_explainability_panel_localizes_titles():
    """When language='English' the rendered panel should contain text pixels
    drawn from English strings (Grad-CAM, Confidence). We don't OCR; we just
    confirm both ES and EN runs produce title pixels of similar magnitude
    (i.e. neither path returned an empty / un-titled panel)."""
    import numpy as np
    from spine_segmentation.deployment.app import annotate_explainability_panel

    cam = np.full((150, 150, 3), 80, dtype=np.uint8)
    conf = np.full((150, 150, 3), 80, dtype=np.uint8)
    es = annotate_explainability_panel(cam, conf, language_label="Español")
    en = annotate_explainability_panel(cam, conf, language_label="English")

    # Both have a non-trivial number of bright pixels in the header strip.
    for img in (es, en):
        top = img[:32]
        bright = (top > 200).all(axis=2)
        assert bright.sum() > 20


def test_generate_gradcam_applies_prediction_mask():
    """When a prediction_mask is supplied, pixels outside the mask must be 0
    in the returned Grad-CAM. This is the regression pin for fix O."""
    import numpy as np
    import torch
    from unittest.mock import patch, MagicMock
    from spine_segmentation.evaluation.explainability import generate_gradcam

    # Patch GradCAM so we can return a known heatmap without spinning up a model.
    with patch("spine_segmentation.evaluation.explainability.GradCAM") as cam_cls, \
         patch("spine_segmentation.evaluation.explainability.get_target_layer") as gtl:
        instance = MagicMock()
        instance.return_value = np.ones((1, 8, 8), dtype=np.float32)  # cam at every pixel
        cam_cls.return_value = instance
        gtl.return_value = MagicMock()

        # Mask: only the central 4x4 is "spine".
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        result = generate_gradcam(
            model=MagicMock(),
            input_tensor=torch.zeros((1, 3, 8, 8)),
            model_name="unet_resnet50",
            prediction_mask=mask,
        )

    # Outside the mask: must be 0. Inside: nonzero.
    assert result.shape == (8, 8)
    assert result[0, 0] == 0.0
    assert result[7, 7] == 0.0
    assert (result[2:6, 2:6] > 0).all()


def test_generate_confidence_map_applies_prediction_mask():
    """Same as above for the confidence map: outside-spine pixels are 0."""
    import numpy as np
    import torch
    from unittest.mock import MagicMock
    from spine_segmentation.evaluation.explainability import generate_confidence_map

    # Fake model that always returns logits = +5 (high confidence).
    class FakeModel:
        def eval(self): pass
        def __call__(self, x):
            return torch.full((1, 1, 8, 8), 5.0)
        def parameters(self):
            return iter([torch.zeros(1)])

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 3:5] = 1
    conf = generate_confidence_map(
        FakeModel(), torch.zeros((1, 3, 8, 8)), task="binary",
        prediction_mask=mask,
    )
    assert conf.shape == (8, 8)
    assert conf[0, 0] == 0.0  # outside spine -> zeroed
    assert conf[3, 3] > 0.9   # inside spine, sigmoid(5) ~= 0.993


# ----------------------------------------------------------------------------
# i18n module + ES/EN toggle — Ciclo 5.7 (fix N)
# ----------------------------------------------------------------------------

def test_t_returns_spanish_by_default():
    """The Spanish strings are the default — calling t(key) without a lang
    must return the ES variant."""
    from spine_segmentation.deployment.i18n import t
    assert "Curva principal" in t("principal_label")
    assert "convexidad derecha" in t("convex_right")


def test_t_returns_english_when_lang_is_en():
    """Passing lang='en' must return the English string for the same key."""
    from spine_segmentation.deployment.i18n import t
    assert "Principal curve" in t("principal_label", "en")
    assert "convex right" in t("convex_right", "en")


def test_t_falls_back_to_key_when_key_missing():
    """A missing key returns the key itself — visible placeholder so the
    developer notices the gap during testing instead of failing silently."""
    from spine_segmentation.deployment.i18n import t
    assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"
    assert t("nonexistent_key_xyz", "en") == "nonexistent_key_xyz"


def test_t_falls_back_to_spanish_when_lang_unknown():
    """If the user passes an unsupported lang, fall back to Spanish (default)
    instead of crashing — defensive behaviour against UI bugs."""
    from spine_segmentation.deployment.i18n import t
    assert "Curva principal" in t("principal_label", "fr")  # French not supported


def test_label_to_lang_maps_radio_choices():
    """The Gradio radio shows 'Español' / 'English'; the predict closure
    converts those labels back to short lang codes used everywhere else."""
    from spine_segmentation.deployment.i18n import label_to_lang
    assert label_to_lang("Español") == "es"
    assert label_to_lang("English") == "en"
    # Unknown labels fall back to ES (default).
    assert label_to_lang("Klingon") == "es"


def test_build_results_text_renders_in_spanish():
    """When language='es' (or default), the diagnosis text contains Spanish
    UI strings — Curva principal, convexidad, Escoliosis leve, etc."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={
            "success": True, "cobb_angle_deg": 18.0,
            "curves": [{
                "cobb_angle_deg": 18.0,
                "upper_vertebra": "T5", "lower_vertebra": "T12",
                "direction": "right", "rank": 1,
            }],
        },
        cobb_multiclass=None,
        language="es",
    )
    assert "Curva principal" in text
    assert "convexidad derecha" in text
    assert "Escoliosis leve" in text
    assert "EVALUACION" in text
    # English strings must NOT leak through.
    assert "Principal curve" not in text
    assert "Mild scoliosis" not in text


def test_build_results_text_renders_in_english():
    """When language='en', everything is in English."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={
            "success": True, "cobb_angle_deg": 18.0,
            "curves": [{
                "cobb_angle_deg": 18.0,
                "upper_vertebra": "T5", "lower_vertebra": "T12",
                "direction": "right", "rank": 1,
            }],
        },
        cobb_multiclass=None,
        language="en",
    )
    assert "Principal curve" in text
    assert "convex right" in text
    assert "Mild scoliosis" in text
    assert "ASSESSMENT" in text
    # Spanish strings must NOT leak through.
    assert "Curva principal" not in text
    assert "Escoliosis leve" not in text


def test_header_markdown_has_both_languages():
    """The Markdown intro must exist in both languages and differ — sanity
    that the toggle has something meaningful to switch to."""
    from spine_segmentation.deployment.i18n import header_markdown
    es = header_markdown("es")
    en = header_markdown("en")
    assert "Segmentacion Espinal" in es or "Diagnostico" in es
    assert "Spine Segmentation" in en or "Diagnosis" in en
    assert es != en


def test_explain_markdown_has_both_languages():
    """The Explainability-tab markdown also exists in ES and EN (Ciclo 5.8),
    and contains the clinical reading guidance ("Como leerlo" / "How to read")."""
    from spine_segmentation.deployment.i18n import explain_markdown
    es = explain_markdown("es")
    en = explain_markdown("en")
    assert "Grad-CAM" in es
    assert "Grad-CAM" in en
    # Spanish has the reading-guide section header.
    assert "leerlo" in es.lower() or "como" in es.lower()
    # English has the equivalent.
    assert "read it" in en.lower() or "how to" in en.lower()
    assert es != en


# ----------------------------------------------------------------------------
# Live rotation preview — Ciclo 5.6 (fix L)
# ----------------------------------------------------------------------------

def test_preview_rotation_for_display_handles_none():
    """When no image has been uploaded yet, dragging the slider must not
    crash. The helper returns None and the Image component clears."""
    from spine_segmentation.deployment.app import preview_rotation_for_display
    assert preview_rotation_for_display(None, 0) is None
    assert preview_rotation_for_display(None, 25) is None
    assert preview_rotation_for_display(None, -90) is None


def test_preview_rotation_for_display_returns_rotated():
    """The preview must actually rotate the original (delegate to
    rotate_image_for_analysis). At deg=0 the output is the original
    object; at deg=90 it differs (rotation actually applied)."""
    import numpy as np
    from spine_segmentation.deployment.app import preview_rotation_for_display

    orig = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    # Zero rotation: identity short-circuit inside rotate_image_for_analysis.
    assert preview_rotation_for_display(orig, 0.0) is orig
    # 90 deg rotation: same shape but different pixels.
    rotated = preview_rotation_for_display(orig, 90.0)
    assert rotated.shape == orig.shape
    assert rotated.dtype == orig.dtype
    assert not np.array_equal(rotated, orig)


def test_predict_callback_no_longer_takes_rotation_deg():
    """Regression pin: predict() closure must NOT have a `rotation_deg`
    parameter (Ciclo 5.6 simplification — live preview means the displayed
    image is already rotated by the time Analyze fires). Ciclo 5.7 added
    a language_label parameter, so the expected arg count is 2."""
    import inspect
    from spine_segmentation.deployment.app import create_app

    app = create_app(binary_checkpoint=None, multiclass_checkpoint=None)
    found_predict = None
    for fn_def in getattr(app, "fns", {}).values() if hasattr(app, "fns") else []:
        fn = getattr(fn_def, "fn", None)
        if fn is None:
            continue
        if getattr(fn, "__name__", "") == "predict":
            found_predict = fn
            break
    if found_predict is not None:
        sig = inspect.signature(found_predict)
        # Two params: input_image + language_label. NO rotation_deg.
        assert "rotation_deg" not in sig.parameters
        assert "input_image" in sig.parameters
    from spine_segmentation.deployment.app import preview_rotation_for_display
    assert callable(preview_rotation_for_display)


# ----------------------------------------------------------------------------
# Manual rotation control — Ciclo 5.5 (fix K)
# ----------------------------------------------------------------------------

def test_rotate_image_for_analysis_zero_is_identity():
    """Slider sitting at 0 (the default) must return the input array
    untouched — both equal magnitudes (|deg| < 0.5) and exact zero. We
    do NOT want every Analyze click to pay for a no-op warpAffine that
    introduces sub-pixel interpolation noise into the binary segmenter."""
    import numpy as np
    from spine_segmentation.deployment.app import rotate_image_for_analysis

    img = np.random.default_rng(0).integers(0, 256, (64, 48, 3), dtype=np.uint8)
    out_zero = rotate_image_for_analysis(img, 0.0)
    out_tiny = rotate_image_for_analysis(img, 0.3)  # below the 0.5 deg deadband
    out_neg_tiny = rotate_image_for_analysis(img, -0.4)

    assert out_zero is img, "zero rotation should short-circuit, not warp"
    assert out_tiny is img
    assert out_neg_tiny is img


def test_rotate_image_for_analysis_90_swaps_axes():
    """Rotating by +90 deg (CCW in cv2 convention) must turn a vertical
    line into a horizontal one. We verify on a (200, 200, 3) canvas with
    a single vertical white stripe at x=100; after the rotation the
    bright row should be near y=100 instead of a bright column."""
    import numpy as np
    from spine_segmentation.deployment.app import rotate_image_for_analysis

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, 98:103, :] = 255  # vertical bright stripe
    rotated = rotate_image_for_analysis(img, 90.0)

    # Before: column 100 is bright, row sums are uniform.
    col_brightness_before = img.sum(axis=0).sum(axis=-1)
    assert col_brightness_before[100] > col_brightness_before[0]

    # After: row 100 is bright, col sums are uniform.
    row_brightness_after = rotated.sum(axis=1).sum(axis=-1)
    assert row_brightness_after[100] > row_brightness_after[0]
    # And the column at x=100 is no longer the brightest one.
    col_brightness_after = rotated.sum(axis=0).sum(axis=-1)
    assert col_brightness_after[100] < row_brightness_after[100]


def test_rotate_image_for_analysis_handles_none():
    """None input must pass through unchanged so the Gradio predict()
    closure does not have to special-case the "no image uploaded" path."""
    from spine_segmentation.deployment.app import rotate_image_for_analysis
    assert rotate_image_for_analysis(None, 0.0) is None
    assert rotate_image_for_analysis(None, 25.0) is None


# ----------------------------------------------------------------------------
# Orientation detection — Ciclo 5.4 (fix G)
# ----------------------------------------------------------------------------

def test_compute_orientation_info_detects_tilt():
    """A skeleton along a 30 deg line from vertical must be flagged as tilted.
    This is the key behaviour for N_61 (Normal radiograph rotated in frame
    that fooled cobb_from_binary into reporting 4 phantom curves)."""
    import numpy as np
    from spine_segmentation.evaluation.orientation import (
        compute_orientation_info,
    )

    # Build skeleton points along a line tilted 30 deg from vertical.
    # In image coords with y-down, "tilted 30 deg" means dx/dy = tan(30).
    # For y in [0, 400], x = 250 + tan(30deg) * (y - 200).
    n = 100
    ys = np.linspace(0, 400, n)
    xs = 250.0 + np.tan(np.radians(30.0)) * (ys - 200)
    pts = np.column_stack([xs, ys])

    info = compute_orientation_info(pts)
    assert info["success"] is True
    assert info["is_tilted"] is True
    # SVD principal axis vs vertical -> ~30 deg. Allow 2 deg slack for
    # numerical / discretisation noise.
    assert abs(info["tilt_abs_deg"] - 30.0) < 2.0
    assert info["n_points"] == n


def test_compute_orientation_info_vertical_spine():
    """A perfectly vertical skeleton must NOT be flagged as tilted. Small
    horizontal jitter (real spines wobble a few px) is still vertical."""
    import numpy as np
    from spine_segmentation.evaluation.orientation import (
        compute_orientation_info,
    )

    n = 100
    ys = np.linspace(0, 400, n)
    # Jitter of +/- 1 px in x — well within "vertical".
    rng = np.random.default_rng(seed=42)
    xs = 250.0 + rng.uniform(-1.0, 1.0, size=n)
    pts = np.column_stack([xs, ys])

    info = compute_orientation_info(pts)
    assert info["success"] is True
    assert info["is_tilted"] is False
    assert info["tilt_abs_deg"] < 2.0


def test_build_results_text_emits_rotation_warning_when_tilted():
    """When orientation_info reports is_tilted=True, the diagnosis text must
    include the ROTATION WARNING block with the measured angle, threshold,
    and the 'multiclass is rotation-invariant' guidance. Without tilt, the
    block is absent."""
    from spine_segmentation.deployment.app import build_results_text

    # Tilted -> block visible (force language='en' for stable English assertions)
    text_tilted = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 31.8,
                     "curves": [{"cobb_angle_deg": 31.8, "upper_vertebra": "L3",
                                 "lower_vertebra": "L4", "direction": "right",
                                 "rank": 1}]},
        cobb_multiclass={"success": True, "cobb_angle_deg": 0.6,
                         "upper_end_vertebra": "C5", "lower_end_vertebra": "L4"},
        orientation_info={"success": True, "tilt_deg": 18.4, "tilt_abs_deg": 18.4,
                          "is_tilted": True, "threshold_deg": 12.0, "n_points": 200},
        language="en",
    )
    assert "=== ROTATION WARNING ===" in text_tilted
    assert "18.4 deg" in text_tilted
    assert "threshold 12 deg" in text_tilted
    # Ciclo 5.7: copy advises using the rotation slider (not "trust multi").
    assert "rotation slider" in text_tilted.lower() or "binary" in text_tilted.lower()

    # Not tilted -> no block
    text_ok = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 5.0},
        cobb_multiclass=None,
        orientation_info={"success": True, "tilt_deg": 2.0, "tilt_abs_deg": 2.0,
                          "is_tilted": False, "threshold_deg": 12.0, "n_points": 200},
        language="en",
    )
    assert "ROTATION WARNING" not in text_ok

    # No orientation_info -> no block (back-compat path)
    text_none = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 5.0},
        cobb_multiclass=None,
        language="en",
    )
    assert "ROTATION WARNING" not in text_none


def test_compute_orientation_info_handles_empty_or_collinear():
    """Robustness: None input, empty array, or <3 points must NOT crash and
    must return success=False so callers can skip the warning gracefully."""
    import numpy as np
    from spine_segmentation.evaluation.orientation import (
        compute_orientation_info,
    )

    # None
    assert compute_orientation_info(None)["success"] is False

    # Empty
    assert compute_orientation_info(np.zeros((0, 2)))["success"] is False

    # 2 points (insufficient)
    assert compute_orientation_info(np.array([[10, 20], [30, 40]]))["success"] is False

    # Degenerate: 5 identical points (zero spread)
    same = np.tile(np.array([[100, 100]], dtype=float), (5, 1))
    assert compute_orientation_info(same)["success"] is False


# ----------------------------------------------------------------------------
# Label dedup + anti-overlap in viz — Ciclo 5.4 (fix I)
# ----------------------------------------------------------------------------

def test_draw_single_cobb_curve_dedupes_shared_vertebra_label():
    """When two curves share a vertebra (e.g. T9 is the lower end of the
    principal AND the upper end of the secondary), the second call must NOT
    add a duplicate label for that shared vertebra. The accumulators are
    mutated in place so the caller can drive the dedup across curves."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import _draw_single_cobb_curve

    h, w = 400, 200
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Three vertebrae regions; T9 is shared between the two curves below.
    mask[40:60, 80:140] = 6
    mask[180:200, 80:140] = 9   # T9 (shared)
    mask[330:350, 80:140] = 14

    def vert(class_id, name, top, bottom):
        return {
            "class_id": class_id,
            "name": name,
            "centroid_x": 110.0,
            "centroid_y": (top + bottom) / 2.0,
            "bbox": (top, 80, bottom, 140),
            "orientation": 0.0,  # horizontal endplate
        }

    upper1 = vert(6, "T6", 40, 60)
    lower1 = vert(9, "T9", 180, 200)
    upper2 = vert(9, "T9", 180, 200)
    lower2 = vert(14, "L2", 330, 350)

    labeled_vertebrae: set = set()
    placed_label_rects: list = []

    # First curve: T6 -> T9. Both names get added.
    _draw_single_cobb_curve(
        vis, mask, upper1, lower1, cobb_deg=25.0, color=(255, 0, 0),
        draw_speedometer_if_small=False, label_prefix="[Principal] ",
        labeled_vertebrae=labeled_vertebrae,
        placed_label_rects=placed_label_rects,
    )
    assert "T6" in labeled_vertebrae
    assert "T9" in labeled_vertebrae
    rects_after_first = len(placed_label_rects)
    assert rects_after_first == 2

    # Second curve: T9 -> L2. T9 is dedup'd; only L2 adds a new label.
    _draw_single_cobb_curve(
        vis, mask, upper2, lower2, cobb_deg=12.0, color=(255, 100, 200),
        draw_speedometer_if_small=False, label_prefix="[Secundaria] ",
        labeled_vertebrae=labeled_vertebrae,
        placed_label_rects=placed_label_rects,
    )
    assert "L2" in labeled_vertebrae
    # T9 is still tracked (set behaviour) but no extra rect was added for it.
    assert len(placed_label_rects) == rects_after_first + 1


def test_draw_single_cobb_curve_shifts_overlapping_labels_down():
    """If a non-shared vertebra's label position would collide with a
    previously placed label (rare but possible at small image sizes), the
    new label must be shifted vertically until it fits or the function
    accepts the original spot if there's no room."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import _draw_single_cobb_curve

    h, w = 400, 200
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:60, 80:140] = 6
    mask[60:80, 80:140] = 7   # right below T6 -> label would collide

    def vert(cid, name, top, bottom):
        return {
            "class_id": cid, "name": name,
            "centroid_x": 110.0, "centroid_y": (top + bottom) / 2.0,
            "bbox": (top, 80, bottom, 140), "orientation": 0.0,
        }

    upper = vert(6, "T6", 40, 60)
    lower = vert(7, "T7", 60, 80)
    labeled: set = set()
    rects: list = []

    _draw_single_cobb_curve(
        vis, mask, upper, lower, 10.0, (255, 0, 0),
        draw_speedometer_if_small=False, label_prefix="",
        labeled_vertebrae=labeled, placed_label_rects=rects,
    )
    assert len(rects) == 2
    # The two rects must not overlap each other.
    r1, r2 = rects
    from spine_segmentation.evaluation.visualize import _rects_overlap
    assert not _rects_overlap(r1, r2), "labels overlap despite shift logic"


# ----------------------------------------------------------------------------
# Degenerate-curve filtering — Ciclo 5.4 (fix H)
# ----------------------------------------------------------------------------

def test_cobb_from_binary_skips_close_inflection_points():
    """A spline that produces a real S-curve PLUS a sub-vertebra wiggle must
    yield only the real curve. The wiggle (two IPs closer than
    MIN_IP_Y_DISTANCE_PX) produced the bogus "5.9 deg T9-T9" entry on S_22.
    Synth: superimpose a wide sine on a high-freq jitter and verify the
    high-freq pairs get filtered."""
    import numpy as np
    from spine_segmentation.evaluation.cobb_angle import (
        MIN_IP_Y_DISTANCE_PX,
        cobb_from_binary,
    )

    H, W = 800, 400
    mask = np.zeros((H, W), dtype=np.uint8)
    # Big slow sine (real curves) + small fast jitter (noise that would
    # otherwise create dense extra IPs).
    for y in range(H):
        slow = 60 * np.sin(2 * np.pi * y / H * 2.0)         # 2 cycles, real
        fast = 4 * np.sin(2 * np.pi * y / 20.0)             # 40 cycles, noise
        cx = int(W / 2 + slow + fast)
        mask[y, max(0, cx - 8):min(W, cx + 8)] = 1

    # Use a smoothing low enough that the spline tries to pick up the
    # fast jitter — then verify the filter drops sub-vertebral curves.
    r = cobb_from_binary(mask, smoothing_factor=500.0)
    assert r["success"]
    # Every kept curve must have IPs farther apart than the threshold.
    for c in r.get("curves") or []:
        y_dist = abs(c["ip_upper"][1] - c["ip_lower"][1])
        assert y_dist >= MIN_IP_Y_DISTANCE_PX, (
            f"curve with y_dist={y_dist} slipped past the filter "
            f"(threshold {MIN_IP_Y_DISTANCE_PX})"
        )


def test_assign_vertebra_names_drops_same_vertebra_curves():
    """When label-transfer maps both IPs of a curve to the same multiclass
    vertebra, that curve must be REMOVED from the list (not just flagged).
    The remaining curves are re-ranked so that 'principal/secundaria' stay
    sequential. Curves without multiclass names (None==None) survive."""
    from spine_segmentation.evaluation.cobb_angle import (
        assign_vertebra_names_to_curves,
    )

    # Two curves: first ends inside T9 (both IPs near y=200), second spans
    # T5..T10 (legitimate).
    curves = [
        # Degenerate: both IPs map to T9 (centroid_y=200).
        {"ip_upper": (110.0, 195.0), "ip_lower": (110.0, 205.0),
         "cobb_angle_deg": 5.9, "rank": 1},
        # Real: maps to T5 -> T10.
        {"ip_upper": (105.0, 55.0), "ip_lower": (115.0, 250.0),
         "cobb_angle_deg": 30.0, "rank": 2},
    ]
    vertebrae = [
        {"name": "T5",  "centroid_y": 60.0},
        {"name": "T9",  "centroid_y": 200.0},
        {"name": "T10", "centroid_y": 245.0},
    ]
    out = assign_vertebra_names_to_curves(curves, vertebrae)
    # Degenerate curve removed; real one survives and is now rank=1.
    assert len(out) == 1
    assert out[0]["upper_vertebra"] == "T5"
    assert out[0]["lower_vertebra"] == "T10"
    assert out[0]["rank"] == 1
    # Mutation is in-place: caller sees the same shortened list.
    assert curves is out and len(curves) == 1


def test_assign_vertebra_names_keeps_curves_without_multiclass():
    """If multiclass is unavailable (empty list), every curve gets
    upper_vertebra=None and lower_vertebra=None. None == None should NOT
    trigger the degenerate filter — those curves still carry geometric
    info."""
    from spine_segmentation.evaluation.cobb_angle import (
        assign_vertebra_names_to_curves,
    )

    curves = [
        {"ip_upper": (10.0, 20.0), "ip_lower": (15.0, 100.0),
         "cobb_angle_deg": 12.0, "rank": 1},
    ]
    out = assign_vertebra_names_to_curves(curves, [])
    assert len(out) == 1
    assert out[0]["upper_vertebra"] is None
    assert out[0]["lower_vertebra"] is None


# ----------------------------------------------------------------------------
# Binary mask cleanup — Ciclo 5.3 (fix B)
# ----------------------------------------------------------------------------

def test_clean_binary_mask_bridges_vertical_gap():
    """A thoracic fragment + a lumbar fragment separated by a small vertical
    gap must merge into a single connected component after clean_binary_mask.
    Before Ciclo 5.3 the largest-CC step ran first and dropped the smaller
    fragment, leaving only half of the spine for the spline fit."""
    import numpy as np
    from skimage.measure import label as cc_label
    from spine_segmentation.postprocessing.morphology import clean_binary_mask

    H, W = 512, 512
    mask = np.zeros((H, W), dtype=np.uint8)
    # Thoracic fragment: rows 80-230 (height 150), cols 250-270 (width 20)
    mask[80:230, 250:270] = 1
    # Lumbar fragment: rows 250-410 (height 160), cols 250-270 — gap of 20 rows
    mask[250:410, 250:270] = 1
    raw_components = cc_label(mask).max()
    assert raw_components == 2, "fixture must start as two separate components"

    cleaned = clean_binary_mask(mask)

    # After the vertical closing inserted by fix B, both fragments should be
    # bridged into a single connected component.
    n_components = cc_label(cleaned).max()
    assert n_components == 1, (
        f"expected 1 connected component after bridging, got {n_components}"
    )
    # Total area should include both fragments plus the bridge — i.e. clearly
    # more than just the larger fragment alone (which would be the result if
    # the bridge had failed and largest-CC kept only the lumbar piece).
    assert cleaned.sum() > 160 * 20 * 1.5, (
        f"expected combined area, got only {cleaned.sum()} px (single fragment?)"
    )


# ----------------------------------------------------------------------------
# Multi-curve detection — Ciclo 5.2
# ----------------------------------------------------------------------------

def test_cobb_from_binary_detects_two_curves_on_s_shape():
    """A synthetic S-shape spine (two full sine cycles) must yield at least
    2 curves above the noise floor. Before Ciclo 5.2 this was impossible:
    cobb_from_binary used only the 2 extreme inflection points and dropped
    the rest, hiding the compensatory lumbar curve from the report."""
    import numpy as np
    from spine_segmentation.evaluation.cobb_angle import cobb_from_binary

    H, W = 800, 400
    mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        cx = int(W / 2 + 80 * np.sin(2 * np.pi * y / H * 2.0))  # 2 full cycles -> 3 IPs
        mask[y, max(0, cx - 8):min(W, cx + 8)] = 1

    r = cobb_from_binary(mask, smoothing_factor=1000.0)
    assert r["success"]
    assert len(r["curves"]) >= 2, f"expected >=2 curves, got {len(r['curves'])}"
    # Each curve carries the required keys
    for c in r["curves"]:
        assert "cobb_angle_deg" in c
        assert "ip_upper" in c and "ip_lower" in c
        assert "direction" in c
        assert c["cobb_angle_deg"] >= 3.0  # above the min_curve_deg floor
    # cobb_angle_deg back-compat: principal is the largest
    assert r["cobb_angle_deg"] == r["curves"][0]["cobb_angle_deg"]


def test_assign_vertebra_names_to_curves_label_transfer():
    """Each curve must receive `upper_vertebra` / `lower_vertebra` from the
    multiclass detection by nearest-y. This is the label-transfer that lets
    the UI say 'T5 - T12' instead of just '(?, ?)' for each curve."""
    from spine_segmentation.evaluation.cobb_angle import assign_vertebra_names_to_curves

    curves = [
        {"ip_upper": (100.0, 50.0), "ip_lower": (110.0, 200.0)},
        {"ip_upper": (110.0, 200.0), "ip_lower": (120.0, 380.0)},
    ]
    vertebrae = [
        {"name": "T5", "centroid_y": 55.0},
        {"name": "T10", "centroid_y": 195.0},
        {"name": "L4", "centroid_y": 375.0},
    ]
    enriched = assign_vertebra_names_to_curves(curves, vertebrae)
    assert enriched[0]["upper_vertebra"] == "T5"
    assert enriched[0]["lower_vertebra"] == "T10"
    assert enriched[1]["upper_vertebra"] == "T10"
    assert enriched[1]["lower_vertebra"] == "L4"

    # Empty multiclass -> None labels, no crash.
    curves2 = [{"ip_upper": (0.0, 0.0), "ip_lower": (0.0, 100.0)}]
    enriched2 = assign_vertebra_names_to_curves(curves2, [])
    assert enriched2[0]["upper_vertebra"] is None
    assert enriched2[0]["lower_vertebra"] is None


def test_build_results_text_multi_curve_layout():
    """When cobb_binary carries a 2-curve list, the text must list both with
    direction + vertebra names, mention 'S-curve', and pick the Assessment
    from the principal (largest) curve."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={
            "success": True,
            "cobb_angle_deg": 32.0,
            "curves": [
                {
                    "cobb_angle_deg": 32.0,
                    "upper_vertebra": "T5",
                    "lower_vertebra": "T12",
                    "direction": "right",
                    "rank": 1,
                },
                {
                    "cobb_angle_deg": 17.5,
                    "upper_vertebra": "T12",
                    "lower_vertebra": "L4",
                    "direction": "left",
                    "rank": 2,
                },
            ],
        },
        cobb_multiclass={
            "success": True, "cobb_angle_deg": 28.0,
            "upper_end_vertebra": "T6", "lower_end_vertebra": "L1",
        },
        language="es",
    )

    assert "Curva principal" in text
    assert "Curva secundaria" in text
    assert "T5 - T12" in text
    assert "T12 - L4" in text
    assert "convexidad derecha" in text
    assert "convexidad izquierda" in text
    # S-curve descriptor and curve count.
    assert "S-shape" in text or "S-curve" in text or "doble curva" in text
    assert "Numero total de curvas detectadas: 2" in text
    # Assessment is based on the principal angle (32 -> "moderada" in ES,
    # the Spanish severity string is "Escoliosis moderada (25-40 grados)").
    assert "moderada" in text.lower() or "moderate" in text.lower()
    assert "Binary principal" in text


# ----------------------------------------------------------------------------
# Coverage info — Ciclo 5.3 (fix F)
# ----------------------------------------------------------------------------

def test_compute_coverage_info_reports_partial_segmentation():
    """A binary mask that only covers the upper third of the image, paired
    with multiclass vertebrae spread across the whole spine, must be flagged
    as partial coverage. The upper/lower vertebrae and the vertebrae_below
    list are what the UI uses to say 'Lower spine NOT segmented'."""
    import numpy as np
    from spine_segmentation.evaluation.coverage import compute_coverage_info

    H, W = 512, 512
    binary = np.zeros((H, W), dtype=np.uint8)
    binary[50:200, 250:270] = 1  # rows 50-200 only -> ~29% coverage

    # Multiclass detected vertebrae across the full spine.
    multiclass_vertebrae = [
        {"name": "C5", "centroid_y": 60},
        {"name": "C6", "centroid_y": 90},
        {"name": "C7", "centroid_y": 120},
        {"name": "T1", "centroid_y": 150},
        {"name": "T6", "centroid_y": 250},
        {"name": "T12", "centroid_y": 350},
        {"name": "L3", "centroid_y": 430},
    ]

    info = compute_coverage_info(binary, multiclass_vertebrae)

    assert info["success"]
    assert info["is_partial"] is True
    assert abs(info["coverage_ratio"] - (199 - 50) / 512.0) < 0.01
    # upper/lower vertebra: nearest centroid_y to top_y=50 / bottom_y=199.
    assert info["upper_vertebra"] == "C5"
    assert info["lower_vertebra"] == "T1"   # closest to 199 in the fixture
    # Vertebrae in range = those with centroid_y in [50, 199].
    assert set(info["vertebrae_in_range"]) == {"C5", "C6", "C7", "T1"}
    # The ones the binary missed below the range — the warning consumer.
    assert info["vertebrae_below_range"] == ["T6", "T12", "L3"]


def test_compute_coverage_info_reports_full_segmentation():
    """A near-full-image binary mask + plenty of multiclass vertebrae must
    NOT be flagged as partial. This is the happy path for healthy spines."""
    import numpy as np
    from spine_segmentation.evaluation.coverage import compute_coverage_info

    H, W = 512, 512
    binary = np.zeros((H, W), dtype=np.uint8)
    binary[30:490, 240:280] = 1  # ~90% coverage

    multiclass_vertebrae = [
        {"name": f"V{i}", "centroid_y": 40 + i * 20} for i in range(20)
    ]

    info = compute_coverage_info(binary, multiclass_vertebrae)

    assert info["success"]
    assert info["is_partial"] is False
    assert info["coverage_ratio"] > 0.7
    assert info["n_vertebrae"] >= 15
    assert info["vertebrae_below_range"] == []
    assert info["vertebrae_above_range"] == []


def test_compute_coverage_info_handles_empty_mask_and_no_vertebrae():
    """An empty binary mask must return success=False (no crash). With a
    valid mask but no multiclass vertebrae, upper/lower vertebra names
    degrade to None but the coverage ratio is still computed."""
    import numpy as np
    from spine_segmentation.evaluation.coverage import compute_coverage_info

    # Empty mask
    empty = np.zeros((100, 100), dtype=np.uint8)
    info_empty = compute_coverage_info(empty, multiclass_vertebrae=[])
    assert info_empty["success"] is False

    # None mask
    info_none = compute_coverage_info(None, multiclass_vertebrae=[])
    assert info_none["success"] is False

    # Mask with content but no multiclass vertebrae available.
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 40:60] = 1
    info_no_v = compute_coverage_info(mask, multiclass_vertebrae=[])
    assert info_no_v["success"] is True
    assert info_no_v["upper_vertebra"] is None
    assert info_no_v["lower_vertebra"] is None
    assert info_no_v["vertebrae_in_range"] == []


def test_build_results_text_emits_coverage_warning_when_partial():
    """When coverage_info reports partial coverage with lower vertebrae
    missing, the rendered text must include the COVERAGE header, the
    'C6 - T10' range, and the 'Lower spine NOT segmented' warning."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 0.0},
        cobb_multiclass=None,
        coverage_info={
            "success": True,
            "is_partial": True,
            "coverage_ratio": 0.55,
            "vertebrae_in_range": ["C6", "C7", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"],
            "vertebrae_below_range": ["T11", "T12", "L1", "L2", "L3", "L4", "L5"],
            "vertebrae_above_range": [],
            "n_vertebrae": 12,
            "n_expected": 22,
            "upper_vertebra": "C6",
            "lower_vertebra": "T10",
        },
        language="en",  # exact English string assertions
    )

    assert "=== COVERAGE ===" in text
    assert "C6 - T10" in text
    assert "12 of ~22" in text
    assert "~55%" in text
    assert "WARNING" in text
    assert "Lower spine" in text
    assert "T11-L5" in text


def test_build_results_text_no_coverage_warning_when_full():
    """When coverage is full (is_partial=False), the COVERAGE block and
    WARNING line must NOT appear — the UI stays uncluttered for healthy
    cases."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 5.0},
        cobb_multiclass=None,
        coverage_info={
            "success": True,
            "is_partial": False,
            "coverage_ratio": 0.93,
            "n_vertebrae": 20,
            "upper_vertebra": "C3",
            "lower_vertebra": "L5",
            "vertebrae_below_range": [],
            "vertebrae_above_range": [],
        },
    )

    assert "=== COVERAGE ===" not in text
    assert "WARNING" not in text
    assert "Lower spine" not in text


def test_build_results_text_says_inconclusive_when_zero_cobb_and_partial():
    """The most important regression for Ciclo 5.3: a zero Cobb on a partial
    binary mask must NOT be reported as 'Normal'. The previous behavior
    silently labelled S_22-style cases as Normal because cobb=0. After the
    fix, the Assessment is 'Inconclusive — insufficient coverage'."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 0.0},
        cobb_multiclass=None,
        coverage_info={
            "success": True,
            "is_partial": True,
            "coverage_ratio": 0.4,
            "vertebrae_in_range": ["C6", "C7", "T1", "T2"],
            "vertebrae_below_range": ["L1", "L5"],
            "vertebrae_above_range": [],
            "n_vertebrae": 4,
            "upper_vertebra": "C6",
            "lower_vertebra": "T2",
        },
        language="en",  # English-string assertions
    )

    assert "Inconclusive" in text
    assert "Normal" not in text
    assert "ASSESSMENT" in text


def test_draw_cobb_visualization_multi_curve_uses_two_colors():
    """When 2 curves are supplied, the visualization must use BOTH the
    principal red (255, 0, 0) and the secondary magenta (255, 100, 200).
    This is what tells the radiologist apart from a single-curve case."""
    import numpy as np
    from spine_segmentation.evaluation.visualize import draw_cobb_angle_visualization

    h, w = 400, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    # 3 vertebrae stacked vertically
    mask[40:60, 80:140] = 5    # upper -> "T1"
    mask[180:200, 80:140] = 10  # middle -> "T6"
    mask[330:350, 80:140] = 18  # lower -> "L2"

    image = np.full((h, w, 3), 90, dtype=np.uint8)
    cobb_binary = {
        "success": True,
        "cobb_angle_deg": 28.0,
        "curves": [
            {
                "cobb_angle_deg": 28.0,
                "upper_vertebra": "T1",
                "lower_vertebra": "T6",
                "direction": "right",
                "rank": 1,
                "ip_upper": (110.0, 50.0),
                "ip_lower": (110.0, 190.0),
            },
            {
                "cobb_angle_deg": 14.0,
                "upper_vertebra": "T6",
                "lower_vertebra": "L2",
                "direction": "left",
                "rank": 2,
                "ip_upper": (110.0, 190.0),
                "ip_lower": (110.0, 340.0),
            },
        ],
    }
    cobb_multi = {
        "success": True,
        "cobb_angle_deg": 25.0,
        "upper_end_vertebra": "T1",
        "lower_end_vertebra": "L2",
    }
    vis = draw_cobb_angle_visualization(
        image=image,
        multiclass_mask=mask,
        cobb_multiclass_result=cobb_multi,
        cobb_binary_result=cobb_binary,
    )

    # Red pixels (255, 0, 0) — principal curve.
    red = (vis[..., 0] > 200) & (vis[..., 1] < 60) & (vis[..., 2] < 60)
    assert red.sum() > 20, f"expected red pixels for principal, got {red.sum()}"
    # Magenta pixels around (255, 100, 200) — secondary curve.
    magenta = (
        (vis[..., 0] > 200)
        & (vis[..., 1] > 40) & (vis[..., 1] < 180)
        & (vis[..., 2] > 130)
    )
    assert magenta.sum() > 20, f"expected magenta pixels for secondary, got {magenta.sum()}"


# ----------------------------------------------------------------------------
# Ciclo 5.9 — bilingual static reference image in the Explainability tab
# ----------------------------------------------------------------------------

def test_explain_reference_images_exist_and_are_nonempty():
    """Both ES and EN reference PNGs must be committed under
    spine_segmentation/deployment/assets/ and weigh more than 1 KB
    (catches accidental empty-file commits)."""
    import os

    from spine_segmentation.deployment.i18n import explain_reference_path

    for lang in ("es", "en"):
        path = explain_reference_path(lang)
        assert os.path.exists(path), f"missing reference image: {path}"
        size = os.path.getsize(path)
        assert size > 1024, (
            f"reference image suspiciously small ({size} bytes): {path}"
        )


def test_explain_reference_path_distinct_per_lang():
    """ES and EN must resolve to two different PNG files (no silent fallback
    that would make the language toggle a no-op). Unknown languages must
    fall back to the default (Spanish) — same path as 'es'."""
    from spine_segmentation.deployment.i18n import (
        DEFAULT_LANG,
        explain_reference_path,
    )

    es_path = explain_reference_path("es")
    en_path = explain_reference_path("en")
    assert es_path != en_path, "ES and EN reference paths must differ"
    assert es_path.endswith("_es.png")
    assert en_path.endswith("_en.png")
    # Unknown lang collapses to default.
    assert DEFAULT_LANG == "es"
    assert explain_reference_path("xx") == es_path


# ----------------------------------------------------------------------------
# Ciclo 6.1 — chord signed-area convention for curve direction
# (replaces the Ciclo 5.10 midpoint-slope ternary)
# ----------------------------------------------------------------------------

def test_curve_direction_synthetic_parabola_convex_right_patient():
    """Parabola bulging toward viewer-LEFT must report 'right' (patient).

    Coordinate convention used by the cobb pipeline: x grows toward
    the viewer's right, y grows downward. A parabola of the form
    ``x = 50 + 0.005*(y-50)**2`` has its minimum x at y=50 (the
    apex), so the curve bulges to the LEFT of the viewer between the
    two endpoints. By the AP mirror rule, viewer-left = patient-right
    -> expected convexity is 'right'.
    """
    import numpy as np

    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    y = np.linspace(0, 100, 101)
    x = 50 + 0.005 * (y - 50) ** 2
    assert _curve_direction(x, y, ip_a=0, ip_b=100) == "right"


def test_curve_direction_synthetic_parabola_convex_left_patient():
    """Parabola bulging toward viewer-RIGHT must report 'left' (patient).

    Mirror of the previous test: ``x = 50 - 0.005*(y-50)**2`` has its
    maximum x at y=50, so the curve bulges to the RIGHT of the viewer.
    Viewer-right = patient-left -> expected convexity is 'left'.
    """
    import numpy as np

    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    y = np.linspace(0, 100, 101)
    x = 50 - 0.005 * (y - 50) ** 2
    assert _curve_direction(x, y, ip_a=0, ip_b=100) == "left"


def test_curve_direction_s_shape_returns_opposite_lateralities():
    """Canary of Ciclo 6.1: an S-shape must report opposite convexities
    for the two curves separated by an inflection point.

    Two curves separated by an inflection point have OPPOSITE
    convexity by geometric definition of the IP. Under the Ciclo 5.10
    midpoint-slope ternary this property did not hold in general
    (the sweep over 12 cases on 2026-05-22 found 5 of 7 S-shape
    detections reporting the same laterality for both curves), which
    is the bug Ciclo 6.1 was opened to fix. This test FAILS under
    the Ciclo 5.10 implementation.
    """
    import numpy as np

    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    # Ideal S-shape: sinusoid crossing the chord at y=0, y=100, y=200.
    y = np.linspace(0, 200, 201)
    x = 50 + 8 * np.sin(2 * np.pi * y / 200)
    top = _curve_direction(x, y, ip_a=0, ip_b=100)
    bottom = _curve_direction(x, y, ip_a=100, ip_b=200)
    assert top in {"left", "right"}, top
    assert bottom in {"left", "right"}, bottom
    assert top != bottom, (
        f"S-shape sub-curves must have opposite laterality "
        f"(top={top!r}, bottom={bottom!r}); see Ciclo 6.1 addendum."
    )


def test_curve_direction_strong_curve_with_near_vertical_chord_still_works():
    """Strongly curved spine with a near-vertical chord must still
    classify, not collapse to 'unknown' or 'neutral'.

    Construct a sharp curve with both inflection points at x=50 but
    an apex at x=20 in the middle. The chord is exactly vertical
    (chord_dx = 0), which is the worst case for any algorithm that
    naively normalizes by chord direction. Expected: 'right' (apex
    bulges to viewer-left = patient-right).
    """
    import numpy as np

    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    y = np.linspace(0, 100, 101)
    # Parabola with extremes at x=50 and apex at x=20.
    x = 50.0 - 30.0 * (1.0 - ((y - 50.0) / 50.0) ** 2)
    assert _curve_direction(x, y, ip_a=0, ip_b=100) == "right"


def test_curve_direction_below_threshold_returns_neutral():
    """A near-straight spine must collapse to 'neutral', and the
    ``neutral_threshold_px2`` parameter must be a real knob.

    Construct a parabola with a very small coefficient so the signed
    area between curve and chord is well under the 50-px^2 default
    threshold but is NOT exactly zero (a purely linear curve would
    have signed_area = 0 and bypass the threshold check entirely).
    """
    import numpy as np

    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    y = np.linspace(0, 100, 101)
    # Tiny parabola: max x=50 at y=50, min x=49.75 at the extremes.
    # Bulge is to viewer-right (apex x is bigger than extremes) ->
    # patient-left at threshold=0.
    x = 50.0 - 0.0001 * (y - 50.0) ** 2
    # Default threshold (50 px^2) is well above the tiny signed area
    # (~16.7 px^2 by construction) -> neutral.
    assert _curve_direction(x, y, ip_a=0, ip_b=100) == "neutral"
    # Lowering the threshold past the signed area exposes the actual
    # convexity.
    forced = _curve_direction(
        x, y, ip_a=0, ip_b=100, neutral_threshold_px2=1e-9,
    )
    assert forced == "left", forced


def test_curve_direction_invalid_indices_and_degenerate_chord_return_unknown():
    """Edge cases all collapse cleanly to 'unknown' rather than
    crashing or returning a bogus laterality.
    """
    import numpy as np

    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    y = np.linspace(0, 100, 101)
    x = 50.0 + 0.005 * (y - 50.0) ** 2

    # ip_b <= ip_a -> unknown
    assert _curve_direction(x, y, ip_a=80, ip_b=20) == "unknown"
    # ip_b out of range -> unknown
    assert _curve_direction(x, y, ip_a=0, ip_b=9999) == "unknown"
    # ip_a negative -> unknown
    assert _curve_direction(x, y, ip_a=-1, ip_b=10) == "unknown"
    # Length mismatch -> unknown
    assert _curve_direction(x[:50], y, ip_a=0, ip_b=40) == "unknown"
    # Degenerate chord (both IPs at the same point) -> unknown
    x_deg = np.full(101, 50.0)
    y_deg = np.full(101, 25.0)
    assert _curve_direction(x_deg, y_deg, ip_a=0, ip_b=100) == "unknown"


# ----------------------------------------------------------------------------
# Ciclo 5.11 — sample-invariant anchor derivation for reference image
# ----------------------------------------------------------------------------

def _import_generator_helpers():
    """Tiny shim: the script lives under scripts/ which is not a package.
    Add it to sys.path on first use so the helpers can be imported."""
    import importlib
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("generate_explain_reference")


def test_derive_visual_anchors_places_blobs_outside_spine():
    """Ciclo 5.11: anchor positions must be derived from the spine bbox so
    callout arrows always land on something meaningful, no matter which
    sample radiograph is used as the backdrop.

    Pre-Ciclo-5.11 the script had hard-coded blob (cx, cy) and arrow_xy
    values calibrated for S_22's spine layout. When the sample switched to
    S_200 in Ciclo 5.10, those positions ended up over empty black areas
    — the user reported 'apunta a cosas que no tienen sentido'."""
    import numpy as np

    gen = _import_generator_helpers()

    # Synthetic mask: vertical bar in the right-of-center half (mimics
    # the rough geometry of S_200's spine bbox).
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[50:450, 240:280] = 1

    anchors = gen._derive_visual_anchors(mask)

    cx, cy = anchors["centroid"]
    assert 250 <= cx <= 270, f"centroid x should be ~260, got {cx}"
    assert 240 <= cy <= 260, f"centroid y should be ~250, got {cy}"

    xmin, ymin, xmax, ymax = anchors["bbox"]
    assert (xmin, ymin, xmax, ymax) == (240, 50, 279, 449)

    # Blob anchors must sit outside the spine bbox so the synthesized
    # 'spurious' hotspots really are off-anatomy.
    assert anchors["blob_top"][1] < ymin, "blob_top should be ABOVE the spine"
    assert anchors["blob_pelvis"][1] > ymax, "blob_pelvis should be BELOW the spine"
    assert anchors["blob_left"][0] < xmin, "blob_left should be LEFT of the spine"
    assert anchors["blob_right"][0] > xmax, "blob_right should be RIGHT of the spine"

    # Empty-mask fallback returns a usable dict (no crash).
    empty = gen._derive_visual_anchors(np.zeros((512, 512), dtype=np.uint8))
    for key in ("centroid", "bbox", "blob_top", "blob_pelvis", "blob_left", "blob_right"):
        assert key in empty, f"fallback missing key: {key}"


def test_pixel_to_figure_coords_accounts_for_aspect_equal_centering():
    """Ciclo 5.12: image (0, 0) must map to the TOP of the actual rendered
    image rect, NOT the top of the ax rect. With aspect='equal' and an
    ax rect taller than wide in inches, the imshow content is centered
    vertically — image pixel y=0 sits ~0.085 figure-fraction below the
    ax rect top.

    Pre-Ciclo-5.12 the converter assumed the imshow filled the entire
    ax_rect, putting arrow targets ~56 px above the actual blobs in
    AX_CAM_RECT (2.928 in wide x 4.153 in tall, image rendered as
    2.928 x 2.928 in square centered vertically)."""
    gen = _import_generator_helpers()

    rect = (0.21, 0.30, 0.26, 0.58)  # left, bottom, width, height

    # The actual image bbox (computed by _imshow_bbox_in_figure for the
    # default FIG_SIZE_IN = (11.26, 7.16) and img_aspect=1.0):
    #   ax_w_in = 2.9276, ax_h_in = 4.1528 -> width-limited, square 2.9276
    #   vertical margin = (4.1528 - 2.9276) / 2 = 0.6126 in = 0.0856 frac
    #   image rect = (0.21, 0.3856, 0.26, 0.4089)
    expected_image_top_fy = 0.30 + 0.0856 + 0.4089  # ~0.7944
    expected_image_bottom_fy = 0.30 + 0.0856        # ~0.3856

    # Top-left of image -> top-left of *image* rect (NOT ax rect top).
    fx, fy = gen._pixel_to_figure_coords(0, 0, rect, img_size=512)
    assert abs(fx - 0.21) < 1e-6
    assert abs(fy - expected_image_top_fy) < 1e-3, (
        f"Ciclo 5.12: image (0,0) should map to fy~{expected_image_top_fy:.4f}, "
        f"got {fy:.4f}. If the assertion fails by ~0.085 we are back to the "
        f"pre-5.12 bug where the converter assumed the imshow filled the ax rect."
    )

    # Bottom-right of image -> bottom-right of *image* rect.
    fx, fy = gen._pixel_to_figure_coords(512, 512, rect, img_size=512)
    assert abs(fx - 0.47) < 1e-6
    assert abs(fy - expected_image_bottom_fy) < 1e-3

    # Midpoint is preserved by the symmetric centering — same answer pre and
    # post Ciclo 5.12, so this is a quick "did the formula explode?" check.
    fx, fy = gen._pixel_to_figure_coords(256, 256, rect, img_size=512)
    assert abs(fx - 0.34) < 1e-6
    assert abs(fy - 0.59) < 1e-3


def test_imshow_bbox_centers_square_in_tall_rect():
    """Ciclo 5.12: explicit pin for the aspect='equal' centering math.

    AX_CAM_RECT (taller than wide) -> image fits width, centered vertically.
    A hypothetical wider-than-tall rect -> image fits height, centered
    horizontally. Empty mask / default fig size still produces valid output."""
    gen = _import_generator_helpers()

    fig_size_in = (11.26, 7.16)

    # Width-limited path: AX_CAM_RECT
    img_rect = gen._imshow_bbox_in_figure(
        (0.21, 0.30, 0.26, 0.58), fig_size_in, img_aspect=1.0,
    )
    left, bottom, width, height = img_rect
    assert abs(left - 0.21) < 1e-6
    # Image width == ax width.
    assert abs(width - 0.26) < 1e-6
    # Image height = 0.26 * (11.26 / 7.16) = 0.4089... in figure fraction.
    assert abs(height - 0.26 * (11.26 / 7.16)) < 1e-6
    # Bottom is offset by half the empty vertical margin.
    ax_h = 0.58
    expected_bottom = 0.30 + (ax_h - height) / 2.0
    assert abs(bottom - expected_bottom) < 1e-6

    # Height-limited path: a 0.30 wide x 0.20 tall rect at 11.26 x 7.16 in
    # is wider than tall (3.378 in wide, 1.432 in tall). For a square image:
    #   img_h_in = 1.432, img_w_in = 1.432 -> centered horizontally.
    img_rect2 = gen._imshow_bbox_in_figure(
        (0.10, 0.20, 0.30, 0.20), fig_size_in, img_aspect=1.0,
    )
    left2, bottom2, width2, height2 = img_rect2
    assert abs(bottom2 - 0.20) < 1e-6
    assert abs(height2 - 0.20) < 1e-6
    # Image width = 0.20 * (7.16 / 11.26) figure fraction.
    expected_width2 = 0.20 * (7.16 / 11.26)
    assert abs(width2 - expected_width2) < 1e-6
    expected_left2 = 0.10 + (0.30 - expected_width2) / 2.0
    assert abs(left2 - expected_left2) < 1e-6


# ----------------------------------------------------------------------------
# Ciclo 6.2 — bilingual usage instructions + i18n of rotation slider/reset
# ----------------------------------------------------------------------------

def test_instructions_markdown_renders_in_both_languages():
    """The new compact instructions block must render in both Spanish and
    English and cover both pedagogical bullets (rotation + tab legend).

    This pin prevents future refactors from accidentally dropping either
    bullet, which would defeat the whole point of the cycle 6.2 cleanup.
    """
    import re

    from spine_segmentation.deployment.i18n import instructions_markdown

    es_raw = instructions_markdown("es")
    en_raw = instructions_markdown("en")

    # Both must be non-empty real markdown blocks, not the empty-string
    # fallback or the literal key.
    assert es_raw.strip(), "Spanish instructions markdown is empty"
    assert en_raw.strip(), "English instructions markdown is empty"

    # Collapse all whitespace (including hard line wraps inside markdown
    # bold runs like "*ROTATION\nWARNING*") so the substring checks below
    # are robust to the cosmetic line breaks in the source string.
    es = re.sub(r"\s+", " ", es_raw)
    en = re.sub(r"\s+", " ", en_raw)

    # Bullet 1: rotation guidance must mention the Analyze action and
    # the ROTATION WARNING the model emits when it detects tilt.
    assert "Analyze" in es and "Analyze" in en
    assert "ROTATION WARNING" in es and "ROTATION WARNING" in en

    # Bullet 2: tab legend must reference all four output tabs by name
    # (these strings match the tab labels rendered in app.py).
    for tab_name in ("Binary", "Vertebrae", "Cobb Angle", "Explainability"):
        assert tab_name in es, f"ES instructions missing {tab_name!r}"
        assert tab_name in en, f"EN instructions missing {tab_name!r}"

    # The two languages must NOT be identical strings (cheap sanity
    # check that the EN entry is not a copy-paste of the ES entry).
    assert es_raw != en_raw


def test_rotation_slider_label_is_translated():
    """The rotation slider label and the Reset button label must both
    have non-trivial ES + EN entries in DIAGNOSIS_STRINGS (cycle 6.2).

    Before 6.2 these were hardcoded in English directly in app.py. The
    fix added them to i18n.py so the language toggle can rewrite them
    live via gr.update(label=...) / gr.update(value=...).
    """
    from spine_segmentation.deployment.i18n import t

    slider_es = t("rotation_slider_label", "es")
    slider_en = t("rotation_slider_label", "en")
    assert slider_es != "rotation_slider_label", (
        "Spanish key fell through to the fallback — ES translation missing."
    )
    assert slider_en != "rotation_slider_label", (
        "English key fell through to the fallback — EN translation missing."
    )
    # The Spanish label should mention the rotation direction in Spanish.
    assert "horario" in slider_es or "grados" in slider_es

    reset_es = t("rotation_reset_button", "es")
    reset_en = t("rotation_reset_button", "en")
    assert reset_es != "rotation_reset_button"
    assert reset_en != "rotation_reset_button"
    # English Reset should literally be "Reset"; Spanish should not.
    assert reset_en == "Reset"
    assert reset_es != "Reset"
