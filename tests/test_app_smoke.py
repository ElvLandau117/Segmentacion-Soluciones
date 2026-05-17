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
    from the noisy multiclass and false-labelled scoliosis cases as Normal."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 30.0},
        cobb_multiclass={
            "success": True, "cobb_angle_deg": 5.0,
            "upper_end_vertebra": "T6", "lower_end_vertebra": "T12",
        },
    )

    assert "ASSESSMENT (based on Binary)" in text
    # 30 deg falls in Moderate (25-40), not Normal (<10). Without the binary
    # source, the assessment would say Normal (because multi is 5.0).
    assert "Moderate" in text
    assert "Normal" not in text


def test_build_results_text_includes_concordance_when_both_succeed():
    """When both methods produce a Cobb, the CONCORDANCE block must appear with
    the correct label for the difference range."""
    from spine_segmentation.deployment.app import build_results_text

    # diff = 1.5 -> High agreement
    text_high = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 20.0},
        cobb_multiclass={"success": True, "cobb_angle_deg": 21.5,
                         "upper_end_vertebra": "T5", "lower_end_vertebra": "T11"},
    )
    assert "CONCORDANCE" in text_high
    assert "High agreement" in text_high

    # diff = 25 -> Significant discrepancy
    text_disc = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 10.0},
        cobb_multiclass={"success": True, "cobb_angle_deg": 35.0,
                         "upper_end_vertebra": "T5", "lower_end_vertebra": "T11"},
    )
    assert "CONCORDANCE" in text_disc
    assert "Significant discrepancy" in text_disc


def test_build_results_text_falls_back_to_multiclass_when_binary_fails():
    """If the binary method failed (e.g. empty mask), the Assessment must use
    multiclass as a labelled fallback so the user still gets a severity hint.
    No CONCORDANCE block in this case (only one Cobb value)."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": False, "error": "Empty mask after cleaning"},
        cobb_multiclass={"success": True, "cobb_angle_deg": 28.0,
                         "upper_end_vertebra": "T6", "lower_end_vertebra": "L1"},
    )

    assert "ASSESSMENT (based on Multiclass fallback)" in text
    assert "Moderate" in text
    assert "CONCORDANCE" not in text


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
