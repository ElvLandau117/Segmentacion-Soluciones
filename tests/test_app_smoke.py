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

    vis = draw_cobb_angle_visualization(
        image=image,
        multiclass_mask=mask,
        cobb_multiclass_result=cobb_result,
        cobb_binary_deg=10.0,
    )

    assert vis.shape == image.shape
    assert vis.dtype == np.uint8
    # Must have modified pixels (overlay/lines/text). Compare to original.
    assert not np.array_equal(vis, image)
