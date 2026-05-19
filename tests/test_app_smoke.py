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
    """When both methods produce a Cobb, the CONCORDANCIA cross-check block must
    appear with the correct label for the difference range (Ciclo 5.2: header
    renamed CONCORDANCE -> CONCORDANCIA to match the multi-curve UI in Spanish)."""
    from spine_segmentation.deployment.app import build_results_text

    # diff = 1.5 -> High agreement
    text_high = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 20.0},
        cobb_multiclass={"success": True, "cobb_angle_deg": 21.5,
                         "upper_end_vertebra": "T5", "lower_end_vertebra": "T11"},
    )
    assert "CONCORDANCIA" in text_high
    assert "High agreement" in text_high

    # diff = 25 -> Significant discrepancy
    text_disc = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 10.0},
        cobb_multiclass={"success": True, "cobb_angle_deg": 35.0,
                         "upper_end_vertebra": "T5", "lower_end_vertebra": "T11"},
    )
    assert "CONCORDANCIA" in text_disc
    assert "Significant discrepancy" in text_disc


def test_build_results_text_falls_back_to_multiclass_when_binary_fails():
    """If the binary method failed (e.g. empty mask), the Assessment must use
    multiclass as a labelled fallback so the user still gets a severity hint.
    No CONCORDANCIA block in this case (only one Cobb value)."""
    from spine_segmentation.deployment.app import build_results_text

    text = build_results_text(
        cobb_binary={"success": False, "error": "Empty mask after cleaning"},
        cobb_multiclass={"success": True, "cobb_angle_deg": 28.0,
                         "upper_end_vertebra": "T6", "lower_end_vertebra": "L1"},
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
    # Assessment is based on the principal angle (32 -> Moderate).
    assert "Moderate" in text
    assert "Binary principal" in text


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
