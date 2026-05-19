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

    # Tilted -> block visible
    text_tilted = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 31.8,
                     "curves": [{"cobb_angle_deg": 31.8, "upper_vertebra": "L3",
                                 "lower_vertebra": "L4", "direction": "right",
                                 "rank": 1}]},
        cobb_multiclass={"success": True, "cobb_angle_deg": 0.6,
                         "upper_end_vertebra": "C5", "lower_end_vertebra": "L4"},
        orientation_info={"success": True, "tilt_deg": 18.4, "tilt_abs_deg": 18.4,
                          "is_tilted": True, "threshold_deg": 12.0, "n_points": 200},
    )
    assert "=== ROTATION WARNING ===" in text_tilted
    assert "18.4 deg" in text_tilted
    assert "threshold 12 deg" in text_tilted
    assert "multiclass" in text_tilted

    # Not tilted -> no block
    text_ok = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 5.0},
        cobb_multiclass=None,
        orientation_info={"success": True, "tilt_deg": 2.0, "tilt_abs_deg": 2.0,
                          "is_tilted": False, "threshold_deg": 12.0, "n_points": 200},
    )
    assert "ROTATION WARNING" not in text_ok

    # No orientation_info -> no block (back-compat path)
    text_none = build_results_text(
        cobb_binary={"success": True, "cobb_angle_deg": 5.0},
        cobb_multiclass=None,
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
