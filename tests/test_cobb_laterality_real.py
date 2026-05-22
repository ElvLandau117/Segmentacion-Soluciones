"""test_cobb_laterality_real.py — Ciclo 6.1 regression anchored to MaIA.

These tests load real centroidal curves from the MaIA dataset's
``RadiographMetrics/`` directory and assert that ``_curve_direction``
agrees with the laterality derived from the official ground truth
(apex_x vs csvl_x). Skipped automatically when the dataset is not
available locally (it is .gitignored — see repo README).

The dataset's official ``cobb_curve_metrics.i_apex_global`` and
``csvl.x_px`` give us a viewer-perspective laterality without touching
the segmentation pipeline: if the apex x-coordinate is left of the
central sacral vertical line (CSVL), the curve bulges to viewer-left,
which by the AP mirror rule means patient-right. We compare against
that.

Only the principal curve is anchored — the dataset labels exactly one
curve per radiograph, so we cannot validate S-shape secondaries from
ground truth (those are validated via the synthetic S-shape canary in
test_app_smoke.py and via the human-in-the-loop sweep_laterality.py
output).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest


def _resolve_metrics_dir() -> Path:
    """Locate ``RadiographMetrics/`` whether the suite runs from the
    main checkout or from a worktree (where the dataset is .gitignored).

    Checks, in order: ``MAIA_DATASET_ROOT`` env var, the repo this file
    lives in, and the first ancestor up to 5 levels above that does
    contain the dataset folder.
    """
    env = os.environ.get("MAIA_DATASET_ROOT")
    if env:
        candidate = Path(env) / "RadiographMetrics"
        if candidate.exists():
            return candidate

    here = Path(__file__).resolve()
    for ancestor in (here.parents[1], *here.parents[2:7]):
        candidate = ancestor / "MaIA_Scoliosis_Dataset" / "RadiographMetrics"
        if candidate.exists():
            return candidate
    return here.parents[1] / "MaIA_Scoliosis_Dataset" / "RadiographMetrics"


METRICS_DIR = _resolve_metrics_dir()
JSON_DIR = METRICS_DIR / "metrics_json"
CURVES_DIR = METRICS_DIR / "curves_csv"

requires_dataset = pytest.mark.skipif(
    not (JSON_DIR.exists() and CURVES_DIR.exists()),
    reason="MaIA dataset (RadiographMetrics/) not present locally",
)


def _load_case(patient_id: int):
    """Load (x_eval, y_eval, ip_a, ip_b, expected_direction) for a case.

    `expected_direction` is derived from the apex position relative to
    the central sacral vertical line in the official metrics JSON.
    """
    import pandas as pd  # local import keeps top-level import light

    with (JSON_DIR / f"metrics_{patient_id}.json").open() as handle:
        meta = json.load(handle)
    df = pd.read_csv(CURVES_DIR / f"curve_{patient_id}.csv")

    x = df["x_px"].to_numpy(dtype=float)
    y = df["y_px"].to_numpy(dtype=float)

    cm = meta["cobb_curve_metrics"]
    ip_a = int(cm["i_inf_below"])
    ip_b = int(cm["i_inf_above"])
    if ip_a > ip_b:
        ip_a, ip_b = ip_b, ip_a

    apex_idx = int(cm["i_apex_global"])
    apex_x = float(x[apex_idx])
    csvl_x = float(meta["csvl"]["x_px"])

    # Mirror rule: apex on viewer-left (apex_x < csvl_x) = patient-right.
    expected = "right" if apex_x < csvl_x else "left"

    return x, y, ip_a, ip_b, expected, apex_x, csvl_x


@requires_dataset
def test_curve_direction_matches_maia_ground_truth_s_158():
    """S_158 is the pivot case of the Ciclo 5.10 fix. Ground truth has
    apex_x ~ 65.7 and csvl_x = 262, so the curve bulges to viewer-left
    (patient-right). The new chord signed-area implementation must
    report 'right', preserving the Ciclo 5.10 outcome on this case.
    """
    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    x, y, ip_a, ip_b, expected, apex_x, csvl_x = _load_case(158)
    assert expected == "right", (
        f"Sanity: apex_x={apex_x} vs csvl_x={csvl_x} should imply 'right'."
    )
    assert _curve_direction(np.asarray(x), np.asarray(y), ip_a, ip_b) == "right"


@requires_dataset
def test_curve_direction_matches_maia_ground_truth_s_22():
    """S_22 is the case that the baseline sweep on 2026-05-22 reported
    as 'left' under the Ciclo 5.10 implementation, in disagreement with
    the official ground truth (apex_x=116.8 < csvl_x=142 -> patient
    right). The new chord signed-area implementation must report
    'right', closing that miscall.
    """
    from spine_segmentation.evaluation.cobb_angle import _curve_direction

    x, y, ip_a, ip_b, expected, apex_x, csvl_x = _load_case(22)
    assert expected == "right", (
        f"Sanity: apex_x={apex_x} vs csvl_x={csvl_x} should imply 'right'."
    )
    assert _curve_direction(np.asarray(x), np.asarray(y), ip_a, ip_b) == "right"
