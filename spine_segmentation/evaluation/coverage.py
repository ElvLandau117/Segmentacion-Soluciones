"""Coverage analysis of the binary spine segmentation.

When the binary mask only covers part of the spine (e.g. C6-T10 on a case
like S_22 where the lumbar signal is too weak for the network), the Cobb
pipeline silently fits a spline to a partial spine and reports "0 deg —
no clinically meaningful curves". That is misleading: the spine HAS curves,
we just did not segment them.

`compute_coverage_info` quantifies this so the UI can warn the user instead
of pretending the case is Normal. Added in Ciclo 5.3 (fix F).
"""

from __future__ import annotations

import numpy as np


# Anatomical baseline: C3, C4, C5, C6, C7 + T1..T12 + L1..L5 = 22 vertebrae.
# Used as the denominator for the "N of ~22 vertebrae" message.
N_EXPECTED_VERTEBRAE = 22

# Coverage is declared partial when the binary mask's vertical extent is
# below this fraction of the image height, OR when multiclass detected fewer
# vertebrae than this threshold inside the binary range.
COVERAGE_RATIO_THRESHOLD = 0.7
N_VERTEBRAE_THRESHOLD = 15


def compute_coverage_info(
    binary_mask: np.ndarray,
    multiclass_vertebrae: list,
    image_height: int | None = None,
) -> dict:
    """Quantify how much of the spine the binary mask actually covers.

    Args:
        binary_mask: (H, W) uint8 mask {0, 1}. Empty/None handled gracefully.
        multiclass_vertebrae: list of dicts from
            `postprocessing.vertebra_ordering.extract_vertebra_info`, each
            with at least `name` and `centroid_y`. Pass [] if multiclass is
            unavailable — then upper/lower vertebra names degrade to None.
        image_height: optional override; defaults to `binary_mask.shape[0]`.

    Returns:
        dict with:
          - success (bool): False if binary mask is empty / None.
          - top_y, bottom_y (int): vertical extent of the binary mask, pixels.
          - coverage_ratio (float): (bottom_y - top_y) / image_height, [0, 1].
          - vertebrae_in_range (list[str]): names of multiclass vertebrae whose
            centroid_y falls within [top_y, bottom_y].
          - vertebrae_below_range (list[str]): multiclass vertebrae the binary
            mask MISSED on the inferior side (centroid_y > bottom_y).
          - vertebrae_above_range (list[str]): multiclass vertebrae the binary
            mask MISSED on the superior side (centroid_y < top_y).
          - n_vertebrae (int): len(vertebrae_in_range).
          - n_expected (int): N_EXPECTED_VERTEBRAE = 22 (C3-L5).
          - is_partial (bool): True iff coverage_ratio < 0.7 OR
            (multiclass available AND n_vertebrae < 15). The two-condition
            OR matters because a tall mask with very few labelled vertebrae
            (multiclass detection failed) and a short mask with many labels
            (binary missed something multiclass caught) are both "partial".
          - upper_vertebra, lower_vertebra (str|None): names closest to top_y
            and bottom_y of the binary mask — used by the UI to say
            "Binary mask covers: C6 - T10".
    """
    result = {
        "success": False,
        "top_y": None,
        "bottom_y": None,
        "coverage_ratio": 0.0,
        "vertebrae_in_range": [],
        "vertebrae_below_range": [],
        "vertebrae_above_range": [],
        "n_vertebrae": 0,
        "n_expected": N_EXPECTED_VERTEBRAE,
        "is_partial": True,
        "upper_vertebra": None,
        "lower_vertebra": None,
    }

    if binary_mask is None or int(np.asarray(binary_mask).sum()) == 0:
        return result

    H = image_height if image_height is not None else binary_mask.shape[0]
    row_sums = np.asarray(binary_mask).sum(axis=1)
    rows = np.where(row_sums > 0)[0]
    if len(rows) == 0:
        return result

    top_y = int(rows.min())
    bottom_y = int(rows.max())
    coverage_ratio = float(bottom_y - top_y) / float(H) if H > 0 else 0.0

    vertebrae_in_range: list[str] = []
    vertebrae_below_range: list[str] = []
    vertebrae_above_range: list[str] = []
    upper_vertebra: str | None = None
    lower_vertebra: str | None = None

    if multiclass_vertebrae:
        # Sort once by y so the *_below/*_above lists are anatomically ordered
        # (cervical -> lumbar) for nicer UI strings.
        sorted_v = sorted(multiclass_vertebrae, key=lambda v: v.get("centroid_y", 1e9))
        for v in sorted_v:
            cy = v.get("centroid_y", -1)
            name = v.get("name")
            if name is None:
                continue
            if top_y <= cy <= bottom_y:
                vertebrae_in_range.append(name)
            elif cy > bottom_y:
                vertebrae_below_range.append(name)
            else:
                vertebrae_above_range.append(name)

        # Closest-name labeling for the coverage line ("C6 - T10")
        upper = min(sorted_v, key=lambda v: abs(v.get("centroid_y", 1e9) - top_y))
        lower = min(sorted_v, key=lambda v: abs(v.get("centroid_y", 1e9) - bottom_y))
        upper_vertebra = upper.get("name")
        lower_vertebra = lower.get("name")

    n_vertebrae = len(vertebrae_in_range)
    is_partial = coverage_ratio < COVERAGE_RATIO_THRESHOLD
    if multiclass_vertebrae and n_vertebrae < N_VERTEBRAE_THRESHOLD:
        is_partial = True

    result.update({
        "success": True,
        "top_y": top_y,
        "bottom_y": bottom_y,
        "coverage_ratio": coverage_ratio,
        "vertebrae_in_range": vertebrae_in_range,
        "vertebrae_below_range": vertebrae_below_range,
        "vertebrae_above_range": vertebrae_above_range,
        "n_vertebrae": n_vertebrae,
        "is_partial": is_partial,
        "upper_vertebra": upper_vertebra,
        "lower_vertebra": lower_vertebra,
    })

    return result
