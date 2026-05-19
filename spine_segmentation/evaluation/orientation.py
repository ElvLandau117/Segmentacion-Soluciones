"""Spine orientation analysis (Ciclo 5.4 fix G).

The binary Cobb method fits a B-spline `x = f(y)` over the spine skeleton.
That parameterization assumes the spine is roughly vertical in the frame; a
rotated input image traces a curved x(y) path even on a perfectly straight
spine, and the second derivative produces SPURIOUS inflection points that
get reported as Cobb angles.

`compute_orientation_info` measures the principal axis of the skeleton
point cloud via SVD and tells the UI when the input image is tilted enough
that the binary readings should be taken with a grain of salt. The
multiclass method (per-vertebra endplate angle) is roughly rotation-
invariant, so it remains a reliable fallback in those cases.

Empirically derived on N_61.jpg (a Normal case rotated in the frame that
produced 4 phantom curves with the binary method).
"""

from __future__ import annotations

import numpy as np


# Tilt threshold above which we surface a rotation warning. 12 degrees is
# more than typical clinical capture jitter but well under a frank
# re-orientation, so true positives outweigh false alarms.
TILT_THRESHOLD_DEG = 12.0


def compute_orientation_info(skeleton_points: np.ndarray | None) -> dict:
    """Detect whether the spine skeleton is significantly off-vertical.

    Uses SVD on the (n, 2) skeleton point cloud. The first singular vector
    is the principal axis of the points; its angle with the image's vertical
    axis (y) tells us how tilted the spine is in the frame.

    Args:
        skeleton_points: array of shape (N, 2) with columns [x, y] in image
            coordinates (origin top-left, y-down). May be None or empty.

    Returns:
        dict with:
          - success (bool): False if the input has <3 points or is degenerate.
          - tilt_deg (float): signed angle of the principal axis vs vertical.
            Range (-90, 90]. 0 means the spine is perfectly vertical; +ve
            means the top of the spine leans to the right relative to the
            bottom (in image x).
          - tilt_abs_deg (float): abs(tilt_deg) for easy thresholding.
          - is_tilted (bool): abs(tilt_deg) > TILT_THRESHOLD_DEG.
          - threshold_deg (float): the threshold used (TILT_THRESHOLD_DEG).
          - n_points (int): how many skeleton points were considered.
    """
    result = {
        "success": False,
        "tilt_deg": 0.0,
        "tilt_abs_deg": 0.0,
        "is_tilted": False,
        "threshold_deg": TILT_THRESHOLD_DEG,
        "n_points": 0,
    }

    if skeleton_points is None:
        return result

    pts = np.asarray(skeleton_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        return result

    # Center the cloud — SVD on raw image coords would mix translation into
    # the principal axis. Centroid is the natural reference.
    centered = pts - pts.mean(axis=0)

    # If the cloud collapsed to a point (all skeleton on a single pixel,
    # extremely rare but possible on a 1-pixel mask), bail out.
    if np.linalg.norm(centered) < 1e-6:
        return result

    # SVD: U is point-space (n x 2), S are the singular values (2,),
    # Vt is the principal-axis frame (2 x 2). The first row of Vt is the
    # direction of maximum variance — the principal axis of the spine.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)

    # Reject degenerate inputs where the spread is essentially 1-D OR has
    # no variance at all in the secondary axis (line of identical y's).
    if s[0] < 1e-6:
        return result

    principal = vt[0]  # (x_dir, y_dir)

    # Vertical axis in image coords is (0, 1) (y-down). The angle of the
    # principal axis vs vertical is atan2(|x_component|, |y_component|).
    # Using arctan2(x, y) (note swapped order vs the usual atan2(y, x))
    # gives 0 for a perfectly vertical principal direction and increases
    # as the principal direction tilts away from y.
    px, py = float(principal[0]), float(principal[1])

    # We want the signed angle vs the +y axis. atan2(px, py) is positive
    # when px > 0 (principal direction leans right of vertical). Wrap into
    # (-90, 90] because the principal axis is sign-ambiguous (direction
    # could point top->bottom or bottom->top; only the line matters).
    tilt_rad = np.arctan2(px, py)
    tilt_deg = float(np.degrees(tilt_rad))
    # Wrap to (-90, 90]: principal axes are line-like, not vectors.
    if tilt_deg > 90.0:
        tilt_deg -= 180.0
    elif tilt_deg <= -90.0:
        tilt_deg += 180.0

    tilt_abs = abs(tilt_deg)

    result.update({
        "success": True,
        "tilt_deg": tilt_deg,
        "tilt_abs_deg": tilt_abs,
        "is_tilted": tilt_abs > TILT_THRESHOLD_DEG,
        "n_points": int(pts.shape[0]),
    })
    return result
