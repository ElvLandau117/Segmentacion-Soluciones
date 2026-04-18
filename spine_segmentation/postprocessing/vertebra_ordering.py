"""
Vertebra identification and anatomical ordering from multiclass segmentation masks.
Enforces anatomical constraints (top-to-bottom ordering: C3 -> L5).
"""

import cv2
import numpy as np
from skimage.measure import regionprops, label
from typing import List, Tuple, Optional

from spine_segmentation.config import VERTEBRAE_ORDER


def extract_vertebra_info(
    mask: np.ndarray,
    class_names: dict,
    min_area: int = 100,
) -> List[dict]:
    """
    Extract information about each detected vertebra from a multiclass mask.

    Args:
        mask: (H, W) integer class labels
        class_names: dict mapping class_id -> name
        min_area: Minimum pixel area to consider a valid detection

    Returns:
        List of dicts with keys: class_id, name, centroid_x, centroid_y,
        area, bbox, orientation, major_axis, minor_axis
    """
    vertebrae = []

    for class_id in range(1, max(mask.max() + 1, 1)):
        if class_id not in class_names:
            continue

        name = class_names[class_id]
        class_mask = (mask == class_id).astype(np.uint8)

        if class_mask.sum() < min_area:
            continue

        # Get region properties
        labeled = label(class_mask)
        regions = regionprops(labeled)

        if not regions:
            continue

        # Take the largest region for this class
        region = max(regions, key=lambda r: r.area)

        if region.area < min_area:
            continue

        vertebrae.append({
            "class_id": class_id,
            "name": name,
            "centroid_y": region.centroid[0],
            "centroid_x": region.centroid[1],
            "area": region.area,
            "bbox": region.bbox,  # (min_row, min_col, max_row, max_col)
            "orientation": region.orientation,  # Angle in radians (-pi/2 to pi/2)
            "major_axis": region.major_axis_length,
            "minor_axis": region.minor_axis_length,
            "moments": region.moments_central,
        })

    # Sort by vertical position (top to bottom)
    vertebrae.sort(key=lambda v: v["centroid_y"])

    return vertebrae


def validate_anatomical_order(vertebrae: List[dict]) -> Tuple[bool, List[str]]:
    """
    Check if detected vertebrae follow correct anatomical order (top to bottom).

    Args:
        vertebrae: List of vertebra info dicts sorted by centroid_y

    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    if len(vertebrae) < 2:
        return True, []

    warnings = []

    for i in range(len(vertebrae) - 1):
        name_i = vertebrae[i]["name"]
        name_j = vertebrae[i + 1]["name"]

        # Check if both are standard vertebrae
        if name_i not in VERTEBRAE_ORDER or name_j not in VERTEBRAE_ORDER:
            continue

        idx_i = VERTEBRAE_ORDER.index(name_i)
        idx_j = VERTEBRAE_ORDER.index(name_j)

        if idx_i >= idx_j:
            warnings.append(
                f"Order violation: {name_i} (y={vertebrae[i]['centroid_y']:.0f}) "
                f"appears above {name_j} (y={vertebrae[j]['centroid_y']:.0f})"
            )

    return len(warnings) == 0, warnings


def compute_endplate_angles(vertebrae: List[dict]) -> List[dict]:
    """
    Estimate the endplate orientation for each vertebra based on the
    shape (orientation) of its segmentation mask.

    The orientation from regionprops gives the angle of the major axis,
    which approximates the vertebral body tilt.

    Args:
        vertebrae: List of vertebra info dicts

    Returns:
        Same list with added 'endplate_angle_deg' key
    """
    for vert in vertebrae:
        # The orientation from regionprops is in radians, measuring the angle
        # of the major axis relative to the horizontal.
        # For vertebrae, this approximates the endplate tilt.
        angle_rad = vert["orientation"]
        angle_deg = np.degrees(angle_rad)
        vert["endplate_angle_deg"] = angle_deg

    return vertebrae


def identify_end_vertebrae(vertebrae: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Identify the upper and lower end vertebrae of the scoliotic curve.
    The end vertebrae are the ones with the maximum tilt relative to horizontal
    at the extremes of the curve.

    For the Cobb angle calculation:
    - Upper end vertebra: most tilted vertebra in the upper portion of the curve
    - Lower end vertebra: most tilted vertebra in the lower portion of the curve

    Args:
        vertebrae: List of vertebra info dicts with endplate angles

    Returns:
        Tuple of (upper_end_vertebra, lower_end_vertebra)
    """
    if len(vertebrae) < 3:
        return None, None

    vertebrae = compute_endplate_angles(vertebrae)

    # Find the apex (vertebra with maximum lateral displacement)
    # Approximate by finding the vertebra furthest from the line connecting
    # the top and bottom vertebrae
    top = vertebrae[0]
    bottom = vertebrae[-1]

    if top["centroid_y"] == bottom["centroid_y"]:
        return None, None

    # Line from top to bottom centroid
    dx = bottom["centroid_x"] - top["centroid_x"]
    dy = bottom["centroid_y"] - top["centroid_y"]
    line_length = np.sqrt(dx**2 + dy**2)

    max_dist = 0
    apex_idx = len(vertebrae) // 2

    for i, vert in enumerate(vertebrae):
        # Distance from point to line
        dist = abs(dy * vert["centroid_x"] - dx * vert["centroid_y"]
                   + bottom["centroid_x"] * top["centroid_y"]
                   - bottom["centroid_y"] * top["centroid_x"]) / max(line_length, 1e-6)
        if dist > max_dist:
            max_dist = dist
            apex_idx = i

    # Upper end vertebra: most tilted above apex
    upper_candidates = vertebrae[:apex_idx + 1]
    if upper_candidates:
        upper_end = max(upper_candidates, key=lambda v: abs(v["endplate_angle_deg"]))
    else:
        upper_end = vertebrae[0]

    # Lower end vertebra: most tilted below apex
    lower_candidates = vertebrae[apex_idx:]
    if lower_candidates:
        lower_end = max(lower_candidates, key=lambda v: abs(v["endplate_angle_deg"]))
    else:
        lower_end = vertebrae[-1]

    return upper_end, lower_end
