"""
Morphological post-processing for segmentation masks.
Cleans up predictions and extracts connected components.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, remove_small_objects


def clean_binary_mask(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Post-process binary segmentation mask.
    1. Threshold
    2. Morphological opening (remove noise)
    3. Keep largest connected component
    4. Morphological closing (fill holes)

    Args:
        mask: (H, W) binary prediction {0, 1}
        min_size: Minimum component size in pixels

    Returns:
        Cleaned binary mask
    """
    mask = mask.astype(np.uint8)

    # Morphological opening (remove small noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Keep only the largest connected component
    labeled = label(mask)
    if labeled.max() == 0:
        return mask

    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    mask = (labeled == largest.label).astype(np.uint8)

    # Morphological closing (fill small holes)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def clean_multiclass_mask(
    mask: np.ndarray,
    num_classes: int,
    min_vertebra_size: int = 100,
) -> np.ndarray:
    """
    Post-process multiclass segmentation mask.
    For each class: keep only the largest connected component,
    remove small detections.

    Args:
        mask: (H, W) integer class labels
        num_classes: Total number of classes
        min_vertebra_size: Minimum pixel count for a valid vertebra detection

    Returns:
        Cleaned multiclass mask
    """
    cleaned = np.zeros_like(mask)

    for c in range(1, num_classes):  # Skip background (0)
        class_mask = (mask == c).astype(np.uint8)
        if class_mask.sum() < min_vertebra_size:
            continue

        # Keep largest connected component for this class
        labeled = label(class_mask)
        if labeled.max() == 0:
            continue

        regions = regionprops(labeled)
        largest = max(regions, key=lambda r: r.area)

        if largest.area >= min_vertebra_size:
            cleaned[labeled == largest.label] = c

    return cleaned


def extract_spine_skeleton(binary_mask: np.ndarray, min_branch_length: int = 50) -> np.ndarray:
    """
    Extract the spine centerline from a binary mask using skeletonization.
    Removes branches to get a single spine curve.

    Based on the approach from the previous semester's notebook.

    Args:
        binary_mask: (H, W) binary spine mask
        min_branch_length: Minimum branch length to keep

    Returns:
        (H, W) skeleton mask (single pixel wide)
    """
    # Close any gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Skeletonize
    skeleton = skeletonize(closed > 0).astype(np.uint8)

    # Remove branches: iteratively remove branch endpoints
    # A branch endpoint has exactly 1 neighbor
    skeleton = _remove_branches(skeleton, min_length=min_branch_length)

    return skeleton


def _remove_branches(skeleton: np.ndarray, min_length: int = 50) -> np.ndarray:
    """Remove short branches from skeleton, keeping main spine axis."""
    # Label connected components
    labeled = label(skeleton)
    if labeled.max() == 0:
        return skeleton

    # Keep only the longest component
    regions = regionprops(labeled)
    if not regions:
        return skeleton

    largest = max(regions, key=lambda r: r.area)
    main_skeleton = (labeled == largest.label).astype(np.uint8)

    return main_skeleton


def get_skeleton_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Extract ordered (x, y) coordinates from a skeleton mask.
    Points are ordered from top to bottom (anatomical order).

    Args:
        skeleton: (H, W) binary skeleton mask

    Returns:
        np.ndarray of shape (N, 2) with columns [x, y]
    """
    # Get all skeleton pixel coordinates
    ys, xs = np.where(skeleton > 0)

    if len(xs) == 0:
        return np.array([]).reshape(0, 2)

    # Sort by y-coordinate (top to bottom)
    sort_idx = np.argsort(ys)
    points = np.column_stack([xs[sort_idx], ys[sort_idx]])

    return points
