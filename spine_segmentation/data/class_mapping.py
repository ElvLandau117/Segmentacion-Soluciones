"""
Class mapping strategies for multiclass segmentation.
Supports multiple class grouping schemes for the MaIA Scoliosis Dataset.
"""

import numpy as np


def get_remap_table(scheme: str = "vertebrae_24") -> np.ndarray:
    """
    Returns a lookup table (LUT) of length 256 that maps original pixel IDs
    to new class IDs based on the selected scheme.

    Args:
        scheme: One of 'vertebrae_24', 'full_36', 'regional_5'

    Returns:
        np.ndarray of shape (256,) with dtype uint8.
        Usage: new_mask = lut[original_mask]
    """
    lut = np.zeros(256, dtype=np.uint8)

    if scheme == "vertebrae_24":
        # Keep vertebrae IDs 0-22 as-is, collapse entities (23-35) to class 23
        for i in range(23):
            lut[i] = i
        for i in range(23, 36):
            lut[i] = 23  # "other_structures"

    elif scheme == "full_36":
        # Keep all 36 classes unchanged
        for i in range(36):
            lut[i] = i

    elif scheme == "regional_5":
        # 0=background, 1=cervical(C3-C7), 2=thoracic(T1-T12),
        # 3=lumbar(L1-L5), 4=other
        lut[0] = 0                    # background
        for i in range(1, 6):
            lut[i] = 1                # cervical: C7(1), C6(2), C5(3), C4(4), C3(5)
        for i in range(6, 18):
            lut[i] = 2                # thoracic: T1(6) through T12(17)
        for i in range(18, 23):
            lut[i] = 3                # lumbar: L1(18) through L5(22)
        for i in range(23, 36):
            lut[i] = 4                # other entities

    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'vertebrae_24', 'full_36', or 'regional_5'")

    return lut


def get_num_classes(scheme: str = "vertebrae_24") -> int:
    """Returns the number of classes for the given scheme."""
    mapping = {
        "vertebrae_24": 24,
        "full_36": 36,
        "regional_5": 5,
    }
    if scheme not in mapping:
        raise ValueError(f"Unknown scheme: {scheme}")
    return mapping[scheme]


def get_class_names(scheme: str = "vertebrae_24") -> dict:
    """Returns a dict mapping class_id -> class_name for the given scheme."""
    if scheme == "vertebrae_24":
        return {
            0: "background",
            1: "C7", 2: "C6", 3: "C5", 4: "C4", 5: "C3",
            6: "T1", 7: "T2", 8: "T3", 9: "T4", 10: "T5",
            11: "T6", 12: "T7", 13: "T8", 14: "T9", 15: "T10",
            16: "T11", 17: "T12",
            18: "L1", 19: "L2", 20: "L3", 21: "L4", 22: "L5",
            23: "other_structures",
        }
    elif scheme == "full_36":
        from spine_segmentation.config import FULL_CLASS_NAMES
        return FULL_CLASS_NAMES.copy()
    elif scheme == "regional_5":
        return {
            0: "background",
            1: "cervical",
            2: "thoracic",
            3: "lumbar",
            4: "other",
        }
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def remap_mask(mask: np.ndarray, scheme: str = "vertebrae_24") -> np.ndarray:
    """
    Remap a multiclass mask from original 36-class IDs to the target scheme.

    Args:
        mask: np.ndarray with pixel values representing original class IDs (0-35)
        scheme: Target class scheme

    Returns:
        np.ndarray with remapped class IDs
    """
    lut = get_remap_table(scheme)
    return lut[mask]
