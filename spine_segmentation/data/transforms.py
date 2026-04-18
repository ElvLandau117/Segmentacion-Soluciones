"""
Augmentation and preprocessing pipelines using Albumentations.
All transforms handle image+mask pairs with identical spatial transforms.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from spine_segmentation.config import IMAGENET_MEAN, IMAGENET_STD, TRAIN_CONFIG


def resize_with_padding(image: np.ndarray, mask: np.ndarray, target_size: int) -> tuple:
    """
    Resize image preserving aspect ratio (longest edge -> target_size),
    then pad shorter edge to create a square.

    Args:
        image: np.ndarray (H, W, 3) RGB image
        mask: np.ndarray (H, W) mask
        target_size: Target square size (e.g., 512)

    Returns:
        Tuple of (resized_padded_image, resized_padded_mask, padding_info)
        padding_info = (pad_top, pad_bottom, pad_left, pad_right, original_h, original_w)
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_image = cv2.copyMakeBorder(
        resized_image, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    padded_mask = cv2.copyMakeBorder(
        resized_mask, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    padding_info = (pad_top, pad_bottom, pad_left, pad_right, h, w)
    return padded_image, padded_mask, padding_info


def get_train_transforms(image_size: int = None) -> A.Compose:
    """
    Training augmentation pipeline with spatial + pixel-level transforms.

    Spatial transforms are applied identically to image AND mask.
    Pixel-level transforms are applied ONLY to the image.
    """
    image_size = image_size or TRAIN_CONFIG["image_size"]

    return A.Compose([
        # --- Spatial transforms (applied to both image and mask) ---
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=0.3,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            p=0.3,
        ),
        A.HorizontalFlip(p=0.5),
        # NOTE: NO VerticalFlip - vertebrae have strict top-to-bottom order

        # --- Pixel-level transforms (applied ONLY to image) ---
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        # --- Normalization (always applied) ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = None) -> A.Compose:
    """
    Validation/test transforms: only normalization, no augmentation.
    Resize + padding is handled in the Dataset class before this.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = None) -> A.Compose:
    """Inference transforms (same as validation)."""
    return get_val_transforms(image_size)


def denormalize_image(image_tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        image_tensor: Tensor (C, H, W) or np.ndarray (H, W, C) normalized

    Returns:
        np.ndarray (H, W, C) in [0, 255] uint8
    """
    import torch

    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # C,H,W -> H,W,C
    else:
        img = image_tensor.copy()

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
