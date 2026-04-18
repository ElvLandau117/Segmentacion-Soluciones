"""
PyTorch Dataset classes for binary and multiclass spine segmentation.
Handles loading radiographs and their corresponding segmentation masks.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from spine_segmentation.config import (
    DATASET_ROOT,
    DATASET_INDEX_CSV,
    TRAIN_CONFIG,
)
from spine_segmentation.data.class_mapping import remap_mask, get_num_classes
from spine_segmentation.data.transforms import (
    resize_with_padding,
    get_train_transforms,
    get_val_transforms,
)
from spine_segmentation.data.splits import load_splits


class SpineBinaryDataset(Dataset):
    """
    PyTorch Dataset for binary spine segmentation.
    Returns: image (3, H, W), mask (1, H, W) with values {0, 1}.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = None,
        transforms=None,
        splits_dict: dict = None,
    ):
        """
        Args:
            split: One of 'train', 'val', 'test'
            image_size: Target image size (default from config)
            transforms: Albumentations transform pipeline (auto-selected if None)
            splits_dict: Pre-loaded splits dict (loads from file if None)
        """
        self.split = split
        self.image_size = image_size or TRAIN_CONFIG["image_size"]

        # Load dataset index
        self.df = pd.read_csv(DATASET_INDEX_CSV)

        # Filter by split
        splits = splits_dict or load_splits()
        split_images = splits[split]
        self.df = self.df[self.df["image"].isin(split_images)].reset_index(drop=True)

        # Set transforms
        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(self.image_size)
        else:
            self.transforms = get_val_transforms(self.image_size)

        print(f"BinaryDataset [{split}]: {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image (RGB)
        img_path = str(DATASET_ROOT / row["radiograph_path"])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load binary mask (grayscale)
        mask_path = str(DATASET_ROOT / row["label_binary_path"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Threshold mask to {0, 1} (JPG compression artifacts)
        mask = (mask > 127).astype(np.uint8)

        # Resize with padding to preserve aspect ratio
        image, mask, padding_info = resize_with_padding(image, mask, self.image_size)

        # Apply augmentations
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]        # (3, H, W) float32
        mask_tensor = augmented["mask"]           # (H, W) uint8

        # Mask to float and add channel dim: (H, W) -> (1, H, W)
        mask_tensor = mask_tensor.unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": row["image"],
            "condition": row["split"],  # Normal or Scoliosis
        }


class SpineMulticlassDataset(Dataset):
    """
    PyTorch Dataset for multiclass vertebrae segmentation.
    Returns: image (3, H, W), mask (H, W) with integer class labels.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = None,
        scheme: str = "vertebrae_24",
        transforms=None,
        splits_dict: dict = None,
    ):
        """
        Args:
            split: One of 'train', 'val', 'test'
            image_size: Target image size (default from config)
            scheme: Class mapping scheme ('vertebrae_24', 'full_36', 'regional_5')
            transforms: Albumentations transform pipeline
            splits_dict: Pre-loaded splits dict
        """
        self.split = split
        self.image_size = image_size or TRAIN_CONFIG["image_size"]
        self.scheme = scheme
        self.num_classes = get_num_classes(scheme)

        # Load dataset index
        self.df = pd.read_csv(DATASET_INDEX_CSV)

        # Filter by split
        splits = splits_dict or load_splits()
        split_images = splits[split]
        self.df = self.df[self.df["image"].isin(split_images)].reset_index(drop=True)

        # Set transforms
        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(self.image_size)
        else:
            self.transforms = get_val_transforms(self.image_size)

        print(f"MulticlassDataset [{split}] (scheme={scheme}, "
              f"classes={self.num_classes}): {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image (RGB)
        img_path = str(DATASET_ROOT / row["radiograph_path"])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load multiclass mask - CRITICAL: use IMREAD_UNCHANGED to preserve IDs
        mask_path = str(DATASET_ROOT / row["multiclass_id_png"])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Handle multi-channel mask (some PNGs might load as 3-channel)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Remap classes based on scheme
        mask = remap_mask(mask, self.scheme)

        # Resize with padding
        image, mask, padding_info = resize_with_padding(image, mask, self.image_size)

        # Apply augmentations
        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]        # (3, H, W) float32
        mask_tensor = augmented["mask"]           # (H, W) uint8

        # Mask to long for CrossEntropyLoss
        mask_tensor = mask_tensor.long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": row["image"],
            "condition": row["split"],
        }


def get_dataloaders(
    task: str = "binary",
    scheme: str = "vertebrae_24",
    batch_size: int = None,
    num_workers: int = None,
    splits_dict: dict = None,
) -> dict:
    """
    Create DataLoaders for train, val, and test sets.

    Args:
        task: 'binary' or 'multiclass'
        scheme: Class mapping scheme (only for multiclass)
        batch_size: Batch size (default from config)
        num_workers: Number of data loading workers
        splits_dict: Pre-loaded splits dict

    Returns:
        dict with keys 'train', 'val', 'test', each a DataLoader
    """
    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    num_workers = num_workers or TRAIN_CONFIG["num_workers"]
    splits_dict = splits_dict or load_splits()

    DatasetClass = SpineBinaryDataset if task == "binary" else SpineMulticlassDataset

    loaders = {}
    for split_name in ["train", "val", "test"]:
        kwargs = {
            "split": split_name,
            "splits_dict": splits_dict,
        }
        if task == "multiclass":
            kwargs["scheme"] = scheme

        dataset = DatasetClass(**kwargs)

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=TRAIN_CONFIG["pin_memory"],
            drop_last=(split_name == "train"),
        )

    return loaders
