"""
Stratified train/val/test split for the MaIA Scoliosis Dataset.
Ensures reproducible splits with balanced Normal/Scoliosis ratios.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from spine_segmentation.config import (
    DATASET_INDEX_CSV,
    SPLITS_FILE,
    TRAIN_CONFIG,
)


def create_splits(
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    random_seed: int = None,
    save_path: Path = None,
) -> dict:
    """
    Create stratified train/val/test splits from the dataset index.

    Args:
        train_ratio: Fraction for training (default from config)
        val_ratio: Fraction for validation (default from config)
        test_ratio: Fraction for test (default from config)
        random_seed: Random seed for reproducibility
        save_path: Path to save the split assignments as JSON

    Returns:
        dict with keys 'train', 'val', 'test', each containing a list of image names
    """
    train_ratio = train_ratio or TRAIN_CONFIG["train_ratio"]
    val_ratio = val_ratio or TRAIN_CONFIG["val_ratio"]
    test_ratio = test_ratio or TRAIN_CONFIG["test_ratio"]
    random_seed = random_seed or TRAIN_CONFIG["random_seed"]
    save_path = save_path or SPLITS_FILE

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Load dataset index
    df = pd.read_csv(DATASET_INDEX_CSV)
    images = df["image"].values
    conditions = df["split"].values  # "Normal" or "Scoliosis"

    n_total = len(images)
    print(f"Total samples: {n_total}")
    print(f"  Normal: {(conditions == 'Normal').sum()}")
    print(f"  Scoliosis: {(conditions == 'Scoliosis').sum()}")

    # Step 1: Split into train+val vs test (stratified)
    test_size = test_ratio
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    trainval_idx, test_idx = next(sss_test.split(images, conditions))

    # Step 2: Split train+val into train vs val (stratified)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_seed)
    trainval_images = images[trainval_idx]
    trainval_conditions = conditions[trainval_idx]
    train_idx_local, val_idx_local = next(sss_val.split(trainval_images, trainval_conditions))

    train_idx = trainval_idx[train_idx_local]
    val_idx = trainval_idx[val_idx_local]

    # Build split dict
    splits = {
        "train": images[train_idx].tolist(),
        "val": images[val_idx].tolist(),
        "test": images[test_idx].tolist(),
    }

    # Print statistics
    for split_name, split_images in splits.items():
        split_conditions = conditions[np.isin(images, split_images)]
        n_normal = (split_conditions == "Normal").sum()
        n_scoliosis = (split_conditions == "Scoliosis").sum()
        print(f"  {split_name}: {len(split_images)} images "
              f"(Normal={n_normal}, Scoliosis={n_scoliosis})")

    # Save to JSON
    if save_path:
        save_path = Path(save_path)
        split_data = {
            "random_seed": random_seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "splits": splits,
        }
        with open(save_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Splits saved to {save_path}")

    return splits


def load_splits(path: Path = None) -> dict:
    """
    Load previously saved split assignments.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    path = path or SPLITS_FILE
    if not Path(path).exists():
        print(f"Splits file not found at {path}. Creating new splits...")
        return create_splits(save_path=path)

    with open(path, "r") as f:
        data = json.load(f)

    splits = data["splits"]
    print(f"Loaded splits from {path}")
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return splits


if __name__ == "__main__":
    splits = create_splits()
