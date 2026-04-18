"""
Central configuration for the Spine Segmentation project.
All hyperparameters, paths, and constants are defined here.
"""

import os
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "MaIA_Scoliosis_Dataset"

# Dataset subdirectories
NORMAL_DIR = DATASET_ROOT / "Normal"
SCOLIOSIS_DIR = DATASET_ROOT / "Scoliosis"
BINARY_MASK_DIR = DATASET_ROOT / "LabelBinaryJPG"
MULTICLASS_ID_DIR = DATASET_ROOT / "LabelMultiClass_ID_PNG"
MULTICLASS_GRAY_DIR = DATASET_ROOT / "LabelMultiClass_Gray_JPG"
MULTICLASS_COLOR_DIR = DATASET_ROOT / "LabelMultiClass_Color_JPG"
METRICS_DIR = DATASET_ROOT / "RadiographMetrics"

# Metadata files
DATASET_INDEX_CSV = DATASET_ROOT / "dataset_index.csv"
LABELS_DICT_JSON = DATASET_ROOT / "labels_dictionary.json"
RADIOGRAPH_METRICS_CSV = METRICS_DIR / "radiograph_metrics.csv"

# Output directories
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
SPLITS_FILE = PROJECT_ROOT / "data_splits.json"

# Ensure output directories exist
CHECKPOINTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Class Definitions (from labels_dictionary.json)
# =============================================================================

# Full 36-class mapping: pixel_id -> label_name
FULL_CLASS_NAMES = {
    0: "background",
    1: "C7", 2: "C6", 3: "C5", 4: "C4", 5: "C3",       # Cervical (reverse order)
    6: "T1", 7: "T2", 8: "T3", 9: "T4", 10: "T5",       # Thoracic
    11: "T6", 12: "T7", 13: "T8", 14: "T9", 15: "T10",
    16: "T11", 17: "T12",
    18: "L1", 19: "L2", 20: "L3", 21: "L4", 22: "L5",   # Lumbar
    23: "Entity_10", 24: "Entity_12", 25: "Entity_13",    # Anatomical entities
    26: "Entity_14", 27: "Entity_16", 28: "Entity_17",
    29: "Entity_18", 30: "Entity_21", 31: "Entity_3",
    32: "Entity_6", 33: "Entity_8",
    34: "Entity_34", 35: "Entity_35",
}

# Vertebrae-only 24-class scheme (primary approach)
# Maps original IDs -> new IDs (entities collapsed to background)
VERTEBRAE_CLASS_NAMES = {
    0: "background",
    1: "C7", 2: "C6", 3: "C5", 4: "C4", 5: "C3",
    6: "T1", 7: "T2", 8: "T3", 9: "T4", 10: "T5",
    11: "T6", 12: "T7", 13: "T8", 14: "T9", 15: "T10",
    16: "T11", 17: "T12",
    18: "L1", 19: "L2", 20: "L3", 21: "L4", 22: "L5",
    23: "other_structures",
}

# Anatomical ordering: vertebrae from top to bottom
VERTEBRAE_ORDER = [
    "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
]

# Color map for multiclass visualization (from labels_dictionary.json)
MULTICLASS_COLORS = {
    0: (0, 0, 0),         # background - black
    1: (242, 12, 12),     # C7 - red
    2: (242, 51, 12),     # C6
    3: (242, 91, 12),     # C5
    4: (242, 130, 12),    # C4
    5: (242, 169, 12),    # C3
    6: (242, 209, 12),    # T1
    7: (235, 242, 12),    # T2
    8: (196, 242, 12),    # T3
    9: (156, 242, 12),    # T4
    10: (117, 242, 12),   # T5
    11: (77, 242, 12),    # T6
    12: (38, 242, 12),    # T7
    13: (12, 242, 25),    # T8
    14: (12, 242, 64),    # T9
    15: (12, 242, 104),   # T10
    16: (12, 242, 143),   # T11
    17: (12, 242, 183),   # T12
    18: (12, 242, 222),   # L1
    19: (12, 222, 242),   # L2
    20: (12, 183, 242),   # L3
    21: (12, 143, 242),   # L4
    22: (12, 104, 242),   # L5
    23: (128, 128, 128),  # other_structures - gray
}

# =============================================================================
# Training Hyperparameters
# =============================================================================
TRAIN_CONFIG = {
    # Data
    "image_size": 512,
    "batch_size": 12,
    "num_workers": 4,
    "pin_memory": True,

    # Split
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,

    # Optimization
    "num_epochs": 150,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-4,
    "weight_decay": 1e-4,
    "early_stopping_patience": 20,

    # Scheduler
    "scheduler_T0": 10,
    "scheduler_T_mult": 2,
    "scheduler_eta_min": 1e-6,

    # Mixed precision
    "use_amp": True,

    # Gradient accumulation (if needed for larger batch sizes)
    "accumulation_steps": 1,
}

# =============================================================================
# Model Configurations
# =============================================================================
MODEL_CONFIGS = {
    "unet_resnet50": {
        "architecture": "Unet",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
    },
    "unet_efficientnet_b4": {
        "architecture": "Unet",
        "encoder_name": "efficientnet-b4",
        "encoder_weights": "imagenet",
    },
    "deeplabv3plus_resnet50": {
        "architecture": "DeepLabV3Plus",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
    },
    "unet_mit_b3": {
        "architecture": "Unet",
        "encoder_name": "mit_b3",
        "encoder_weights": "imagenet",
    },
    "manet_mit_b5": {
        "architecture": "MAnet",
        "encoder_name": "mit_b5",
        "encoder_weights": "imagenet",
    },
}

# =============================================================================
# ImageNet Normalization (required for pretrained encoders)
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# =============================================================================
# MLflow Configuration
# =============================================================================
MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "spine_segmentation"

# =============================================================================
# Number of classes per scheme
# =============================================================================
NUM_CLASSES_BINARY = 1       # spine vs background
NUM_CLASSES_VERTEBRAE = 24   # 23 vertebrae + 1 other_structures + background = 24
NUM_CLASSES_FULL = 36        # all original classes
