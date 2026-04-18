"""
Segmentation evaluation metrics: Dice, IoU, Pixel Accuracy, per-class metrics.
"""

import torch
import numpy as np
from typing import Optional


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient between two binary arrays."""
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard Index)."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel-level accuracy."""
    correct = (pred == target).sum()
    total = target.size
    return correct / max(total, 1)


def compute_metrics_binary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute metrics for binary segmentation.

    Args:
        predictions: (B, 1, H, W) sigmoid probabilities or logits
        targets: (B, 1, H, W) binary masks {0, 1}
        threshold: Threshold for converting probabilities to binary

    Returns:
        dict with dice, iou, pixel_acc averages
    """
    if predictions.dim() == 4 and predictions.shape[1] == 1:
        preds = predictions.squeeze(1)  # (B, H, W)
        tgts = targets.squeeze(1)       # (B, H, W)
    else:
        preds = predictions
        tgts = targets

    # Apply sigmoid if needed (check range)
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)

    preds_binary = (preds > threshold).float().numpy()
    tgts_np = tgts.float().numpy()

    dices, ious, accs = [], [], []

    for i in range(len(preds_binary)):
        dices.append(dice_coefficient(preds_binary[i], tgts_np[i]))
        ious.append(iou_score(preds_binary[i], tgts_np[i]))
        accs.append(pixel_accuracy(preds_binary[i], tgts_np[i]))

    return {
        "val_dice": np.mean(dices),
        "val_dice_mean": np.mean(dices),
        "val_iou_mean": np.mean(ious),
        "val_pixel_acc": np.mean(accs),
        "val_dice_std": np.std(dices),
    }


def compute_metrics_multiclass(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> dict:
    """
    Compute metrics for multiclass segmentation.

    Args:
        predictions: (B, C, H, W) raw logits
        targets: (B, H, W) integer class labels
        num_classes: Number of classes
        ignore_background: If True, exclude class 0 from mean Dice/IoU

    Returns:
        dict with per-class and mean metrics
    """
    preds_classes = predictions.argmax(dim=1).numpy()  # (B, H, W)
    tgts_np = targets.numpy()                           # (B, H, W)

    # Per-class Dice and IoU across all images
    class_dices = np.zeros(num_classes)
    class_ious = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)  # How many images contain each class

    for c in range(num_classes):
        pred_c = (preds_classes == c).astype(np.float32)
        target_c = (tgts_np == c).astype(np.float32)

        # Only compute for images where the class is present in ground truth
        for i in range(len(tgts_np)):
            if target_c[i].sum() > 0:
                class_dices[c] += dice_coefficient(pred_c[i], target_c[i])
                class_ious[c] += iou_score(pred_c[i], target_c[i])
                class_counts[c] += 1

    # Average per-class metrics (only classes that appear)
    valid_mask = class_counts > 0
    class_dices_avg = np.zeros(num_classes)
    class_ious_avg = np.zeros(num_classes)
    class_dices_avg[valid_mask] = class_dices[valid_mask] / class_counts[valid_mask]
    class_ious_avg[valid_mask] = class_ious[valid_mask] / class_counts[valid_mask]

    # Mean Dice/IoU (optionally excluding background)
    start_class = 1 if ignore_background else 0
    valid_fg = valid_mask.copy()
    if ignore_background:
        valid_fg[0] = False

    mean_dice = class_dices_avg[valid_fg].mean() if valid_fg.any() else 0.0
    mean_iou = class_ious_avg[valid_fg].mean() if valid_fg.any() else 0.0

    # Overall pixel accuracy
    pixel_acc_val = pixel_accuracy(preds_classes, tgts_np)

    metrics = {
        "val_dice_mean": float(mean_dice),
        "val_iou_mean": float(mean_iou),
        "val_pixel_acc": float(pixel_acc_val),
        "per_class_dice": class_dices_avg.tolist(),
        "per_class_iou": class_ious_avg.tolist(),
    }

    return metrics


def compute_test_metrics(
    model: torch.nn.Module,
    test_loader,
    task: str = "binary",
    num_classes: int = 1,
    device: str = "cuda",
) -> dict:
    """
    Run full evaluation on the test set.

    Returns:
        dict with comprehensive test metrics
    """
    model.eval()
    all_preds = []
    all_masks = []
    all_names = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                predictions = model(images)

            if task == "binary":
                preds = torch.sigmoid(predictions).cpu()
            else:
                preds = predictions.cpu()

            all_preds.append(preds)
            all_masks.append(masks)
            all_names.extend(batch["image_name"])

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    if task == "binary":
        metrics = compute_metrics_binary(all_preds, all_masks)
    else:
        metrics = compute_metrics_multiclass(all_preds, all_masks, num_classes)

    # Rename keys from val_ to test_
    test_metrics = {}
    for key, value in metrics.items():
        new_key = key.replace("val_", "test_")
        test_metrics[new_key] = value
    test_metrics["image_names"] = all_names

    return test_metrics


def print_metrics_table(metrics: dict, class_names: dict = None):
    """Print a formatted metrics table."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    for key in ["test_dice_mean", "val_dice_mean", "test_iou_mean", "val_iou_mean",
                "test_pixel_acc", "val_pixel_acc"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    # Per-class metrics
    if "per_class_dice" in metrics and class_names:
        print("\n  Per-class Dice scores:")
        for i, dice in enumerate(metrics["per_class_dice"]):
            name = class_names.get(i, f"class_{i}")
            if dice > 0:
                print(f"    {name:20s}: {dice:.4f}")

    print("=" * 50)
