"""
Loss functions for binary and multiclass segmentation.
Includes Dice, Focal, and combined losses with class weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) raw logits
            target: (B, 1, H, W) binary mask {0, 1}
        """
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BinaryComboLoss(nn.Module):
    """
    Combined BCE + Dice Loss for binary segmentation.
    Same approach as previous semester but cleaner implementation.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for multiclass segmentation.
    Handles class imbalance by weighting each class by inverse volume.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) raw logits
            target: (B, H, W) integer class labels
        """
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)  # (B, C, H, W)

        # One-hot encode target: (B, H, W) -> (B, C, H, W)
        target_onehot = F.one_hot(target.long(), num_classes)  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute per-class weights (inverse volume)
        class_volumes = target_onehot.sum(dim=(0, 2, 3))  # (C,)
        weights = 1.0 / (class_volumes ** 2 + self.smooth)

        # Generalized Dice
        intersection = (pred_softmax * target_onehot).sum(dim=(0, 2, 3))  # (C,)
        denominator = (pred_softmax + target_onehot).sum(dim=(0, 2, 3))    # (C,)

        gdl = 1.0 - (2.0 * (weights * intersection).sum() + self.smooth) / (
            (weights * denominator).sum() + self.smooth
        )
        return gdl


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance in multiclass segmentation.
    Focuses on hard-to-classify pixels by down-weighting easy examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) raw logits
            target: (B, H, W) integer class labels
        """
        ce_loss = F.cross_entropy(
            pred, target, weight=self.class_weights, reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MulticlassComboLoss(nn.Module):
    """
    Combined Weighted Cross-Entropy + Generalized Dice Loss for multiclass segmentation.
    Primary loss function for vertebrae segmentation.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        self.gdl = GeneralizedDiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights)
        dice_loss = self.gdl(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def compute_class_weights(dataset, num_classes: int, device: str = "cpu") -> torch.Tensor:
    """
    Compute class weights based on inverse frequency (median frequency balancing).
    Should be run once on the training set.

    Args:
        dataset: PyTorch Dataset (multiclass)
        num_classes: Number of classes
        device: Device for the weight tensor

    Returns:
        torch.Tensor of shape (num_classes,) with class weights
    """
    print("Computing class weights from training data...")
    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["mask"].numpy()
        for c in range(num_classes):
            pixel_counts[c] += (mask == c).sum()

    # Median frequency balancing
    total_pixels = pixel_counts.sum()
    frequencies = pixel_counts / total_pixels
    median_freq = np.median(frequencies[frequencies > 0])

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if frequencies[c] > 0:
            weights[c] = median_freq / frequencies[c]
        else:
            weights[c] = 0.0  # Class not present

    # Clip extreme weights
    weights = np.clip(weights, 0.1, 10.0)

    print(f"  Class weights: min={weights.min():.3f}, max={weights.max():.3f}")
    for c in range(min(num_classes, 24)):
        if pixel_counts[c] > 0:
            print(f"    Class {c}: count={pixel_counts[c]:,}, freq={frequencies[c]:.6f}, weight={weights[c]:.3f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_loss_function(
    task: str = "binary",
    class_weights: torch.Tensor = None,
) -> nn.Module:
    """
    Factory function to get the appropriate loss.

    Args:
        task: 'binary' or 'multiclass'
        class_weights: Optional class weights tensor for multiclass

    Returns:
        Loss module
    """
    if task == "binary":
        return BinaryComboLoss(bce_weight=0.5, dice_weight=0.5)
    else:
        return MulticlassComboLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=class_weights,
        )
