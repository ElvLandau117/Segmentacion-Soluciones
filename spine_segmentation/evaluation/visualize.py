"""
Visualization utilities for segmentation results.
Creates overlays, comparison grids, and confusion matrices.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path
from typing import Optional

from spine_segmentation.config import MULTICLASS_COLORS, OUTPUTS_DIR
from spine_segmentation.data.transforms import denormalize_image
from spine_segmentation.data.class_mapping import get_class_names


def create_binary_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create a side-by-side visualization: original | ground truth overlay | prediction overlay.

    Args:
        image: (H, W, 3) RGB image
        mask: (H, W) binary ground truth {0, 1}
        prediction: (H, W) binary prediction {0, 1}
        alpha: Overlay transparency

    Returns:
        (H, W*3, 3) concatenated visualization
    """
    # Ground truth overlay (green)
    gt_overlay = image.copy()
    gt_mask_rgb = np.zeros_like(image)
    gt_mask_rgb[mask > 0] = [0, 255, 0]
    gt_overlay = cv2.addWeighted(gt_overlay, 1.0, gt_mask_rgb, alpha, 0)

    # Prediction overlay (blue for correct, red for errors)
    pred_overlay = image.copy()
    tp = (prediction > 0) & (mask > 0)    # True positive
    fp = (prediction > 0) & (mask == 0)   # False positive
    fn = (prediction == 0) & (mask > 0)   # False negative

    color_overlay = np.zeros_like(image)
    color_overlay[tp] = [0, 255, 0]       # Green: correct
    color_overlay[fp] = [255, 0, 0]       # Red: false positive
    color_overlay[fn] = [255, 165, 0]     # Orange: false negative
    pred_overlay = cv2.addWeighted(pred_overlay, 1.0, color_overlay, alpha, 0)

    return np.concatenate([image, gt_overlay, pred_overlay], axis=1)


def create_multiclass_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray = None,
    color_map: dict = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a color-coded overlay for multiclass segmentation.

    Args:
        image: (H, W, 3) RGB image
        mask: (H, W) integer class labels
        prediction: (H, W) predicted class labels (optional)
        color_map: dict mapping class_id -> (R, G, B) tuple
        alpha: Overlay transparency

    Returns:
        Visualization image (or concatenated if prediction is provided)
    """
    color_map = color_map or MULTICLASS_COLORS

    def apply_colormap(img, labels):
        overlay = img.copy()
        color_mask = np.zeros_like(img)
        for class_id, color in color_map.items():
            if class_id == 0:
                continue  # Skip background
            region = labels == class_id
            if region.any():
                color_mask[region] = color
        return cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

    gt_overlay = apply_colormap(image, mask)

    if prediction is not None:
        pred_overlay = apply_colormap(image, prediction)
        return np.concatenate([image, gt_overlay, pred_overlay], axis=1)

    return np.concatenate([image, gt_overlay], axis=1)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_dices: list,
    save_path: str = None,
    title: str = "Training Curves",
):
    """Plot training loss and validation Dice curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Dice curve
    ax2.plot(epochs, val_dices, 'g-', label='Val Dice', linewidth=2)
    best_epoch = np.argmax(val_dices) + 1
    best_dice = max(val_dices)
    ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_dice:.4f}\nEpoch {best_epoch}',
                xy=(best_epoch, best_dice),
                xytext=(best_epoch + 5, best_dice - 0.05),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice Score', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    plt.close()


def plot_per_class_dice(
    per_class_dice: list,
    class_names: dict = None,
    save_path: str = None,
    title: str = "Per-Class Dice Scores",
):
    """Bar chart of per-class Dice scores for multiclass segmentation."""
    if class_names is None:
        class_names = get_class_names("vertebrae_24")

    # Filter out background and classes with zero dice
    classes = []
    scores = []
    colors = []
    for i, dice in enumerate(per_class_dice):
        if i == 0:
            continue  # Skip background
        name = class_names.get(i, f"C{i}")
        classes.append(name)
        scores.append(dice)
        # Color by region
        if name.startswith("C"):
            colors.append("#e74c3c")  # Red for cervical
        elif name.startswith("T"):
            colors.append("#2ecc71")  # Green for thoracic
        elif name.startswith("L"):
            colors.append("#3498db")  # Blue for lumbar
        else:
            colors.append("#95a5a6")  # Gray for other

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(classes)), scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean(scores), color='black', linestyle='--', alpha=0.5,
               label=f'Mean: {np.mean(scores):.3f}')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        if score > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=7)

    # Legend for regions
    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='Cervical'),
        mpatches.Patch(color='#2ecc71', label='Thoracic'),
        mpatches.Patch(color='#3498db', label='Lumbar'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class Dice chart saved to {save_path}")
    plt.close()


def draw_cobb_angle_visualization(
    image: np.ndarray,
    multiclass_mask: np.ndarray,
    cobb_multiclass_result: dict,
    cobb_binary_deg: Optional[float] = None,
    scheme: str = "vertebrae_24",
) -> np.ndarray:
    """
    Clinical-style Cobb angle figure inspired by Shi et al. 2025 (Fig. 1):
    green semi-transparent boxes over the upper/lower end vertebrae,
    red tangent lines along their endplate orientations extended across
    the frame, and a header with the numeric angle.

    Args:
        image: (H, W, 3) RGB radiograph (uint8).
        multiclass_mask: (H, W) integer class labels (vertebrae_24 by default).
        cobb_multiclass_result: dict returned by `cobb_from_multiclass`. Must
            include `upper_end_vertebra`, `lower_end_vertebra`, `cobb_angle_deg`
            and `success`.
        cobb_binary_deg: optional Cobb angle from the binary method, drawn as
            a second header line for cross-reference.
        scheme: class mapping scheme used by `extract_vertebra_info`.

    Returns:
        (H, W, 3) uint8 image with overlays.
    """
    # Lazy imports to avoid circulars at module import time
    from spine_segmentation.data.class_mapping import get_class_names
    from spine_segmentation.postprocessing.vertebra_ordering import (
        compute_endplate_angles,
        extract_vertebra_info,
    )

    vis = image.copy()
    if vis.dtype != np.uint8:
        vis = (vis * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
    h, w = vis.shape[:2]

    # If multiclass failed, fall back to a text-only annotation so the panel
    # still carries information (the binary method is usually still available).
    if not cobb_multiclass_result or not cobb_multiclass_result.get("success"):
        if cobb_binary_deg is not None:
            cv2.putText(
                vis, f"Cobb (Binary): {cobb_binary_deg:.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
        cv2.putText(
            vis, "Multiclass Cobb visualization unavailable",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        return vis

    upper_name = cobb_multiclass_result.get("upper_end_vertebra")
    lower_name = cobb_multiclass_result.get("lower_end_vertebra")
    cobb_deg = float(cobb_multiclass_result.get("cobb_angle_deg", 0.0))

    # Re-extract per-vertebra info to access bbox + region for drawing.
    # `cobb_from_multiclass` discards bbox/region in its returned dict.
    class_names = get_class_names(scheme)
    vertebrae = extract_vertebra_info(multiclass_mask, class_names)
    vertebrae = compute_endplate_angles(vertebrae)

    upper_v = next((v for v in vertebrae if v["name"] == upper_name), None)
    lower_v = next((v for v in vertebrae if v["name"] == lower_name), None)

    if upper_v is None or lower_v is None:
        cv2.putText(
            vis, f"Cobb (Multiclass): {cobb_deg:.1f} deg",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        if cobb_binary_deg is not None:
            cv2.putText(
                vis, f"Cobb (Binary): {cobb_binary_deg:.1f} deg",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
        return vis

    # 1) Green semi-transparent fill over the two end vertebrae
    color_mask = np.zeros_like(vis)
    for v in (upper_v, lower_v):
        region = (multiclass_mask == v["class_id"]).astype(np.uint8)
        color_mask[region > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 1.0, color_mask, 0.45, 0)

    # 2) Bounding-box outline for emphasis
    for v in (upper_v, lower_v):
        min_row, min_col, max_row, max_col = v["bbox"]
        cv2.rectangle(vis, (min_col, min_row), (max_col, max_row), (0, 200, 0), 2)

    # 3) Tangent lines along each endplate direction, extended across the frame.
    # skimage orientation is the angle (rad) between the row-axis (vertical)
    # and the major axis, CCW. For a horizontal vertebra orientation ~= pi/2,
    # so (sin, cos) gives a horizontal tangent. We extend in both directions.
    line_len = max(w, h)
    for v in (upper_v, lower_v):
        theta = float(v["orientation"])
        dx = np.sin(theta)
        dy = np.cos(theta)
        cx = float(v["centroid_x"])
        cy = float(v["centroid_y"])
        p1 = (int(round(cx - dx * line_len)), int(round(cy - dy * line_len)))
        p2 = (int(round(cx + dx * line_len)), int(round(cy + dy * line_len)))
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    # 4) Header with both Cobb readings (multiclass = the curve shown, binary = robust check)
    cv2.putText(
        vis, f"Cobb (Multiclass): {cobb_deg:.1f} deg  ({upper_name} - {lower_name})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    if cobb_binary_deg is not None:
        cv2.putText(
            vis, f"Cobb (Binary): {cobb_binary_deg:.1f} deg  (more robust)",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

    # 5) Labels next to each end vertebra (white text, placed to the right of the bbox)
    for v, label in (
        (upper_v, f"Superior end vertebra ({upper_name})"),
        (lower_v, f"Inferior end vertebra ({lower_name})"),
    ):
        min_row, min_col, max_row, max_col = v["bbox"]
        text_x = min(max_col + 5, w - 220)
        text_y = max(20, (min_row + max_row) // 2)
        cv2.putText(
            vis, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

    return vis


def save_prediction_grid(
    images: list,
    masks: list,
    predictions: list,
    names: list,
    task: str = "binary",
    save_path: str = None,
    max_samples: int = 6,
):
    """Save a grid of prediction visualizations."""
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = images[i]
        mask = masks[i]
        pred = predictions[i]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{names[i]} - Original', fontsize=10)
        axes[i, 0].axis('off')

        if task == "binary":
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth', fontsize=10)
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction', fontsize=10)
        else:
            axes[i, 1].imshow(mask, cmap='nipy_spectral', vmin=0, vmax=23)
            axes[i, 1].set_title('Ground Truth', fontsize=10)
            axes[i, 2].imshow(pred, cmap='nipy_spectral', vmin=0, vmax=23)
            axes[i, 2].set_title('Prediction', fontsize=10)

        axes[i, 1].axis('off')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction grid saved to {save_path}")
    plt.close()
