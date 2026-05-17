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


# ----------------------------------------------------------------------------
# Cobb angle visualization helpers (Ciclo 5.1)
# ----------------------------------------------------------------------------

def _endplate_vectors(orientation_rad: float) -> tuple:
    """Return (tangent, perpendicular) unit vectors for a vertebra's endplate
    in image coords (y-down).

    skimage `regionprops.orientation` is the CCW angle (radians) between the
    row-axis (y) and the major axis of the equivalent ellipse, in [-pi/2, pi/2].
    For a horizontal vertebra the major axis is horizontal and orientation~=pi/2,
    so:
      - Tangent ALONG the endplate: (sin theta, cos theta)
      - Perpendicular to the endplate (the Cobb line): (cos theta, -sin theta)
    """
    s = float(np.sin(orientation_rad))
    c = float(np.cos(orientation_rad))
    return (s, c), (c, -s)


def _line_intersection(p1, d1, p2, d2):
    """Intersection of two infinite lines defined by point + direction vector.
    Returns (xi, yi) or None when the lines are parallel within tolerance."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-6:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _angle_deg_of(vec) -> float:
    """Angle in degrees of `vec` measured from +x axis, CW in image coords (y-down).
    cv2.ellipse uses the same convention, so this is what we feed it."""
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def _draw_endplate_marker(vis, centroid, tan_vec, length, color, thickness=2):
    """Short line segment ALONG the endplate centered on `centroid` (in pixels)."""
    cx, cy = centroid
    half = length / 2.0
    p1 = (int(round(cx - tan_vec[0] * half)), int(round(cy - tan_vec[1] * half)))
    p2 = (int(round(cx + tan_vec[0] * half)), int(round(cy + tan_vec[1] * half)))
    cv2.line(vis, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def _draw_angle_arc(vis, center, perp_u, perp_l, radius, angle_deg, color):
    """Draw an arc at `center` between the two perpendiculars, labeled with the angle."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    a1 = _angle_deg_of(perp_u)
    a2 = _angle_deg_of(perp_l)
    start = min(a1, a2)
    end = max(a1, a2)
    if end - start > 180:
        start, end = end, start + 360
    cv2.ellipse(vis, (cx, cy), (int(radius), int(radius)),
                0, start, end, color, 2, lineType=cv2.LINE_AA)
    mid = (start + end) / 2.0
    lx = int(cx + (radius + 18) * np.cos(np.radians(mid)))
    ly = int(cy + (radius + 18) * np.sin(np.radians(mid)))
    cv2.putText(vis, f"{angle_deg:.1f}deg", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)


def _draw_speedometer(vis, angle_deg, position=None):
    """Mini gauge in the bottom-left for tiny Cobb angles where the in-frame
    arc would be unreadable. Renders a 4x-scaled visual angle plus the numeric
    value so the user still gets the message even when the lines are near-parallel.
    """
    h, w = vis.shape[:2]
    radius = 45
    cx = position[0] if position else 65
    cy = position[1] if position else h - 75
    cv2.circle(vis, (cx, cy), radius, (40, 40, 40), -1)
    cv2.circle(vis, (cx, cy), radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(vis, (cx - radius + 5, cy), (cx + radius - 5, cy), (200, 200, 200), 1)
    # Scale 4x so a 4-deg angle still moves the needle visibly.
    visual_angle = float(np.clip(angle_deg * 4.0, -90.0, 90.0))
    rad = np.radians(visual_angle)
    tip = (int(cx + (radius - 8) * np.cos(rad)),
           int(cy + (radius - 8) * np.sin(rad)))
    cv2.line(vis, (cx, cy), tip, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, "Cobb", (cx - 18, cy - radius - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(vis, f"{angle_deg:.1f}deg", (cx - radius + 2, cy + radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def _draw_binary_overlay(vis, cobb_binary_result):
    """Overlay the binary-method internals: fitted spline (thin white polyline)
    and the 2 inflection points (filled yellow circles). Educational + makes
    the binary Cobb number visually traceable."""
    if not cobb_binary_result or not cobb_binary_result.get("success"):
        return
    spline_x = cobb_binary_result.get("spline_x")
    spline_y = cobb_binary_result.get("spline_y")
    if spline_x and spline_y and len(spline_x) > 1:
        pts = np.array(list(zip(spline_x, spline_y)), dtype=np.int32)
        cv2.polylines(vis, [pts], False, (235, 235, 235), 1, lineType=cv2.LINE_AA)
    for (ix, iy) in cobb_binary_result.get("inflection_points") or []:
        cv2.circle(vis, (int(ix), int(iy)), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (int(ix), int(iy)), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)


def draw_cobb_angle_visualization(
    image: np.ndarray,
    multiclass_mask: np.ndarray,
    cobb_multiclass_result: dict,
    cobb_binary_result: Optional[dict] = None,
    scheme: str = "vertebrae_24",
) -> np.ndarray:
    """Clinical-style Cobb angle figure.

    Layout inspired by Fig 1 of Shi et al. 2025 ("Accurate Cobb Angle
    Estimation via SVD-Based Curve Detection and Vertebral Wedging
    Quantification", IEEE J-BHI, arXiv:2509.24898):

      - Green semi-transparent boxes over the upper / lower end vertebrae.
      - Cyan short markers ALONG each endplate (inside the boxes).
      - Red PERPENDICULAR lines to the endplates that meet at the Cobb
        intersection, with an arc drawn at the intersection.
      - A small "Cobb gauge" in the bottom-left when the angle is too
        small (<8 deg) for the in-frame arc to be readable.

    Adapted to our pipeline: the end vertebrae come from
    `cobb_from_multiclass` (segmentation-based, not landmark-based like
    the paper). We additionally overlay the binary-method internals
    (fitted spline + 2 inflection points) because our dataset only has
    segmentation masks; the spline-based binary Cobb is our equivalent
    of the paper's landmark-based curve detection.

    Args:
        image: (H, W, 3) RGB radiograph (uint8 or float in [0,1]).
        multiclass_mask: (H, W) integer class labels (vertebrae_24).
        cobb_multiclass_result: dict from `cobb_from_multiclass` — must
            carry `success`, `upper_end_vertebra`, `lower_end_vertebra`,
            `cobb_angle_deg`.
        cobb_binary_result: dict from `cobb_from_binary`, optional. When
            successful, the spline + inflection points are overlaid and
            the binary angle is shown in the header.
        scheme: class mapping scheme (default vertebrae_24).

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

    cobb_binary_deg = None
    if cobb_binary_result and cobb_binary_result.get("success"):
        cobb_binary_deg = float(cobb_binary_result["cobb_angle_deg"])

    # If multiclass failed: text-only fallback + binary overlay if we have it.
    if not cobb_multiclass_result or not cobb_multiclass_result.get("success"):
        _draw_binary_overlay(vis, cobb_binary_result)
        if cobb_binary_deg is not None:
            cv2.putText(
                vis, f"Cobb (Binary): {cobb_binary_deg:.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
            _draw_speedometer(vis, cobb_binary_deg)
        cv2.putText(
            vis, "Multiclass Cobb visualization unavailable",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        return vis

    upper_name = cobb_multiclass_result.get("upper_end_vertebra")
    lower_name = cobb_multiclass_result.get("lower_end_vertebra")
    cobb_deg = float(cobb_multiclass_result.get("cobb_angle_deg", 0.0))

    # Re-extract per-vertebra info to access bbox + orientation for drawing.
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
        _draw_binary_overlay(vis, cobb_binary_result)
        return vis

    # 1) Green semi-transparent fill over the two end vertebrae masks.
    color_mask = np.zeros_like(vis)
    for v in (upper_v, lower_v):
        region = (multiclass_mask == v["class_id"]).astype(np.uint8)
        color_mask[region > 0] = [0, 200, 0]
    vis = cv2.addWeighted(vis, 1.0, color_mask, 0.45, 0)

    # 2) Bbox outlines.
    for v in (upper_v, lower_v):
        min_row, min_col, max_row, max_col = v["bbox"]
        cv2.rectangle(vis, (min_col, min_row), (max_col, max_row), (0, 220, 0), 2)

    # 3) Endplate markers (cyan) along the endplate, inside each box.
    # 4) Collect perpendicular vectors for the angle calculation.
    perp_vecs = []
    centroids = []
    for v in (upper_v, lower_v):
        tan_vec, perp_vec = _endplate_vectors(float(v["orientation"]))
        cx = float(v["centroid_x"])
        cy = float(v["centroid_y"])
        box_width = v["bbox"][3] - v["bbox"][1]
        marker_len = max(20.0, box_width * 0.7)
        _draw_endplate_marker(vis, (cx, cy), tan_vec, marker_len,
                              (0, 255, 255), thickness=2)
        perp_vecs.append(perp_vec)
        centroids.append((cx, cy))

    # 5) Intersection of the perpendiculars.
    intersection = _line_intersection(
        centroids[0], perp_vecs[0],
        centroids[1], perp_vecs[1],
    )
    intersection_in_frame = (
        intersection is not None
        and 0 <= intersection[0] < w
        and 0 <= intersection[1] < h
    )

    # 6) Perpendicular red lines: from each centroid to the intersection if in-frame,
    # otherwise a fixed-length ray oriented toward the other centroid.
    if intersection_in_frame:
        for cen in centroids:
            cv2.line(
                vis,
                (int(round(cen[0])), int(round(cen[1]))),
                (int(round(intersection[0])), int(round(intersection[1]))),
                (0, 0, 255), 2, lineType=cv2.LINE_AA,
            )
    else:
        fixed_len = 180.0
        for cen, vec in zip(centroids, perp_vecs):
            other = centroids[1] if cen is centroids[0] else centroids[0]
            sign = 1.0 if (vec[0] * (other[0] - cen[0])
                           + vec[1] * (other[1] - cen[1])) > 0 else -1.0
            p2 = (int(round(cen[0] + sign * vec[0] * fixed_len)),
                  int(round(cen[1] + sign * vec[1] * fixed_len)))
            cv2.line(
                vis,
                (int(round(cen[0])), int(round(cen[1]))),
                p2,
                (0, 0, 255), 2, lineType=cv2.LINE_AA,
            )

    # 7) Arc at the intersection (skip when angle is too small to be readable).
    if intersection_in_frame and cobb_deg > 1.0:
        radius = float(np.clip(30.0 + cobb_deg * 1.5, 35.0, 90.0))
        _draw_angle_arc(vis, intersection, perp_vecs[0], perp_vecs[1],
                        radius, cobb_deg, (0, 0, 255))

    # 8) Speedometer fallback for tiny angles or out-of-frame intersections.
    if cobb_deg < 8.0 or not intersection_in_frame:
        _draw_speedometer(vis, cobb_deg)

    # 9) Headers (both Cobb readings).
    cv2.putText(
        vis, f"Cobb (Multiclass): {cobb_deg:.1f} deg  ({upper_name} - {lower_name})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        lineType=cv2.LINE_AA,
    )
    if cobb_binary_deg is not None:
        cv2.putText(
            vis, f"Cobb (Binary): {cobb_binary_deg:.1f} deg  (more robust)",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            lineType=cv2.LINE_AA,
        )

    # 10) End-vertebra labels next to the boxes.
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
            lineType=cv2.LINE_AA,
        )

    # 11) Binary overlay LAST so the spline + inflection points sit on top
    # of the green fill (otherwise they would disappear under the addWeighted).
    _draw_binary_overlay(vis, cobb_binary_result)

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
