"""
Cobb Angle Calculation Pipeline.
Two methods:
  Method A: From binary segmentation (skeleton + B-spline + inflection points)
  Method B: From multiclass segmentation (vertebra endplate orientations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from pathlib import Path
from typing import Tuple, Optional

from spine_segmentation.config import RADIOGRAPH_METRICS_CSV, OUTPUTS_DIR
from spine_segmentation.postprocessing.morphology import (
    clean_binary_mask,
    extract_spine_skeleton,
    get_skeleton_points,
)
from spine_segmentation.postprocessing.vertebra_ordering import (
    extract_vertebra_info,
    identify_end_vertebrae,
    compute_endplate_angles,
)
from spine_segmentation.data.class_mapping import get_class_names


# ============================================================================
# Method A: Cobb Angle from Binary Segmentation
# (Replicates and improves the previous semester's approach)
# ============================================================================

def _cobb_between_inflection_points(dx_dy: np.ndarray, ip_a: int, ip_b: int) -> float:
    """Cobb angle (deg) between the tangent slopes at two inflection-point indices.

    Uses the standard formula: angle = arctan((m1 - m2) / (1 + m1 * m2)).
    Returns 90 deg when the lines are perpendicular (within tolerance).
    """
    m1 = float(dx_dy[ip_a])
    m2 = float(dx_dy[ip_b])
    denom = 1.0 + m1 * m2
    if abs(denom) < 1e-10:
        return 90.0
    return float(abs(np.degrees(np.arctan((m1 - m2) / denom))))


def _curve_direction(dx_dy: np.ndarray, ip_a: int, ip_b: int) -> str:
    """Convexity direction of a curve between two inflection points.

    Sign of the slope at the midpoint indexes the side the curve bulges to.
    Negative slope (x decreases with y in our `x = f(y)` parameterization, which
    in image coords means the spine moves LEFT as we go down) -> convexity
    "right" (the curve apex bulges to the right). Positive -> "left".
    Returns "unknown" if the indices are bad.
    """
    if ip_b <= ip_a or ip_b >= len(dx_dy):
        return "unknown"
    mid_idx = (ip_a + ip_b) // 2
    mid_slope = float(dx_dy[mid_idx])
    if abs(mid_slope) < 1e-3:
        return "neutral"
    return "right" if mid_slope < 0 else "left"


def cobb_from_binary(
    binary_mask: np.ndarray,
    smoothing_factor: float = 1500.0,
    min_curve_deg: float = 2.0,
) -> dict:
    """Calculate the Cobb angle from a binary spine segmentation mask, with
    support for multiple curves (S-shape, triple).

    Pipeline:
    1. Clean mask
    2. Skeletonize
    3. Fit B-spline to skeleton points
    4. Compute first and second derivatives
    5. Find ALL inflection points (second derivative zero crossings)
    6. For EACH adjacent pair of inflection points, compute a Cobb angle.
       Filter out curves below `min_curve_deg` (noise), sort by magnitude.

    Args:
        binary_mask: (H, W) binary spine mask {0, 1}
        smoothing_factor: B-spline smoothing parameter. Lowered from 5000 to
            1500 in Ciclo 5.3 (fix C) so subtle inflection points in mild /
            compensatory curves are not smoothed away. The internal fallback
            still divides this by 5 if <2 IPs are found, so a default of 1500
            yields a fallback of 300 — aggressive but only triggered on very
            flat splines.
        min_curve_deg: ignore curves below this angle. Lowered from 3.0 to 2.0
            in Ciclo 5.3 (fix D) to surface mild compensatory curves that
            still carry clinical signal. Curves under 2 deg are likely spline
            noise and stay filtered.

    Returns:
        dict with:
          - cobb_angle_deg (float): back-compat; the angle of the principal curve.
          - curves (list[dict]): one entry per detected curve, sorted desc by angle.
            Each entry: cobb_angle_deg, ip_upper (x, y), ip_lower (x, y),
            slope_upper, slope_lower, direction ("right"|"left"|"neutral"|"unknown"), rank.
          - all_inflection_points (list[(x, y)]): every IP, useful for the binary overlay.
          - inflection_points (list[(x, y)]): the 2 extremes of the principal curve (back-compat).
          - spline_x, spline_y: the fitted curve (list of floats).
    """
    result = {
        "cobb_angle_deg": None,
        "method": "binary_skeleton",
        "success": False,
        "error": None,
        "curves": [],
        "all_inflection_points": [],
    }

    try:
        # Step 1: Clean mask
        cleaned = clean_binary_mask(binary_mask)
        if cleaned.sum() == 0:
            result["error"] = "Empty mask after cleaning"
            return result

        # Step 2: Skeletonize
        skeleton = extract_spine_skeleton(cleaned)
        points = get_skeleton_points(skeleton)

        if len(points) < 10:
            result["error"] = f"Too few skeleton points: {len(points)}"
            return result

        # Step 3: Fit B-spline (x as function of y, since spine is vertical)
        x_coords = points[:, 0].astype(float)
        y_coords = points[:, 1].astype(float)

        # Remove duplicate y values (average x at same y)
        unique_y, indices = np.unique(y_coords, return_inverse=True)
        unique_x = np.array([x_coords[indices == i].mean() for i in range(len(unique_y))])

        if len(unique_y) < 5:
            result["error"] = "Too few unique y coordinates"
            return result

        # Fit spline: x = f(y)
        spline = UnivariateSpline(unique_y, unique_x, s=smoothing_factor, k=3)

        # Step 4: Evaluate spline and derivatives
        y_eval = np.linspace(unique_y.min(), unique_y.max(), 500)
        x_eval = spline(y_eval)
        dx_dy = spline.derivative(n=1)(y_eval)    # First derivative
        d2x_dy2 = spline.derivative(n=2)(y_eval)  # Second derivative

        # Step 5: Find ALL inflection points (zero crossings of second derivative)
        sign_changes = np.where(np.diff(np.sign(d2x_dy2)))[0]

        if len(sign_changes) < 2:
            # Try with less smoothing — picks up subtler curves
            spline2 = UnivariateSpline(unique_y, unique_x, s=smoothing_factor / 5, k=3)
            x_eval = spline2(y_eval)
            dx_dy = spline2.derivative(n=1)(y_eval)
            d2x_dy2 = spline2.derivative(n=2)(y_eval)
            sign_changes = np.where(np.diff(np.sign(d2x_dy2)))[0]

        result["spline_x"] = x_eval.tolist()
        result["spline_y"] = y_eval.tolist()
        result["all_inflection_points"] = [
            (float(x_eval[i]), float(y_eval[i])) for i in sign_changes
        ]

        if len(sign_changes) < 2:
            # Truly straight spine: 0 or 1 inflection points -> no curve at all.
            result["error"] = "Could not find enough inflection points"
            result["cobb_angle_deg"] = 0.0
            result["success"] = True
            return result

        # Step 6: For EACH adjacent pair of inflection points, compute a Cobb.
        # This is the key change vs the previous version, which only used the
        # 2 outermost IPs and missed compensatory / S-shape curves.
        candidates = []
        for i in range(len(sign_changes) - 1):
            ip_a = int(sign_changes[i])
            ip_b = int(sign_changes[i + 1])
            angle = _cobb_between_inflection_points(dx_dy, ip_a, ip_b)
            if angle < min_curve_deg:
                continue
            candidates.append({
                "cobb_angle_deg": angle,
                "ip_upper": (float(x_eval[ip_a]), float(y_eval[ip_a])),
                "ip_lower": (float(x_eval[ip_b]), float(y_eval[ip_b])),
                "slope_upper": float(dx_dy[ip_a]),
                "slope_lower": float(dx_dy[ip_b]),
                "direction": _curve_direction(dx_dy, ip_a, ip_b),
            })

        if not candidates:
            # Many small inflections but none above the noise floor.
            result["cobb_angle_deg"] = 0.0
            result["success"] = True
            # Still set inflection_points (back-compat) to the 2 outermost.
            result["inflection_points"] = [
                result["all_inflection_points"][0],
                result["all_inflection_points"][-1],
            ]
            return result

        # Sort by magnitude descending and assign ranks.
        candidates.sort(key=lambda c: c["cobb_angle_deg"], reverse=True)
        for rank, c in enumerate(candidates, start=1):
            c["rank"] = rank

        principal = candidates[0]
        result["curves"] = candidates
        result["cobb_angle_deg"] = principal["cobb_angle_deg"]
        result["success"] = True
        # Back-compat: inflection_points = the 2 IPs of the principal curve.
        result["inflection_points"] = [principal["ip_upper"], principal["ip_lower"]]

    except Exception as e:
        result["error"] = str(e)

    return result


def assign_vertebra_names_to_curves(
    curves: list,
    multiclass_vertebrae: list,
) -> list:
    """Attach `upper_vertebra` and `lower_vertebra` names to each curve dict.

    For each curve, find the multiclass-detected vertebra whose centroid_y is
    closest to the curve's ip_upper/ip_lower y-coordinates. The multiclass model
    is used here only for NAMING (label transfer) — it is NOT used to compute
    the Cobb angle itself, because per-vertebra masks are too noisy on our data.

    Args:
        curves: list of dicts from `cobb_from_binary`'s "curves" key.
        multiclass_vertebrae: list of dicts from
            `postprocessing.vertebra_ordering.extract_vertebra_info`, each
            carrying at least `name` and `centroid_y`.

    Returns:
        The same list with `upper_vertebra`, `lower_vertebra` keys added.
        When no multiclass vertebrae are available, both names are None.
    """
    for curve in curves:
        if not multiclass_vertebrae:
            curve["upper_vertebra"] = None
            curve["lower_vertebra"] = None
            continue
        ip_up_y = curve["ip_upper"][1]
        ip_low_y = curve["ip_lower"][1]
        upper = min(multiclass_vertebrae, key=lambda v: abs(v["centroid_y"] - ip_up_y))
        lower = min(multiclass_vertebrae, key=lambda v: abs(v["centroid_y"] - ip_low_y))
        curve["upper_vertebra"] = upper["name"]
        curve["lower_vertebra"] = lower["name"]
    return curves


# ============================================================================
# Method B: Cobb Angle from Multiclass Segmentation
# (Main contribution - uses vertebra endplate orientations)
# ============================================================================

def cobb_from_multiclass(
    multiclass_mask: np.ndarray,
    scheme: str = "vertebrae_24",
) -> dict:
    """
    Calculate the Cobb angle from multiclass vertebrae segmentation.

    This is the clinically standard method:
    1. Identify individual vertebrae from the segmentation mask
    2. Compute endplate orientation for each vertebra (via PCA/moments)
    3. Find the upper and lower end vertebrae of the curve
    4. Cobb angle = angle between the perpendicular lines to the endplates

    Args:
        multiclass_mask: (H, W) integer class labels
        scheme: Class mapping scheme used

    Returns:
        dict with cobb_angle_deg, end_vertebrae, all_vertebrae info
    """
    result = {
        "cobb_angle_deg": None,
        "method": "multiclass_endplate",
        "success": False,
        "error": None,
        "upper_end_vertebra": None,
        "lower_end_vertebra": None,
        "vertebrae_detected": [],
    }

    try:
        class_names = get_class_names(scheme)

        # Step 1: Extract vertebra information
        vertebrae = extract_vertebra_info(multiclass_mask, class_names)

        # Filter only true vertebrae (not "other_structures" or "background")
        vertebrae = [v for v in vertebrae if v["name"] in [
            "C3", "C4", "C5", "C6", "C7",
            "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
            "L1", "L2", "L3", "L4", "L5",
        ]]

        result["vertebrae_detected"] = [v["name"] for v in vertebrae]

        if len(vertebrae) < 3:
            result["error"] = f"Too few vertebrae detected: {len(vertebrae)}"
            return result

        # Step 2: Compute endplate angles
        vertebrae = compute_endplate_angles(vertebrae)

        # Step 3: Identify end vertebrae
        upper_end, lower_end = identify_end_vertebrae(vertebrae)

        if upper_end is None or lower_end is None:
            result["error"] = "Could not identify end vertebrae"
            return result

        result["upper_end_vertebra"] = upper_end["name"]
        result["lower_end_vertebra"] = lower_end["name"]

        # Step 4: Compute Cobb angle
        # The Cobb angle is the angle between the superior endplate of the upper
        # end vertebra and the inferior endplate of the lower end vertebra
        angle_upper = upper_end["endplate_angle_deg"]
        angle_lower = lower_end["endplate_angle_deg"]

        # Cobb angle = difference in endplate tilts
        cobb_angle = abs(angle_upper - angle_lower)

        # Clamp to valid range
        cobb_angle = min(cobb_angle, 90.0)

        result["cobb_angle_deg"] = float(cobb_angle)
        result["success"] = True
        result["all_vertebrae"] = [
            {
                "name": v["name"],
                "centroid": (v["centroid_x"], v["centroid_y"]),
                "angle_deg": v["endplate_angle_deg"],
                "area": v["area"],
            }
            for v in vertebrae
        ]

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# Ground Truth Comparison
# ============================================================================

def load_ground_truth_cobb_angles() -> pd.DataFrame:
    """
    Load ground truth Cobb angles from the dataset metrics.

    Returns:
        DataFrame with columns: patient_id, cobb_angle_deg
    """
    df = pd.read_csv(RADIOGRAPH_METRICS_CSV)
    return df[["patient_id", "cobb_angle_deg"]]


def evaluate_cobb_angles(
    predictions: dict,
    ground_truth: pd.DataFrame = None,
) -> dict:
    """
    Compare predicted Cobb angles against ground truth.

    Args:
        predictions: dict mapping image_name -> predicted_cobb_angle
        ground_truth: DataFrame with patient_id and cobb_angle_deg

    Returns:
        dict with MAE, correlation, per-image errors, etc.
    """
    if ground_truth is None:
        ground_truth = load_ground_truth_cobb_angles()

    pred_angles = []
    gt_angles = []
    image_names = []

    for img_name, pred_angle in predictions.items():
        if pred_angle is None:
            continue

        # Extract patient ID from image name (e.g., "S_21.jpg" -> 21)
        if img_name.startswith("S_"):
            patient_id = int(img_name.split("_")[1].split(".")[0])
        else:
            continue  # Normal cases don't have Cobb angles

        gt_row = ground_truth[ground_truth["patient_id"] == patient_id]
        if gt_row.empty:
            continue

        gt_angle = gt_row["cobb_angle_deg"].values[0]
        pred_angles.append(pred_angle)
        gt_angles.append(gt_angle)
        image_names.append(img_name)

    if not pred_angles:
        return {"error": "No matching predictions and ground truth"}

    pred_angles = np.array(pred_angles)
    gt_angles = np.array(gt_angles)
    errors = np.abs(pred_angles - gt_angles)

    results = {
        "mae_deg": float(np.mean(errors)),
        "std_deg": float(np.std(errors)),
        "median_error_deg": float(np.median(errors)),
        "max_error_deg": float(np.max(errors)),
        "correlation": float(np.corrcoef(pred_angles, gt_angles)[0, 1])
            if len(pred_angles) > 1 else 0.0,
        "within_5_deg": float((errors <= 5.0).mean()),
        "within_10_deg": float((errors <= 10.0).mean()),
        "n_samples": len(pred_angles),
        "per_image": {
            name: {"pred": float(p), "gt": float(g), "error": float(e)}
            for name, p, g, e in zip(image_names, pred_angles, gt_angles, errors)
        },
    }

    return results


def plot_cobb_comparison(
    results: dict,
    save_path: str = None,
    title: str = "Cobb Angle: Predicted vs Ground Truth",
):
    """
    Create scatter plot and Bland-Altman plot for Cobb angle comparison.
    """
    per_image = results.get("per_image", {})
    if not per_image:
        print("No data for Cobb angle comparison plot")
        return

    pred_angles = [v["pred"] for v in per_image.values()]
    gt_angles = [v["gt"] for v in per_image.values()]

    pred_arr = np.array(pred_angles)
    gt_arr = np.array(gt_angles)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1.scatter(gt_arr, pred_arr, alpha=0.6, edgecolors='black', linewidth=0.5)
    max_val = max(gt_arr.max(), pred_arr.max()) + 5
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect agreement')
    ax1.set_xlabel('Ground Truth Cobb Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Predicted Cobb Angle (degrees)', fontsize=12)
    ax1.set_title('Predicted vs Ground Truth', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.text(0.05, 0.95, f'MAE: {results["mae_deg"]:.1f}°\nr: {results["correlation"]:.3f}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Bland-Altman plot
    mean_vals = (pred_arr + gt_arr) / 2
    diff_vals = pred_arr - gt_arr
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    ax2.scatter(mean_vals, diff_vals, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2,
                label=f'Mean: {mean_diff:.1f}°')
    ax2.axhline(y=mean_diff + 1.96 * std_diff, color='gray', linestyle='--',
                label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.1f}°')
    ax2.axhline(y=mean_diff - 1.96 * std_diff, color='gray', linestyle='--',
                label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.1f}°')
    ax2.set_xlabel('Mean of Predicted and GT (degrees)', fontsize=12)
    ax2.set_ylabel('Difference (Predicted - GT) (degrees)', fontsize=12)
    ax2.set_title('Bland-Altman Plot', fontsize=14)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "cobb_angle_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Cobb angle comparison saved to {save_path}")
    plt.close()
