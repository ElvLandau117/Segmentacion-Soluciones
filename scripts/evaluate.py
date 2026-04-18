"""
Comprehensive evaluation script: runs all models on the test set,
computes metrics, generates visualizations, and compares results.

Usage: python scripts/evaluate.py
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spine_segmentation.config import (
    MODEL_CONFIGS, CHECKPOINTS_DIR, OUTPUTS_DIR, TRAIN_CONFIG,
)
from spine_segmentation.data.dataset import get_dataloaders
from spine_segmentation.data.splits import load_splits
from spine_segmentation.data.class_mapping import get_class_names
from spine_segmentation.models.smp_models import create_model
from spine_segmentation.evaluation.metrics import compute_test_metrics, print_metrics_table
from spine_segmentation.evaluation.visualize import (
    plot_per_class_dice,
    save_prediction_grid,
)
from spine_segmentation.evaluation.cobb_angle import (
    cobb_from_binary,
    cobb_from_multiclass,
    evaluate_cobb_angles,
    plot_cobb_comparison,
)
from spine_segmentation.data.transforms import denormalize_image


def evaluate_all_models():
    """Evaluate all trained models and generate comparison table."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits()

    results_table = []

    # Find all checkpoints
    checkpoints = list(CHECKPOINTS_DIR.glob("*.pth"))
    if not checkpoints:
        print("No checkpoints found in", CHECKPOINTS_DIR)
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print("=" * 80)

    for ckpt_path in sorted(checkpoints):
        print(f"\nEvaluating: {ckpt_path.stem}")
        print("-" * 60)

        # Load checkpoint metadata
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_name = checkpoint.get("model_name", "unknown")
        task = checkpoint.get("task", "binary")
        num_classes = checkpoint.get("num_classes", 1)

        # Create model and load weights
        try:
            model = create_model(model_name, num_classes=num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue

        # Create test dataloader
        if task == "binary":
            loaders = get_dataloaders(task="binary", splits_dict=splits)
        else:
            loaders = get_dataloaders(task="multiclass", scheme="vertebrae_24", splits_dict=splits)

        # Compute metrics
        test_metrics = compute_test_metrics(
            model, loaders["test"],
            task=task, num_classes=num_classes,
            device=device,
        )

        # Print results
        class_names = get_class_names("vertebrae_24") if task == "multiclass" else None
        print_metrics_table(test_metrics, class_names=class_names)

        # Save per-class dice chart for multiclass
        if task == "multiclass" and "per_class_dice" in test_metrics:
            plot_per_class_dice(
                test_metrics["per_class_dice"],
                class_names=class_names,
                save_path=str(OUTPUTS_DIR / f"per_class_dice_{ckpt_path.stem}.png"),
                title=f"Per-Class Dice - {model_name}",
            )

        # Compute Cobb angles on test set
        cobb_predictions = {}
        with torch.no_grad():
            for batch in loaders["test"]:
                images = batch["image"].to(device)
                masks_gt = batch["mask"]
                names = batch["image_name"]

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    preds = model(images)

                for i, name in enumerate(names):
                    if not name.startswith("S_"):
                        continue  # Skip normal cases

                    if task == "binary":
                        pred_mask = (torch.sigmoid(preds[i, 0]).cpu().numpy() > 0.5).astype(np.uint8)
                        cobb_result = cobb_from_binary(pred_mask)
                    else:
                        pred_mask = preds[i].argmax(dim=0).cpu().numpy().astype(np.uint8)
                        cobb_result = cobb_from_multiclass(pred_mask)

                    if cobb_result["success"]:
                        cobb_predictions[name] = cobb_result["cobb_angle_deg"]

        # Evaluate Cobb angles
        cobb_metrics = {}
        if cobb_predictions:
            cobb_metrics = evaluate_cobb_angles(cobb_predictions)
            if "mae_deg" in cobb_metrics:
                print(f"\n  Cobb Angle MAE: {cobb_metrics['mae_deg']:.1f} degrees")
                print(f"  Correlation: {cobb_metrics.get('correlation', 0):.3f}")
                print(f"  Within 5 deg: {cobb_metrics.get('within_5_deg', 0)*100:.1f}%")

                plot_cobb_comparison(
                    cobb_metrics,
                    save_path=str(OUTPUTS_DIR / f"cobb_comparison_{ckpt_path.stem}.png"),
                    title=f"Cobb Angle - {model_name} ({task})",
                )

        # Add to results table
        row = {
            "Model": model_name,
            "Task": task,
            "Dice Mean": test_metrics.get("test_dice_mean", 0),
            "IoU Mean": test_metrics.get("test_iou_mean", 0),
            "Pixel Acc": test_metrics.get("test_pixel_acc", 0),
            "Cobb MAE (deg)": cobb_metrics.get("mae_deg", None),
            "Cobb Corr": cobb_metrics.get("correlation", None),
        }
        results_table.append(row)

    # Print final comparison table
    if results_table:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON TABLE")
        print("=" * 80)
        df = pd.DataFrame(results_table)
        print(df.to_string(index=False, float_format="%.4f"))

        # Save to CSV
        csv_path = OUTPUTS_DIR / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved to {csv_path}")


if __name__ == "__main__":
    evaluate_all_models()
