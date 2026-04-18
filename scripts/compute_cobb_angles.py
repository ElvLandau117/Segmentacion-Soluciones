"""
Batch Cobb angle computation and comparison against ground truth.
Runs both binary and multiclass methods on the test set.

Usage: python scripts/compute_cobb_angles.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spine_segmentation.config import CHECKPOINTS_DIR, OUTPUTS_DIR, MODEL_CONFIGS
from spine_segmentation.data.dataset import get_dataloaders
from spine_segmentation.data.splits import load_splits
from spine_segmentation.models.smp_models import create_model
from spine_segmentation.evaluation.cobb_angle import (
    cobb_from_binary,
    cobb_from_multiclass,
    evaluate_cobb_angles,
    plot_cobb_comparison,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits()

    print("=" * 60)
    print("COBB ANGLE COMPUTATION AND EVALUATION")
    print("=" * 60)

    all_results = {}

    # Process each checkpoint
    for ckpt_path in sorted(CHECKPOINTS_DIR.glob("*.pth")):
        print(f"\nProcessing: {ckpt_path.stem}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_name = checkpoint.get("model_name", "unknown")
        task = checkpoint.get("task", "binary")
        num_classes = checkpoint.get("num_classes", 1)

        try:
            model = create_model(model_name, num_classes=num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if task == "binary":
            loaders = get_dataloaders(task="binary", splits_dict=splits)
        else:
            loaders = get_dataloaders(task="multiclass", scheme="vertebrae_24", splits_dict=splits)

        # Compute Cobb angles for test set
        predictions = {}
        with torch.no_grad():
            for batch in loaders["test"]:
                images = batch["image"].to(device)
                names = batch["image_name"]

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    preds = model(images)

                for i, name in enumerate(names):
                    if not name.startswith("S_"):
                        continue

                    if task == "binary":
                        pred_mask = (torch.sigmoid(preds[i, 0]).cpu().numpy() > 0.5).astype(np.uint8)
                        result = cobb_from_binary(pred_mask)
                    else:
                        pred_mask = preds[i].argmax(dim=0).cpu().numpy().astype(np.uint8)
                        result = cobb_from_multiclass(pred_mask)

                    if result["success"] and result["cobb_angle_deg"] is not None:
                        predictions[name] = result["cobb_angle_deg"]

        # Evaluate against ground truth
        if predictions:
            eval_results = evaluate_cobb_angles(predictions)
            key = f"{model_name}_{task}"
            all_results[key] = eval_results

            print(f"  Samples evaluated: {eval_results['n_samples']}")
            print(f"  MAE: {eval_results['mae_deg']:.1f} degrees")
            print(f"  Correlation: {eval_results.get('correlation', 0):.3f}")
            print(f"  Within 5 deg: {eval_results.get('within_5_deg', 0)*100:.1f}%")
            print(f"  Within 10 deg: {eval_results.get('within_10_deg', 0)*100:.1f}%")

            plot_cobb_comparison(
                eval_results,
                save_path=str(OUTPUTS_DIR / f"cobb_{key}.png"),
                title=f"Cobb Angle - {model_name} ({task})",
            )

    # Summary comparison
    if all_results:
        print("\n" + "=" * 60)
        print("COBB ANGLE COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':>30} {'MAE (deg)':>10} {'Corr':>8} {'<5 deg':>8} {'<10 deg':>8}")
        print("-" * 70)
        for key, res in all_results.items():
            print(f"{key:>30} {res['mae_deg']:>10.1f} {res.get('correlation',0):>8.3f} "
                  f"{res.get('within_5_deg',0)*100:>7.1f}% {res.get('within_10_deg',0)*100:>7.1f}%")

        # Save summary
        summary_path = OUTPUTS_DIR / "cobb_angle_summary.json"
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_image"}
                   for k, v in all_results.items()}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
