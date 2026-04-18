"""
Train multiclass vertebrae segmentation models.
Usage: python scripts/train_multiclass.py --model unet_resnet50 --scheme vertebrae_24
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spine_segmentation.config import MODEL_CONFIGS, TRAIN_CONFIG
from spine_segmentation.data.dataset import get_dataloaders, SpineMulticlassDataset
from spine_segmentation.data.splits import load_splits
from spine_segmentation.data.class_mapping import get_num_classes, get_class_names
from spine_segmentation.models.smp_models import create_model
from spine_segmentation.models.losses import get_loss_function, compute_class_weights
from spine_segmentation.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train multiclass vertebrae segmentation")
    parser.add_argument(
        "--model", type=str, default="unet_resnet50",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model architecture to train"
    )
    parser.add_argument(
        "--scheme", type=str, default="vertebrae_24",
        choices=["vertebrae_24", "full_36", "regional_5"],
        help="Class mapping scheme"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--skip-weights", action="store_true",
                       help="Skip class weight computation (use uniform weights)")
    args = parser.parse_args()

    config = TRAIN_CONFIG.copy()
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.image_size:
        config["image_size"] = args.image_size
    if args.lr:
        config["decoder_lr"] = args.lr
        config["encoder_lr"] = args.lr / 10

    num_classes = get_num_classes(args.scheme)
    class_names = get_class_names(args.scheme)

    print("=" * 60)
    print("MULTICLASS VERTEBRAE SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Scheme: {args.scheme} ({num_classes} classes)")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load splits
    splits = load_splits()

    # Create dataloaders
    loaders = get_dataloaders(
        task="multiclass",
        scheme=args.scheme,
        batch_size=config["batch_size"],
        splits_dict=splits,
    )

    # Compute class weights from training data
    class_weights = None
    if not args.skip_weights:
        train_dataset = SpineMulticlassDataset(
            split="train", scheme=args.scheme, splits_dict=splits
        )
        class_weights = compute_class_weights(train_dataset, num_classes, device=device)

    # Create model
    model = create_model(args.model, num_classes=num_classes)

    # Create loss function
    loss_fn = get_loss_function(task="multiclass", class_weights=class_weights)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=loss_fn,
        task="multiclass",
        model_name=args.model,
        num_classes=num_classes,
        config=config,
    )

    results = trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    trainer.load_best_model()

    from spine_segmentation.evaluation.metrics import compute_test_metrics, print_metrics_table
    test_metrics = compute_test_metrics(
        trainer.model, loaders["test"],
        task="multiclass", num_classes=num_classes,
        device=trainer.device,
    )
    print_metrics_table(test_metrics, class_names=class_names)

    print("\nDone!")


if __name__ == "__main__":
    main()
