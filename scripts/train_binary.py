"""
Train binary spine segmentation models.
Usage: python scripts/train_binary.py --model unet_resnet50
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spine_segmentation.config import MODEL_CONFIGS, TRAIN_CONFIG
from spine_segmentation.data.dataset import get_dataloaders
from spine_segmentation.data.splits import load_splits
from spine_segmentation.models.smp_models import create_model
from spine_segmentation.models.losses import get_loss_function
from spine_segmentation.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train binary spine segmentation")
    parser.add_argument(
        "--model", type=str, default="unet_resnet50",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model architecture to train"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size")
    parser.add_argument("--lr", type=float, default=None, help="Override decoder learning rate")
    args = parser.parse_args()

    # Override config if specified
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

    print("=" * 60)
    print("BINARY SPINE SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Create data splits (or load existing)
    splits = load_splits()

    # Create dataloaders
    loaders = get_dataloaders(
        task="binary",
        batch_size=config["batch_size"],
        splits_dict=splits,
    )

    # Create model
    model = create_model(args.model, num_classes=1)

    # Create loss function
    loss_fn = get_loss_function(task="binary")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=loss_fn,
        task="binary",
        model_name=args.model,
        num_classes=1,
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
        task="binary", num_classes=1,
        device=trainer.device,
    )
    print_metrics_table(test_metrics)

    print("\nDone!")


if __name__ == "__main__":
    main()
