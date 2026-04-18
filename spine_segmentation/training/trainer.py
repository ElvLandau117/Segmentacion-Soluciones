"""
Training loop with MLflow experiment tracking, mixed precision,
early stopping, and differential learning rates.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from spine_segmentation.config import (
    TRAIN_CONFIG,
    CHECKPOINTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from spine_segmentation.models.smp_models import get_model_params_groups
from spine_segmentation.evaluation.metrics import compute_metrics_binary, compute_metrics_multiclass


class Trainer:
    """
    Unified trainer for binary and multiclass segmentation models.
    Supports MLflow logging, mixed precision, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        task: str = "binary",
        model_name: str = "model",
        num_classes: int = 1,
        device: str = None,
        config: dict = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.task = task
        self.model_name = model_name
        self.num_classes = num_classes
        self.config = config or TRAIN_CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model and loss to device
        self.model = self.model.to(self.device)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)

        # Optimizer with differential LR
        param_groups = get_model_params_groups(
            self.model,
            encoder_lr=self.config["encoder_lr"],
            decoder_lr=self.config["decoder_lr"],
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config["scheduler_T0"],
            T_mult=self.config["scheduler_T_mult"],
            eta_min=self.config["scheduler_eta_min"],
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.config["use_amp"] else None
        self.use_amp = self.config["use_amp"] and self.device == "cuda"

        # Early stopping
        self.best_val_dice = 0.0
        self.patience_counter = 0
        self.best_epoch = 0

        # Checkpoint path
        self.checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_{task}_best.pth"

        print(f"\nTrainer initialized:")
        print(f"  Model: {model_name}, Task: {task}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Epochs: {self.config['num_epochs']}, Patience: {self.config['early_stopping_patience']}")

    def train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch. Returns dict of average metrics."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)

            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation. Returns dict of metrics."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_masks = []
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)

            running_loss += loss.item()
            num_batches += 1

            # Collect predictions for metrics
            if self.task == "binary":
                preds = torch.sigmoid(predictions).cpu()
            else:
                preds = predictions.cpu()
            all_preds.append(preds)
            all_masks.append(masks.cpu())

        avg_loss = running_loss / max(num_batches, 1)
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute segmentation metrics
        if self.task == "binary":
            metrics = compute_metrics_binary(all_preds, all_masks)
        else:
            metrics = compute_metrics_multiclass(all_preds, all_masks, self.num_classes)

        metrics["val_loss"] = avg_loss
        return metrics

    def train(self) -> dict:
        """
        Full training loop with early stopping and MLflow logging.
        Returns dict with best metrics.
        """
        num_epochs = self.config["num_epochs"]
        patience = self.config["early_stopping_patience"]

        # Setup MLflow
        if HAS_MLFLOW:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        run_name = f"{self.model_name}_{self.task}_{time.strftime('%Y%m%d_%H%M%S')}"

        mlflow_context = mlflow.start_run(run_name=run_name) if HAS_MLFLOW else _DummyContext()

        with mlflow_context:
            # Log parameters
            if HAS_MLFLOW:
                mlflow.log_params({
                    "model_name": self.model_name,
                    "task": self.task,
                    "num_classes": self.num_classes,
                    "image_size": self.config["image_size"],
                    "batch_size": self.config["batch_size"],
                    "encoder_lr": self.config["encoder_lr"],
                    "decoder_lr": self.config["decoder_lr"],
                    "weight_decay": self.config["weight_decay"],
                    "num_epochs": num_epochs,
                    "early_stopping_patience": patience,
                    "use_amp": self.use_amp,
                    "device": self.device,
                })

            print(f"\n{'='*60}")
            print(f"Starting training: {run_name}")
            print(f"{'='*60}\n")

            for epoch in range(num_epochs):
                epoch_start = time.time()

                # Train
                train_metrics = self.train_one_epoch(epoch)

                # Validate
                val_metrics = self.validate(epoch)

                # Step scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Extract key metric
                val_dice = val_metrics.get("val_dice_mean", val_metrics.get("val_dice", 0.0))

                epoch_time = time.time() - epoch_start

                # Print epoch summary
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Dice: {val_dice:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # Log to MLflow
                if HAS_MLFLOW:
                    log_metrics = {
                        "train_loss": train_metrics["train_loss"],
                        "val_loss": val_metrics["val_loss"],
                        "val_dice_mean": val_dice,
                        "learning_rate": current_lr,
                    }
                    # Add IoU and pixel accuracy if available
                    for key in ["val_iou_mean", "val_pixel_acc"]:
                        if key in val_metrics:
                            log_metrics[key] = val_metrics[key]
                    mlflow.log_metrics(log_metrics, step=epoch)

                # Early stopping check
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    # Save best checkpoint
                    self._save_checkpoint(epoch, val_metrics)
                    print(f"  >>> New best model! Dice={val_dice:.4f} saved to {self.checkpoint_path}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1} "
                              f"(no improvement for {patience} epochs)")
                        break

            # Log final best metrics
            if HAS_MLFLOW:
                mlflow.log_metrics({
                    "best_val_dice": self.best_val_dice,
                    "best_epoch": self.best_epoch,
                })
                mlflow.log_artifact(str(self.checkpoint_path))

            print(f"\n{'='*60}")
            print(f"Training complete! Best Dice: {self.best_val_dice:.4f} at epoch {self.best_epoch+1}")
            print(f"Best model saved to: {self.checkpoint_path}")
            print(f"{'='*60}")

        return {
            "best_val_dice": self.best_val_dice,
            "best_epoch": self.best_epoch,
            "checkpoint_path": str(self.checkpoint_path),
        }

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "metrics": metrics,
            "model_name": self.model_name,
            "task": self.task,
            "num_classes": self.num_classes,
            "config": self.config,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_best_model(self):
        """Load the best checkpoint into the model."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} "
              f"(Dice={checkpoint['best_val_dice']:.4f})")
        return self.model


class _DummyContext:
    """Dummy context manager when MLflow is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
