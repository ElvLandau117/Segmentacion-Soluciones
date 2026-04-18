"""
Inference pipeline for spine segmentation.
Loads trained models and runs end-to-end prediction with post-processing.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from spine_segmentation.config import (
    CHECKPOINTS_DIR,
    TRAIN_CONFIG,
    MULTICLASS_COLORS,
)
from spine_segmentation.models.smp_models import create_model
from spine_segmentation.data.transforms import resize_with_padding, get_inference_transforms
from spine_segmentation.data.class_mapping import get_class_names, get_num_classes
from spine_segmentation.postprocessing.morphology import clean_binary_mask, clean_multiclass_mask
from spine_segmentation.evaluation.cobb_angle import cobb_from_binary, cobb_from_multiclass


class SpineSegmentationPipeline:
    """
    End-to-end inference pipeline for spine segmentation and Cobb angle calculation.
    """

    def __init__(
        self,
        binary_checkpoint: str = None,
        multiclass_checkpoint: str = None,
        binary_model_name: str = "unet_resnet50",
        multiclass_model_name: str = "unet_resnet50",
        scheme: str = "vertebrae_24",
        device: str = None,
        image_size: int = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size or TRAIN_CONFIG["image_size"]
        self.scheme = scheme
        self.num_classes = get_num_classes(scheme)
        self.class_names = get_class_names(scheme)
        self.transforms = get_inference_transforms(self.image_size)

        # Load binary model
        self.binary_model = None
        if binary_checkpoint:
            self.binary_model = self._load_model(
                binary_checkpoint, binary_model_name, num_classes=1
            )

        # Load multiclass model
        self.multiclass_model = None
        if multiclass_checkpoint:
            self.multiclass_model = self._load_model(
                multiclass_checkpoint, multiclass_model_name, num_classes=self.num_classes
            )

        print(f"Pipeline initialized on {self.device}")
        print(f"  Binary model: {'loaded' if self.binary_model else 'not loaded'}")
        print(f"  Multiclass model: {'loaded' if self.multiclass_model else 'not loaded'}")

    def _load_model(self, checkpoint_path: str, model_name: str, num_classes: int):
        """Load a model from checkpoint."""
        model = create_model(model_name, num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image: np.ndarray) -> dict:
        """
        Run full prediction pipeline on a single image.

        Args:
            image: (H, W, 3) RGB image (uint8)

        Returns:
            dict with all results (masks, Cobb angles, visualizations)
        """
        results = {
            "original_image": image,
            "binary_mask": None,
            "multiclass_mask": None,
            "binary_overlay": None,
            "multiclass_overlay": None,
            "cobb_binary": None,
            "cobb_multiclass": None,
            "vertebrae_detected": [],
            "cobb_visualization": None,
        }

        # Preprocess
        image_resized, _, padding_info = resize_with_padding(
            image, np.zeros(image.shape[:2], dtype=np.uint8), self.image_size
        )
        augmented = self.transforms(image=image_resized)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Binary prediction
        if self.binary_model is not None:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    binary_logits = self.binary_model(input_tensor)

            binary_prob = torch.sigmoid(binary_logits).cpu().numpy()[0, 0]
            binary_mask = (binary_prob > 0.5).astype(np.uint8)
            binary_mask = clean_binary_mask(binary_mask)
            results["binary_mask"] = binary_mask
            results["binary_overlay"] = self._create_overlay(image_resized, binary_mask, "binary")

            # Cobb angle from binary
            cobb_result = cobb_from_binary(binary_mask)
            results["cobb_binary"] = cobb_result

        # Multiclass prediction
        if self.multiclass_model is not None:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    multi_logits = self.multiclass_model(input_tensor)

            multi_probs = torch.softmax(multi_logits, dim=1).cpu().numpy()[0]
            multi_mask = multi_probs.argmax(axis=0).astype(np.uint8)
            multi_mask = clean_multiclass_mask(multi_mask, self.num_classes)
            results["multiclass_mask"] = multi_mask
            results["multiclass_overlay"] = self._create_overlay(
                image_resized, multi_mask, "multiclass"
            )

            # Cobb angle from multiclass
            cobb_result = cobb_from_multiclass(multi_mask, self.scheme)
            results["cobb_multiclass"] = cobb_result
            results["vertebrae_detected"] = cobb_result.get("vertebrae_detected", [])

        # Create Cobb angle visualization
        results["cobb_visualization"] = self._create_cobb_visualization(results, image_resized)

        return results

    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        task: str,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create a colored overlay visualization."""
        overlay = image.copy()

        if task == "binary":
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = [0, 255, 0]  # Green for spine
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
        else:
            color_mask = np.zeros_like(image)
            for class_id, color in MULTICLASS_COLORS.items():
                if class_id == 0:
                    continue
                region = mask == class_id
                if region.any():
                    color_mask[region] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

        return overlay

    def _create_cobb_visualization(
        self,
        results: dict,
        image: np.ndarray,
    ) -> np.ndarray:
        """Create a visualization showing the Cobb angle measurement."""
        vis = image.copy()

        # Draw Cobb angle info from binary method
        cobb_binary = results.get("cobb_binary")
        if cobb_binary and cobb_binary.get("success"):
            angle = cobb_binary["cobb_angle_deg"]
            # Draw inflection points and lines
            if "inflection_points" in cobb_binary:
                for pt in cobb_binary["inflection_points"]:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)

            cv2.putText(vis, f"Binary Cobb: {angle:.1f} deg",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw Cobb angle info from multiclass method
        cobb_multi = results.get("cobb_multiclass")
        if cobb_multi and cobb_multi.get("success"):
            angle = cobb_multi["cobb_angle_deg"]
            upper = cobb_multi.get("upper_end_vertebra", "?")
            lower = cobb_multi.get("lower_end_vertebra", "?")
            cv2.putText(vis, f"Multiclass Cobb: {angle:.1f} deg ({upper}-{lower})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # List detected vertebrae
        vertebrae = results.get("vertebrae_detected", [])
        if vertebrae:
            text = f"Vertebrae: {', '.join(vertebrae[:10])}"
            if len(vertebrae) > 10:
                text += f"... (+{len(vertebrae)-10})"
            cv2.putText(vis, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis
