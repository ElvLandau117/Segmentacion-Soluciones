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
from spine_segmentation.postprocessing.morphology import (
    clean_binary_mask,
    clean_multiclass_mask,
    extract_spine_skeleton,
    get_skeleton_points,
)
from spine_segmentation.evaluation.cobb_angle import (
    assign_vertebra_names_to_curves,
    cobb_from_binary,
    cobb_from_multiclass,
)
from spine_segmentation.evaluation.coverage import compute_coverage_info
from spine_segmentation.evaluation.orientation import compute_orientation_info
from spine_segmentation.postprocessing.vertebra_ordering import (
    compute_endplate_angles,
    extract_vertebra_info,
)


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
            "coverage_info": None,
            "orientation_info": None,
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
            # Ciclo 5.3 fix A: lower threshold 0.5 -> 0.3 to accept marginal
            # spine pixels (esp. lumbar zone). Fragmented detections get
            # bridged by the vertical-kernel closing inside clean_binary_mask
            # (fix B) and then filtered to the largest connected component,
            # so the only blobs that survive are still spine-shaped.
            binary_mask = (binary_prob > 0.3).astype(np.uint8)
            binary_mask = clean_binary_mask(binary_mask)
            results["binary_mask"] = binary_mask
            results["binary_overlay"] = self._create_overlay(image_resized, binary_mask, "binary")

            # Cobb angle from binary
            cobb_result = cobb_from_binary(binary_mask)
            results["cobb_binary"] = cobb_result

            # Ciclo 5.4 fix G: detect spine rotation. Use the same skeleton
            # extraction the Cobb pipeline uses, so the orientation reading
            # matches the data Cobb actually saw. SVD on the points gives
            # the principal axis vs vertical; the UI warns when |tilt| > 12 deg.
            skeleton = extract_spine_skeleton(binary_mask)
            skeleton_points = get_skeleton_points(skeleton)
            results["orientation_info"] = compute_orientation_info(skeleton_points)

        # Multiclass prediction.
        # `vertebrae` is also consumed by the coverage analysis (Ciclo 5.3),
        # so we keep it accessible outside the conditional. When there is no
        # multiclass model it stays as an empty list and coverage degrades
        # gracefully (no upper/lower vertebra names, just a ratio).
        vertebrae: list = []
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

            # Extract vertebra info once — needed for both label transfer
            # (Ciclo 5.2) and coverage analysis (Ciclo 5.3 fix F).
            vertebrae = extract_vertebra_info(multi_mask, self.class_names)
            vertebrae = compute_endplate_angles(vertebrae)

            # Cycle 5.2: enrich every binary-detected curve with vertebra names
            # using the multiclass detection (label transfer). The multiclass
            # mask is noisy for Cobb computation but useful for NAMING.
            #
            # Cycle 5.4 fix H: `assign_vertebra_names_to_curves` also drops
            # curves with upper == lower (degenerate). After it returns, the
            # top-level `cobb_angle_deg` and `inflection_points` may be out
            # of sync with the now-filtered curves list. Re-sync from the
            # principal of the filtered list, or zero out if the entire list
            # was filtered (rare: every curve degenerate).
            if (
                results.get("cobb_binary")
                and results["cobb_binary"].get("curves")
            ):
                assign_vertebra_names_to_curves(
                    results["cobb_binary"]["curves"], vertebrae
                )
                curves_after = results["cobb_binary"].get("curves") or []
                if curves_after:
                    principal = curves_after[0]
                    results["cobb_binary"]["cobb_angle_deg"] = (
                        principal["cobb_angle_deg"]
                    )
                    results["cobb_binary"]["inflection_points"] = [
                        principal["ip_upper"], principal["ip_lower"],
                    ]
                else:
                    # All curves were degenerate (upper == lower). Treat as
                    # a straight spine for downstream consumers.
                    results["cobb_binary"]["cobb_angle_deg"] = 0.0
                    results["cobb_binary"]["inflection_points"] = []

        # Ciclo 5.3 fix F: compute how much of the spine the binary mask
        # actually covered. The UI uses this to warn the user when "0 deg"
        # might really mean "we missed half the spine".
        binary_mask = results.get("binary_mask")
        if binary_mask is not None:
            results["coverage_info"] = compute_coverage_info(
                binary_mask=binary_mask,
                multiclass_vertebrae=vertebrae,
                image_height=self.image_size,
            )

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
        """Create a clinical-style Cobb angle visualization.

        Delegates to `draw_cobb_angle_visualization`, which draws the Shi et al.
        2025-style figure (green end-vertebra boxes + red tangent lines + numeric
        header). Falls back to a plain text overlay when the multiclass mask is
        unavailable.
        """
        from spine_segmentation.evaluation.visualize import draw_cobb_angle_visualization

        cobb_binary = results.get("cobb_binary")
        cobb_multi = results.get("cobb_multiclass")
        multiclass_mask = results.get("multiclass_mask")

        cobb_binary_deg = None
        if cobb_binary and cobb_binary.get("success"):
            cobb_binary_deg = float(cobb_binary["cobb_angle_deg"])

        if multiclass_mask is None or cobb_multi is None:
            # Multiclass not available — produce a minimal text-only annotation
            vis = image.copy()
            if cobb_binary_deg is not None:
                cv2.putText(
                    vis, f"Cobb (Binary): {cobb_binary_deg:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )
            vertebrae = results.get("vertebrae_detected", [])
            if vertebrae:
                text = f"Vertebrae: {', '.join(vertebrae[:10])}"
                if len(vertebrae) > 10:
                    text += f"... (+{len(vertebrae) - 10})"
                cv2.putText(
                    vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
            return vis

        # Pass the full binary dict so the visualization can overlay the
        # spline + inflection points (Ciclo 5.1), not just the angle number.
        return draw_cobb_angle_visualization(
            image=image,
            multiclass_mask=multiclass_mask,
            cobb_multiclass_result=cobb_multi,
            cobb_binary_result=cobb_binary,
            scheme=self.scheme,
        )
