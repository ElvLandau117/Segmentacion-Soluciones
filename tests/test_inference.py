"""
test_inference.py — verify the inference pipeline's API contract.

We test two layers:
  1. Pipeline initialization with no checkpoints (boundary case: must
     still return a usable object, just one that returns empty results).
  2. End-to-end predict() with real checkpoints — only runs when .pth
     files are present (marked requires_checkpoints).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_pipeline_with_no_checkpoints_returns_empty_results(dummy_radiograph):
    """When both checkpoints are None, predict() should produce a well-formed
    dict with None values instead of crashing — the app degrades to 'info only'."""
    from spine_segmentation.deployment.inference import SpineSegmentationPipeline

    pipeline = SpineSegmentationPipeline(
        binary_checkpoint=None,
        multiclass_checkpoint=None,
    )

    results = pipeline.predict(dummy_radiograph)

    # Contract: predict() always returns a dict with this exact shape
    expected_keys = {
        "original_image",
        "binary_mask",
        "multiclass_mask",
        "binary_overlay",
        "multiclass_overlay",
        "cobb_binary",
        "cobb_multiclass",
        "vertebrae_detected",
        "cobb_visualization",
    }
    assert expected_keys.issubset(results.keys())

    # With no models, the predictions stay empty/None
    assert results["binary_mask"] is None
    assert results["multiclass_mask"] is None
    assert results["cobb_binary"] is None
    assert results["cobb_multiclass"] is None
    assert results["vertebrae_detected"] == []


def test_pipeline_accepts_grayscale_via_app_wrapper():
    """The Gradio wrapper converts grayscale to RGB before calling predict.
    This test pins that contract: predict expects (H, W, 3) RGB uint8."""
    from spine_segmentation.deployment.inference import SpineSegmentationPipeline

    pipeline = SpineSegmentationPipeline()
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    results = pipeline.predict(rgb)  # must not raise on the (H,W,3) shape
    assert results["original_image"].shape == (128, 128, 3)


@pytest.mark.requires_checkpoints
@pytest.mark.slow
def test_pipeline_with_real_checkpoint(dummy_radiograph):
    """End-to-end smoke test against a real .pth.
    Skipped automatically when no checkpoints are present (see conftest)."""
    from spine_segmentation.config import CHECKPOINTS_DIR
    from spine_segmentation.deployment.inference import SpineSegmentationPipeline

    # Pick the first multiclass weight we find
    multi_pth = next(
        (p for p in CHECKPOINTS_DIR.glob("*multiclass*.pth")),
        None,
    )
    if multi_pth is None:
        pytest.skip("no multiclass .pth available")

    # Derive model name from filename convention: <model_name>_multiclass_best.pth
    model_name = multi_pth.stem.replace("_multiclass_best", "")

    pipeline = SpineSegmentationPipeline(
        multiclass_checkpoint=str(multi_pth),
        multiclass_model_name=model_name,
    )
    results = pipeline.predict(dummy_radiograph)

    assert results["multiclass_mask"] is not None
    assert results["multiclass_mask"].dtype == np.uint8
    # Cobb result is a dict (success or error), never None when we have a model
    assert isinstance(results["cobb_multiclass"], dict)
