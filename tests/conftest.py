"""
conftest.py — shared fixtures for the spine segmentation test suite.

The tests are intentionally lightweight: they verify wiring (config,
shims, error paths) without needing real model weights. Tests that
DO need weights are marked `requires_checkpoints` and skipped when
the .pth files are not present locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def dummy_radiograph() -> np.ndarray:
    """A small 'looks like a radiograph' RGB array. Not a real X-ray; just
    something the pipeline can swallow without crashing."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(256, 128, 3), dtype=np.uint8)


@pytest.fixture
def isolated_checkpoints_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CHECKPOINTS_DIR at an empty temp dir for a single test.
    Forces a config re-import so the override takes effect."""
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path))
    # Drop cached module so the next import re-reads env vars
    import sys
    for mod in list(sys.modules):
        if mod.startswith("spine_segmentation.config"):
            del sys.modules[mod]
    return tmp_path


def have_local_checkpoints() -> bool:
    """True iff there is at least one .pth in the default CHECKPOINTS_DIR."""
    from spine_segmentation.config import CHECKPOINTS_DIR
    return any(CHECKPOINTS_DIR.glob("*.pth"))


# Auto-skip the @requires_checkpoints marker when no .pth is present
def pytest_collection_modifyitems(config, items):
    if have_local_checkpoints():
        return
    skip_marker = pytest.mark.skip(reason="no .pth checkpoints available locally")
    for item in items:
        if "requires_checkpoints" in item.keywords:
            item.add_marker(skip_marker)
