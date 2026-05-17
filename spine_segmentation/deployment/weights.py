"""
weights.py — resolve and download model checkpoints.

Single source of truth for "where are the .pth files?". Tries local cache
first (CHECKPOINTS_DIR); falls back to Hugging Face Hub if HF_REPO_ID is
configured. Avoids re-downloading on container restart because hf_hub_download
checks the file hash against the local cache.

Usage:
    from spine_segmentation.deployment.weights import ensure_weights

    paths = ensure_weights()
    pipeline = SpineSegmentationPipeline(
        multiclass_checkpoint=str(paths["multiclass"]),
        binary_checkpoint=str(paths["binary"]),
        ...
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from spine_segmentation.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BINARY_WEIGHT,
    DEFAULT_MULTICLASS_WEIGHT,
    HF_REPO_ID,
    HF_TOKEN,
)


class WeightsNotAvailable(RuntimeError):
    """Raised when a required weight is neither cached locally nor downloadable."""


def ensure_weights(
    multiclass_weight: Optional[str] = None,
    binary_weight: Optional[str] = None,
) -> dict[str, Optional[Path]]:
    """
    Make sure the requested weights are available on disk.

    For each weight:
      1. If already in CHECKPOINTS_DIR, use it.
      2. Otherwise, if HF_REPO_ID is set, download from Hugging Face Hub
         into CHECKPOINTS_DIR.
      3. Otherwise, return None for that weight (the app will start without it
         and the corresponding UI tab will be informational only).

    Args:
        multiclass_weight: filename of the multiclass .pth inside the HF repo.
            Defaults to DEFAULT_MULTICLASS_WEIGHT from config.
        binary_weight: filename of the binary .pth. Defaults to DEFAULT_BINARY_WEIGHT.

    Returns:
        {"multiclass": Path | None, "binary": Path | None}
    """
    mc_name = multiclass_weight or DEFAULT_MULTICLASS_WEIGHT
    bin_name = binary_weight or DEFAULT_BINARY_WEIGHT

    return {
        "multiclass": _resolve_one(mc_name),
        "binary": _resolve_one(bin_name),
    }


def _resolve_one(filename: str) -> Optional[Path]:
    """Resolve a single weight: local cache first, then HF Hub."""
    local_path = CHECKPOINTS_DIR / filename
    if local_path.exists():
        return local_path

    if not HF_REPO_ID:
        print(
            f"[weights] {filename} not found locally and HF_REPO_ID is not set; "
            f"the corresponding model will not be loaded."
        )
        return None

    # Lazy import: huggingface_hub is only needed when we actually have to download
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise WeightsNotAvailable(
            "huggingface_hub is not installed but HF_REPO_ID is set. "
            "Run: pip install huggingface_hub"
        ) from exc

    print(f"[weights] Downloading {filename} from {HF_REPO_ID} ...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=str(CHECKPOINTS_DIR),
        token=HF_TOKEN or None,
    )
    print(f"[weights] -> {downloaded}")
    return Path(downloaded)
