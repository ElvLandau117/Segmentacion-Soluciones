"""
test_weights.py — verify the weight-resolution logic in deployment.weights.

ensure_weights() must:
  1. Return paths to existing local files.
  2. Return None (gracefully, no crash) when neither cache nor HF is available.
  3. NOT attempt a network call when HF_REPO_ID is empty.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _fresh_imports():
    """Force re-import of config + weights so env-var changes take effect."""
    for mod in list(sys.modules):
        if mod.startswith("spine_segmentation.config") or \
           mod.startswith("spine_segmentation.deployment.weights"):
            del sys.modules[mod]


def test_uses_local_file_when_present(monkeypatch, tmp_path):
    """If the .pth is already in CHECKPOINTS_DIR, return its Path immediately."""
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setenv("HF_REPO_ID", "")
    _fresh_imports()

    from spine_segmentation.config import DEFAULT_MULTICLASS_WEIGHT
    fake_pth = tmp_path / DEFAULT_MULTICLASS_WEIGHT
    fake_pth.write_bytes(b"\x00" * 16)  # any bytes — not a real checkpoint

    from spine_segmentation.deployment.weights import ensure_weights
    paths = ensure_weights()

    assert paths["multiclass"] == fake_pth
    assert paths["binary"] is None  # not present, no HF -> None


def test_returns_none_without_hf_repo_and_no_cache(monkeypatch, tmp_path):
    """No HF_REPO_ID and no cached file -> None for both, no exception."""
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setenv("HF_REPO_ID", "")
    _fresh_imports()

    from spine_segmentation.deployment.weights import ensure_weights
    paths = ensure_weights()

    assert paths == {"multiclass": None, "binary": None}


def test_does_not_call_hf_when_repo_id_empty(monkeypatch, tmp_path):
    """Crucially: the function must not import huggingface_hub when there's nothing to fetch."""
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setenv("HF_REPO_ID", "")
    _fresh_imports()

    # Sabotage hf_hub_download: if it's called, the test fails loudly.
    import huggingface_hub
    def _explode(*args, **kwargs):
        raise AssertionError("hf_hub_download was called despite HF_REPO_ID=''")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _explode)

    from spine_segmentation.deployment.weights import ensure_weights
    paths = ensure_weights()  # must not raise

    assert paths == {"multiclass": None, "binary": None}


def test_custom_filenames_are_respected(monkeypatch, tmp_path):
    """Passing explicit names overrides the config defaults."""
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path))
    monkeypatch.setenv("HF_REPO_ID", "")
    _fresh_imports()

    (tmp_path / "custom_multi.pth").write_bytes(b"\x00")
    (tmp_path / "custom_bin.pth").write_bytes(b"\x00")

    from spine_segmentation.deployment.weights import ensure_weights
    paths = ensure_weights(
        multiclass_weight="custom_multi.pth",
        binary_weight="custom_bin.pth",
    )

    assert paths["multiclass"] == tmp_path / "custom_multi.pth"
    assert paths["binary"] == tmp_path / "custom_bin.pth"
