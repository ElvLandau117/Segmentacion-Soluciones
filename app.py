"""
app.py — Hugging Face Spaces entrypoint.

HF Spaces (Gradio SDK) imports this file at boot and looks for a top-level
`demo` variable. We build the Blocks at import time so HF can introspect it,
and also call `.launch()` if the file is executed directly (so it works as
a plain `python app.py` locally).

The runtime configuration is fully env-driven:
  - HF_REPO_ID         (required) — model weights repo on HF Hub
  - HF_TOKEN           (optional) — only for private weight repos
  - CHECKPOINTS_DIR    (default /data/checkpoints on Spaces) — cache dir
  - MEDICAL_DISCLAIMER (optional) — override the disclaimer text

For Docker / Hetzner / local dev where you want full control of host & port,
use `python -m app.main` instead — that path honors APP_HOST / APP_PORT.

See docs/HF_SPACES_SETUP.md for the deployment runbook.
"""

from spine_segmentation.config import DEFAULT_BINARY_MODEL, DEFAULT_MULTICLASS_MODEL
from spine_segmentation.deployment.app import create_app
from spine_segmentation.deployment.weights import ensure_weights

# Boot-time: resolve weights. On HF Spaces, the first run downloads ~200 MB
# from HF Hub into the persistent storage; subsequent runs hit the cache.
_paths = ensure_weights()

# Top-level `demo` — HF Spaces discovers Gradio apps by this convention.
# Model names passed explicitly so the architecture matches the .pth files
# (binary = unet_resnet50, multiclass = deeplabv3plus_resnet50 — Ciclo 3 winner).
demo = create_app(
    binary_checkpoint=str(_paths["binary"]) if _paths["binary"] else None,
    multiclass_checkpoint=str(_paths["multiclass"]) if _paths["multiclass"] else None,
    binary_model_name=DEFAULT_BINARY_MODEL,
    multiclass_model_name=DEFAULT_MULTICLASS_MODEL,
)


if __name__ == "__main__":
    # Bind to 0.0.0.0 explicitly. HF Spaces sometimes re-invokes app.py as a
    # script (not import) after a worker crash; without server_name="0.0.0.0"
    # Gradio refuses to bind in the container and raises:
    #   "When localhost is not accessible, a shareable link must be created"
    # Local runs also work — Gradio accepts 0.0.0.0 and binds to all interfaces.
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
