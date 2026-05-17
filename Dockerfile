# =============================================================================
# Dockerfile — Spine Segmentation Web App (Ciclo 4)
#
# CPU-only image. Weights are NOT baked in: the container downloads them
# from Hugging Face Hub at boot (see spine_segmentation/deployment/weights.py).
# This keeps the image small and makes weight swaps a runtime concern.
#
# Build:   docker build -t spine-app:latest .
# Run:     docker run --rm -p 7860:7860 \
#              -e HF_REPO_ID=<user>/spine-checkpoints \
#              -v spine_checkpoints:/data/checkpoints \
#              spine-app:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder: install Python deps into the global site-packages
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Build-time system deps (some wheels need gcc / libs to compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (much smaller than the CUDA build)
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
        torchvision --index-url https://download.pytorch.org/whl/cpu

# App dependencies (includes huggingface_hub for weight download)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2 — runtime: minimal image with non-root user
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Runtime system deps + curl for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (security best practice)
RUN useradd --create-home --shell /bin/bash --uid 1000 app

WORKDIR /home/app

# Copy installed Python packages from builder (system site-packages,
# readable by any user)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (NO checkpoints — they come from HF Hub at runtime)
COPY --chown=app:app spine_segmentation/ ./spine_segmentation/
COPY --chown=app:app app/ ./app/

# HF Hub cache directory. Mount a named volume here in docker-compose
# so weights survive container restarts.
RUN mkdir -p /data/checkpoints && chown -R app:app /data/checkpoints
ENV CHECKPOINTS_DIR=/data/checkpoints

# Sensible defaults — overridable via docker-compose / -e
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

USER app

# Healthcheck — Gradio responds at "/" with 200 once it has finished booting.
# start-period=120s gives the first HF download time to complete.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fs "http://localhost:${APP_PORT}/" >/dev/null || exit 1

# Run via the app/ shim so the entrypoint convention from the rubric holds
CMD ["python", "-m", "app.main"]
