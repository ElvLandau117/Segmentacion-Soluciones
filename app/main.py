"""
app/main.py — Production entrypoint for the Spine Segmentation web app.

This is a thin shim. The real application lives in
`spine_segmentation.deployment.app` (the package keeps domain code in
one place; this entrypoint just satisfies the `app/` rubric convention).

Run with any of:
    python app/main.py
    python -m app.main
    # inside the container (see Dockerfile):
    python -m app.main

Configuration is fully driven by env vars (see .env.example).
"""

from spine_segmentation.deployment.app import main

if __name__ == "__main__":
    main()
