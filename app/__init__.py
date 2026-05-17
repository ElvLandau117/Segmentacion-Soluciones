"""
app/ — entrypoint convention for the deployment (Ciclo 4).

The rubric mentions `app/` or `src/` as the standard place for the
application code. To honor that convention without breaking the imports
used by notebooks, scripts and the rest of the package, this folder
only contains a thin entrypoint (`main.py`) that re-exports the real
implementation from `spine_segmentation.deployment`.

See [README.md](../README.md#despliegue) for the full deployment story.
"""
