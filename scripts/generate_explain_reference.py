"""generate_explain_reference.py — Ciclo 5.9 one-shot asset generator.

Produces the two bilingual reference PNGs that live above the dynamic
Explainability panel in the Gradio app:

    spine_segmentation/deployment/assets/explainability_reference_es.png
    spine_segmentation/deployment/assets/explainability_reference_en.png

These images are an educational legend — they teach a clinician how to read
the dynamic Grad-CAM + Confidence panel BEFORE seeing their own case. The
mockup comes from a clinical collaborator (medico companero); since the
source PSD/PNG was not available as a file at generation time, this script
recreates the layout from scratch with matplotlib so the ES and EN variants
stay 100% visually consistent.

Usage:
    python scripts/generate_explain_reference.py --lang es
    python scripts/generate_explain_reference.py --lang en
    python scripts/generate_explain_reference.py --lang both    # default

Run from the repo root. Requires matplotlib + numpy + opencv-python (already
listed in requirements.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke

# ---------------------------------------------------------------------------
# Bilingual strings — must match docs/PROMPT_PROXIMO_CHAT.md sec "Strings
# del mockup a traducir". Keep them as plain Latin-1-safe text (the matplotlib
# default font handles the accent marks fine).
# ---------------------------------------------------------------------------
STRINGS = {
    "es": {
        "tab_label": "Explainability",
        "panel_title": "Grad-CAM (izquierda) | Mapa de Confianza (derecha)",
        # 5 callouts: (number, color_name, title, body)
        "callouts": [
            (1, "red",    "Mayor activacion\ndel modelo",
             "Zonas calidas (rojo/\namarillo) indican regiones\nque mas influyeron en\nla prediccion."),
            (2, "green",  "Trayectoria esperada\nde la columna",
             "El modelo centra su\natencion en la columna\nvertebral, siguiendo su\ncurvatura anatomica."),
            (3, "orange", "Activacion fuera\nde la region anatomica",
             "Activacion en zonas\nexternas a la columna\n(bordes, artefactos o\nestructuras no relevantes)."),
            (4, "green",  "Alta confianza en la\ntrayectoria segmentada",
             "Las zonas verdes indican\nalta certeza del modelo\nen la prediccion pixel\na pixel."),
            (5, "orange", "Bordes de menor\ncerteza",
             "Los bordes (amarillo/\nnaranja) presentan menor\nconfianza y son criticos\npara el angulo de Cobb."),
        ],
        "cbar_left_low":  "Menor influencia",
        "cbar_left_high": "Mayor influencia",
        "cbar_right_low":  "Baja confianza",
        "cbar_right_high": "Alta confianza",
        "caption_left":  ("Grad-CAM (izquierda): Regiones que influyeron en la decision del modelo.\n"
                          "Zonas calidas (rojo) = alta influencia."),
        "caption_right": ("Mapa de Confianza (derecha): Certeza del modelo por pixel.\n"
                          "Verde = alta confianza | Rojo = baja confianza (el medico debe revisar)."),
        "disclaimer": ("Este sistema es una herramienta de apoyo. "
                       "No reemplaza el criterio del especialista."),
    },
    "en": {
        "tab_label": "Explainability",
        "panel_title": "Grad-CAM (left) | Confidence Map (right)",
        "callouts": [
            (1, "red",    "Highest model\nactivation",
             "Warm zones (red/yellow)\nshow the regions that\nmost influenced the\nprediction."),
            (2, "green",  "Expected spine\ntrajectory",
             "The model focuses on the\nvertebral column, following\nits anatomical curvature."),
            (3, "orange", "Activation outside the\nanatomical region",
             "Activation in regions\noutside the spine (borders,\nartifacts, or irrelevant\nstructures)."),
            (4, "green",  "High confidence in the\nsegmented trajectory",
             "Green zones indicate high\nmodel certainty in the\npixel-by-pixel prediction."),
            (5, "orange", "Lower-certainty\nedges",
             "Edges (yellow/orange)\nshow lower confidence\nand are critical for the\nCobb angle calculation."),
        ],
        "cbar_left_low":  "Lower influence",
        "cbar_left_high": "Higher influence",
        "cbar_right_low":  "Low confidence",
        "cbar_right_high": "High confidence",
        "caption_left":  ("Grad-CAM (left): Regions that influenced the model's decision.\n"
                          "Warm zones (red) = high influence."),
        "caption_right": ("Confidence Map (right): Per-pixel model certainty.\n"
                          "Green = high confidence | Red = low confidence (clinician must review)."),
        "disclaimer": ("This system is a support tool. "
                       "It does not replace the specialist's judgment."),
    },
}

# Colors for callout borders. Soft tones that read well over light backgrounds.
CALLOUT_COLOR = {
    "red":    "#E74C3C",
    "green":  "#27AE60",
    "orange": "#E67E22",
}

# ---------------------------------------------------------------------------
# Figure layout constants (Ciclo 5.11): keeping them at module level so the
# pixel-to-figure converter and _build_figure agree on the rectangles.
# Each tuple is (left, bottom, width, height) in figure fraction.
# ---------------------------------------------------------------------------
AX_CAM_RECT = (0.21, 0.30, 0.26, 0.58)
AX_CONF_RECT = (0.53, 0.30, 0.26, 0.58)


def _derive_visual_anchors(spine_mask: np.ndarray) -> dict:
    """Compute the visual anchors used by the figure: spine bbox, centroid,
    and 4 off-anatomy blob positions (above head, lateral left, pelvis,
    lateral right) — all relative to the spine bbox so they stay 'outside
    the spine' for any sample.

    Returns a dict of pixel coords (px, py) in the 512x512 image space:

        {
            "centroid":    (cx, cy),
            "bbox":        (xmin, ymin, xmax, ymax),
            "blob_top":    (cx, ymin - margin)          # above the spine
            "blob_pelvis": (cx, ymax + margin)          # below the spine
            "blob_left":   (xmin - margin, mid_y)       # left flank
            "blob_right":  (xmax + margin, lower_y)     # right flank
        }

    Used by ``_simulate_gradcam`` to place spurious off-spine activations
    AND by ``_build_figure`` to compute callout arrow targets in figure
    fraction. Both consumers stay in sync this way: every arrow always
    lands on either the spine (centroid / bbox edge) or a visible blob.
    """
    ys, xs = np.where(spine_mask > 0)
    if len(xs) == 0:
        # Empty mask -> fall back to image-center anchors so the script
        # still produces a non-crashing reference image.
        return {
            "centroid": (256, 256),
            "bbox": (200, 100, 312, 412),
            "blob_top": (256, 40),
            "blob_pelvis": (256, 470),
            "blob_left": (160, 280),
            "blob_right": (360, 320),
        }
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    cx, cy = int(xs.mean()), int(ys.mean())
    h = max(ymax - ymin, 1)
    return {
        "centroid":    (cx, cy),
        "bbox":        (xmin, ymin, xmax, ymax),
        "blob_top":    (cx,                       max(20,  ymin - 25)),
        "blob_pelvis": (cx,                       min(490, ymax + 50)),
        "blob_left":   (max(50,  xmin - 70),      ymin + h // 2),
        "blob_right":  (min(490, xmax + 70),      ymin + (2 * h) // 3),
    }


def _pixel_to_figure_coords(
    px: int, py: int,
    ax_rect: tuple[float, float, float, float],
    img_size: int = 512,
) -> tuple[float, float]:
    """Convert pixel (px, py) inside a ``size x size`` ``imshow`` to figure
    fraction (fx, fy) using the axes rect ``(left, bottom, width, height)``.

    Note: matplotlib figure y-axis is flipped vs image y-axis — image pixel
    (0, 0) sits at the TOP of the axes rect, which is ``bottom + height`` in
    figure coords.
    """
    left, bottom, width, height = ax_rect
    fx = left + (px / img_size) * width
    fy = bottom + ((img_size - py) / img_size) * height
    return (fx, fy)


def _load_and_resize_grayscale(path: Path, size: int = 512) -> np.ndarray:
    """Read a grayscale X-ray, letterbox-resize to (size, size)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    h, w = img.shape
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _load_and_resize_mask(path: Path, size: int = 512) -> np.ndarray:
    """Same letterbox resize as the radiograph, but with nearest interpolation."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    h, w = img.shape
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return (canvas > 127).astype(np.uint8)


def _simulate_gradcam(spine_mask: np.ndarray, anchors: dict, size: int = 512) -> np.ndarray:
    """Synthesize a Grad-CAM-like heatmap concentrated on the spine, with
    illustrative spurious hot-spots (off-spine) at positions derived from
    ``anchors`` (Ciclo 5.11 refactor) so callout #3 always has something
    visible to point at, regardless of which sample radiograph is used.

    Returns a float32 (size, size) array in [0, 1].
    """
    heat = spine_mask.astype(np.float32)
    # Strong activation along the spine; dilate so it bleeds slightly past
    # the mask edges (mimicking real Grad-CAM smoothing).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    heat = cv2.dilate(heat, kernel, iterations=2)
    heat = cv2.GaussianBlur(heat, (31, 31), 0)

    def gaussian_blob(canvas: np.ndarray, cx: int, cy: int, sigma: float, peak: float):
        ys, xs = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
        g = peak * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        np.maximum(canvas, g, out=canvas)

    # Off-anatomy hot-spots: each placed at an anchor derived from the spine
    # bbox so they sit OUTSIDE the spine even when the sample changes.
    bt_x, bt_y = anchors["blob_top"]
    bp_x, bp_y = anchors["blob_pelvis"]
    bl_x, bl_y = anchors["blob_left"]
    br_x, br_y = anchors["blob_right"]
    gaussian_blob(heat, cx=bt_x, cy=bt_y, sigma=22, peak=0.95)   # top-of-head (above spine)
    gaussian_blob(heat, cx=bl_x, cy=bl_y, sigma=28, peak=0.55)   # left flank artifact
    gaussian_blob(heat, cx=bp_x, cy=bp_y, sigma=30, peak=0.85)   # pelvis hotspot (callout #3)
    gaussian_blob(heat, cx=br_x, cy=br_y, sigma=20, peak=0.45)   # right side artifact

    # Normalize and percentile-clip (matches the live pipeline behavior from
    # Ciclo 5.8 fix R).
    heat = np.clip(heat, 0, None)
    p95 = np.percentile(heat[heat > 0], 95) if heat.any() else 1.0
    heat = np.clip(heat / max(p95, 1e-6), 0, 1)
    return heat.astype(np.float32)


def _simulate_confidence(spine_mask: np.ndarray) -> np.ndarray:
    """Synthesize a per-pixel confidence map: high in the spine interior,
    yellow/orange at the edges (where the model is less sure), 0 outside.

    Returns float32 in [0, 1].
    """
    mask = spine_mask.astype(np.float32)
    # Distance transform from the mask boundary, inside the spine, normalized
    # so the interior is ~1 and the edge is ~0.4.
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    if dt.max() > 0:
        dt = dt / dt.max()
    conf = np.where(mask > 0, 0.45 + 0.55 * dt, 0.0)
    conf = cv2.GaussianBlur(conf, (5, 5), 0)
    return conf.astype(np.float32)


def _blend_jet(gray: np.ndarray, heat: np.ndarray, mask: np.ndarray | None = None,
               alpha: float = 0.55) -> np.ndarray:
    """Overlay a jet-colored heatmap on a grayscale radiograph. When `mask` is
    given, pixels outside the mask keep the grayscale radiograph (no heat)."""
    cmap = plt.get_cmap("jet")
    colored = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = (alpha * colored + (1 - alpha) * base).astype(np.uint8)
    if mask is not None:
        outside = (mask == 0)[..., None]
        out = np.where(outside, base, out)
    return out


def _blend_rdylgn(gray: np.ndarray, conf: np.ndarray, mask: np.ndarray,
                  alpha: float = 0.75) -> np.ndarray:
    """Overlay a RdYlGn colormap on the radiograph, masked to the spine."""
    cmap = plt.get_cmap("RdYlGn")
    colored = (cmap(conf)[:, :, :3] * 255).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out = (alpha * colored + (1 - alpha) * base).astype(np.uint8)
    outside = (mask == 0)[..., None]
    return np.where(outside, base, out).astype(np.uint8)


def _draw_callout(ax, number: int, color_key: str, title: str, body: str,
                  box_xy: tuple[float, float], arrow_xy: tuple[float, float],
                  box_w: float = 0.165, box_h: float = 0.20) -> None:
    """Draw a numbered annotation box with a leader arrow to (arrow_xy).

    Coordinates are in axes fraction (0..1).
    """
    color = CALLOUT_COLOR[color_key]
    bx, by = box_xy

    # Background rounded rectangle for the callout
    fancy = mpatches.FancyBboxPatch(
        (bx, by), box_w, box_h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.6, edgecolor=color, facecolor="#FFFFFF",
        transform=ax.transAxes, zorder=10,
    )
    ax.add_patch(fancy)

    # Numbered circle in the top-left of the box
    circle = mpatches.Circle(
        (bx + 0.018, by + box_h - 0.022),
        radius=0.013, facecolor=color, edgecolor="white", linewidth=1.5,
        transform=ax.transAxes, zorder=11,
    )
    ax.add_patch(circle)
    ax.text(bx + 0.018, by + box_h - 0.022, str(number),
            color="white", fontsize=8, fontweight="bold",
            ha="center", va="center",
            transform=ax.transAxes, zorder=12)

    # Title (next to number). Up to 2 lines at fontsize 8 — taller titles
    # would push into the body region below.
    ax.text(bx + 0.038, by + box_h - 0.022, title,
            color="#222222", fontsize=7.8, fontweight="bold",
            ha="left", va="top",
            transform=ax.transAxes, zorder=12,
            linespacing=1.18)

    # Body starts comfortably below a 2-line title. Tuned empirically so 4
    # body lines at fontsize 6.2 sit inside box_h=0.20 with breathing room.
    ax.text(bx + 0.008, by + box_h - 0.082, body,
            color="#444444", fontsize=6.2, ha="left", va="top",
            transform=ax.transAxes, zorder=12,
            linespacing=1.18)

    # Leader arrow from the box edge nearest the target to arrow_xy.
    # Use the default "->" arrowstyle (small triangular head) — without it,
    # head_length and head_width are interpreted in points and scaled by
    # mutation_scale, producing huge filled triangles that obscure the panel.
    box_cx = bx + box_w / 2
    ax_x, _ = arrow_xy
    if ax_x > box_cx:
        anchor = (bx + box_w, by + box_h * 0.55)
    else:
        anchor = (bx, by + box_h * 0.55)
    arrow = mpatches.FancyArrowPatch(
        anchor, arrow_xy,
        arrowstyle="->",
        linestyle="--", color=color, linewidth=1.2,
        transform=ax.transAxes, zorder=11,
        mutation_scale=14,
    )
    ax.add_patch(arrow)


def _draw_horizontal_colorbar(ax, cmap_name: str, low_label: str, high_label: str) -> None:
    """Draw a horizontal colorbar with a single combined low/high label
    centered UNDER the bar — matches the mockup style and avoids the two
    bars' labels overlapping in the middle of the figure."""
    gradient = np.linspace(0, 1, 256)[None, :]
    ax.imshow(gradient, aspect="auto", cmap=cmap_name)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.8)
    # Combined label centered below the bar.
    ax.text(0.5, -0.9, f"{low_label}  <-->  {high_label}",
            ha="center", va="top",
            transform=ax.transAxes, fontsize=7.8, color="#333333")


def _build_figure(
    cam_img: np.ndarray,
    conf_img: np.ndarray,
    strings: dict,
    anchors: dict,
) -> plt.Figure:
    """Compose the full reference panel: title strip, 2 X-ray panels with
    callouts + arrows, 2 horizontal colorbars, captions, and disclaimer.

    Ciclo 5.11: ``anchors`` (output of ``_derive_visual_anchors``) is used
    to compute the 5 arrow targets in figure fraction so they always land
    on the spine or on a visible off-anatomy blob — invariant to the
    specific sample radiograph used as backdrop.
    """
    fig = plt.figure(figsize=(11.26, 7.16), dpi=100, facecolor="white")

    # ---- Title strip (mimics the app's tab label, for visual continuity)
    ax_title = fig.add_axes([0.04, 0.93, 0.92, 0.05])
    ax_title.set_facecolor("#EAF2FB")
    ax_title.text(0.012, 0.5, strings["panel_title"], ha="left", va="center",
                  fontsize=10, fontweight="bold", color="#1F4E79",
                  transform=ax_title.transAxes)
    ax_title.set_xticks([]); ax_title.set_yticks([])
    for spine in ax_title.spines.values():
        spine.set_edgecolor("#B0C4DE")

    # ---- Left panel (Grad-CAM)
    # Width 0.26 — narrower than 0.30 so the left-side callouts (x=0.025 +
    # box_w 0.165 ending at 0.19) clear the panel's left edge with ~2% margin.
    ax_cam = fig.add_axes(list(AX_CAM_RECT))
    ax_cam.imshow(cam_img)
    ax_cam.set_xticks([]); ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_edgecolor("#999999")

    # ---- Right panel (Confidence)
    # Width 0.26 ending at x=0.79 — so the right-side callouts (x=0.81 +
    # box_w 0.165 ending at 0.975) clear the panel's right edge with margin.
    ax_conf = fig.add_axes(list(AX_CONF_RECT))
    ax_conf.set_facecolor("#1F6B3A")  # dark green backdrop visible behind the spine
    ax_conf.imshow(conf_img)
    ax_conf.set_xticks([]); ax_conf.set_yticks([])
    for spine in ax_conf.spines.values():
        spine.set_edgecolor("#999999")

    # ---- 5 callouts: 3 around the cam panel, 2 around the confidence panel
    c1, c2, c3, c4, c5 = strings["callouts"]

    # Coordinates are in FIGURE fraction (so callouts can sit in the side margins)
    # and we use an invisible figure-level overlay axes for unified placement.
    overlay = fig.add_axes([0, 0, 1, 1], frameon=False)
    overlay.set_xticks([]); overlay.set_yticks([])
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    overlay.patch.set_alpha(0)

    # Derive arrow targets from anchors (Ciclo 5.11). Each maps a pixel
    # anchor inside the 512x512 imshow to a figure-fraction coordinate
    # via _pixel_to_figure_coords + the relevant ax rect.
    cx, cy = anchors["centroid"]
    xmin, _ymin, xmax, _ymax = anchors["bbox"]
    bt = anchors["blob_top"]
    bp = anchors["blob_pelvis"]

    arrow_top_cam        = _pixel_to_figure_coords(*bt,                 AX_CAM_RECT)
    arrow_center_cam     = _pixel_to_figure_coords(cx, cy,              AX_CAM_RECT)
    arrow_pelvis_cam     = _pixel_to_figure_coords(*bp,                 AX_CAM_RECT)
    arrow_center_conf    = _pixel_to_figure_coords(cx, cy,              AX_CONF_RECT)
    # Callout #5 anchor: lateral edge of the spine, slightly below midline
    # so the arrow lands clearly in the yellow/orange border zone.
    edge_x = xmax
    edge_y = cy + max(20, (anchors["bbox"][3] - anchors["bbox"][1]) // 4)
    arrow_edge_conf      = _pixel_to_figure_coords(edge_x, edge_y,      AX_CONF_RECT)

    # callouts on the LEFT side of the cam panel (1, 2, 3)
    _draw_callout(overlay, *c1, box_xy=(0.025, 0.70),
                  arrow_xy=arrow_top_cam)         # red hotspot top-of-head
    _draw_callout(overlay, *c2, box_xy=(0.025, 0.46),
                  arrow_xy=arrow_center_cam)      # spine center
    _draw_callout(overlay, *c3, box_xy=(0.025, 0.22),
                  arrow_xy=arrow_pelvis_cam)      # pelvis spurious hotspot

    # callouts on the RIGHT side of the confidence panel (4, 5).
    # Right panel now ends at x=0.79; callouts start at 0.81 so the body text
    # never bleeds back into the visualization.
    _draw_callout(overlay, *c4, box_xy=(0.810, 0.62),
                  arrow_xy=arrow_center_conf)     # green center of spine confidence
    _draw_callout(overlay, *c5, box_xy=(0.810, 0.32),
                  arrow_xy=arrow_edge_conf)       # edge of spine confidence

    # ---- Horizontal colorbars under each panel. Widths matched to panels.
    ax_cbar_left = fig.add_axes([0.22, 0.235, 0.23, 0.022])
    _draw_horizontal_colorbar(ax_cbar_left, "jet",
                              strings["cbar_left_low"], strings["cbar_left_high"])

    ax_cbar_right = fig.add_axes([0.54, 0.235, 0.23, 0.022])
    _draw_horizontal_colorbar(ax_cbar_right, "RdYlGn",
                              strings["cbar_right_low"], strings["cbar_right_high"])

    # ---- Captions under the colorbars (centered on each panel)
    overlay.text(0.335, 0.175, strings["caption_left"],
                 ha="center", va="top", fontsize=7.5, color="#333333",
                 linespacing=1.3)
    overlay.text(0.655, 0.175, strings["caption_right"],
                 ha="center", va="top", fontsize=7.5, color="#333333",
                 linespacing=1.3)

    # ---- Disclaimer footer (medical apoyo)
    ax_disclaimer = fig.add_axes([0.04, 0.025, 0.92, 0.06])
    ax_disclaimer.set_facecolor("#FAFAFA")
    for spine in ax_disclaimer.spines.values():
        spine.set_edgecolor("#CCCCCC")
    ax_disclaimer.set_xticks([]); ax_disclaimer.set_yticks([])
    ax_disclaimer.text(0.5, 0.5, strings["disclaimer"],
                       ha="center", va="center", fontsize=9.5,
                       color="#333333", fontweight="bold",
                       transform=ax_disclaimer.transAxes,
                       path_effects=[withStroke(linewidth=0)])

    return fig


def generate(lang: str, out_path: Path, sample_xray: Path, sample_mask: Path) -> None:
    """Generate one reference PNG for the given language and save it.

    Ciclo 5.11: anchors are derived from the spine mask once and threaded
    through both ``_simulate_gradcam`` (for blob placement) and
    ``_build_figure`` (for arrow targets). This keeps the callout arrows
    landing on visible content regardless of which sample radiograph is
    used as the backdrop.
    """
    gray = _load_and_resize_grayscale(sample_xray, size=512)
    mask = _load_and_resize_mask(sample_mask, size=512)
    anchors = _derive_visual_anchors(mask)
    cam = _simulate_gradcam(mask, anchors)
    conf = _simulate_confidence(mask)
    cam_img = _blend_jet(gray, cam)
    conf_img = _blend_rdylgn(gray, conf, mask)

    strings = STRINGS[lang]
    fig = _build_figure(cam_img, conf_img, strings, anchors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"[ok] {lang}: wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang", choices=["es", "en", "both"], default="both",
        help="Language(s) to generate (default: both).",
    )
    parser.add_argument(
        "--sample-xray",
        type=Path,
        default=Path("MaIA_Scoliosis_Dataset/Scoliosis/S_200.jpg"),
        help="Path to the X-ray image used as backdrop (default: S_200.jpg, "
             "chosen for Ciclo 5.10 — clearly visible S-shape suitable as "
             "an educational reference. Previously was S_22.jpg in Ciclo 5.9).",
    )
    parser.add_argument(
        "--sample-mask",
        type=Path,
        default=Path("MaIA_Scoliosis_Dataset/LabelBinaryJPG/Label_S_200.jpg"),
        help="Path to the matching binary spine mask (default: Label_S_200.jpg).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("spine_segmentation/deployment/assets"),
        help="Output directory (default: spine_segmentation/deployment/assets).",
    )
    args = parser.parse_args()

    langs = ["es", "en"] if args.lang == "both" else [args.lang]
    for lang in langs:
        out_path = args.out_dir / f"explainability_reference_{lang}.png"
        generate(lang, out_path, args.sample_xray, args.sample_mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
