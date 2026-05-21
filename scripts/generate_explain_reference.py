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


def _simulate_gradcam(spine_mask: np.ndarray, size: int = 512) -> np.ndarray:
    """Synthesize a Grad-CAM-like heatmap concentrated on the spine, with a
    few illustrative spurious hot-spots (off-spine) so callout #3 makes sense.

    Returns a float32 (size, size) array in [0, 1].
    """
    heat = spine_mask.astype(np.float32)
    # Strong activation along the spine; dilate so it bleeds slightly past
    # the mask edges (mimicking real Grad-CAM smoothing).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    heat = cv2.dilate(heat, kernel, iterations=2)
    heat = cv2.GaussianBlur(heat, (31, 31), 0)

    # A couple of intentional "spurious" hot-spots so the off-anatomy callout
    # has something to point to. Coordinates chosen by eye for the S_22 layout.
    def gaussian_blob(canvas: np.ndarray, cx: int, cy: int, sigma: float, peak: float):
        ys, xs = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
        g = peak * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        np.maximum(canvas, g, out=canvas)

    gaussian_blob(heat, cx=240, cy=70,  sigma=22, peak=0.95)   # top-of-head (above spine)
    gaussian_blob(heat, cx=120, cy=350, sigma=28, peak=0.55)   # left flank artifact
    gaussian_blob(heat, cx=235, cy=470, sigma=30, peak=0.85)   # pelvis hotspot
    gaussian_blob(heat, cx=420, cy=420, sigma=20, peak=0.45)   # right side artifact

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


def _build_figure(cam_img: np.ndarray, conf_img: np.ndarray, strings: dict) -> plt.Figure:
    """Compose the full reference panel: title strip, 2 X-ray panels with
    callouts + arrows, 2 horizontal colorbars, captions, and disclaimer."""
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
    ax_cam = fig.add_axes([0.21, 0.30, 0.26, 0.58])
    ax_cam.imshow(cam_img)
    ax_cam.set_xticks([]); ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_edgecolor("#999999")

    # ---- Right panel (Confidence)
    # Width 0.26 ending at x=0.79 — so the right-side callouts (x=0.81 +
    # box_w 0.165 ending at 0.975) clear the panel's right edge with margin.
    ax_conf = fig.add_axes([0.53, 0.30, 0.26, 0.58])
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

    # callouts on the LEFT side of the cam panel (1, 2, 3)
    # arrow_xy targets land inside the (now narrower) cam panel at x in
    # [0.21, 0.47]; tuned by eye against S_22 simulated overlays.
    _draw_callout(overlay, *c1, box_xy=(0.025, 0.70),
                  arrow_xy=(0.30, 0.83))     # red hotspot top-of-head
    _draw_callout(overlay, *c2, box_xy=(0.025, 0.46),
                  arrow_xy=(0.32, 0.55))     # spine center
    _draw_callout(overlay, *c3, box_xy=(0.025, 0.22),
                  arrow_xy=(0.31, 0.36))     # bottom spurious hotspot

    # callouts on the RIGHT side of the confidence panel (4, 5).
    # Right panel now ends at x=0.79; callouts start at 0.81 so the body text
    # never bleeds back into the visualization.
    _draw_callout(overlay, *c4, box_xy=(0.810, 0.62),
                  arrow_xy=(0.69, 0.65))     # green center of spine confidence
    _draw_callout(overlay, *c5, box_xy=(0.810, 0.32),
                  arrow_xy=(0.69, 0.40))     # edge of spine confidence

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
    """Generate one reference PNG for the given language and save it."""
    gray = _load_and_resize_grayscale(sample_xray, size=512)
    mask = _load_and_resize_mask(sample_mask, size=512)
    cam = _simulate_gradcam(mask)
    conf = _simulate_confidence(mask)
    cam_img = _blend_jet(gray, cam)
    conf_img = _blend_rdylgn(gray, conf, mask)

    strings = STRINGS[lang]
    fig = _build_figure(cam_img, conf_img, strings)
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
        default=Path("MaIA_Scoliosis_Dataset/Scoliosis/S_22.jpg"),
        help="Path to the X-ray image used as backdrop (default: S_22.jpg).",
    )
    parser.add_argument(
        "--sample-mask",
        type=Path,
        default=Path("MaIA_Scoliosis_Dataset/LabelBinaryJPG/Label_S_22.jpg"),
        help="Path to the matching binary spine mask (default: Label_S_22.jpg).",
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
