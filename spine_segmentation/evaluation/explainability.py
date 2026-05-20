"""
Explainability module for medical AI compliance.
Generates visual explanations of model predictions so clinicians
can understand WHY a decision was made — not just WHAT it is.

Methods:
  1. Grad-CAM: Highlights which regions influenced the prediction most
  2. Attention Maps: Extracted from transformer models (MiT/SegFormer)
  3. Confidence Maps: Per-pixel prediction confidence
  4. Clinical Report: Structured text output with reasoning
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from spine_segmentation.config import OUTPUTS_DIR, MULTICLASS_COLORS
from spine_segmentation.data.transforms import denormalize_image
from spine_segmentation.data.class_mapping import get_class_names


# ============================================================================
# 1. Grad-CAM — Works on ALL models (CNN and Transformer)
# ============================================================================

def get_target_layer(model, model_name: str):
    """
    Get the appropriate target layer for Grad-CAM based on model architecture.
    Returns the last feature extraction layer of the encoder.
    """
    encoder = model.encoder

    if "resnet" in model_name:
        return encoder.layer4[-1]
    elif "efficientnet" in model_name:
        # Last block of EfficientNet
        return encoder._blocks[-1]
    elif "mit" in model_name:
        # Last transformer block of MiT (SegFormer)
        # MiT has patch_embed layers and block layers
        if hasattr(encoder, 'block4'):
            return encoder.block4[-1]
        elif hasattr(encoder, 'blocks'):
            return encoder.blocks[-1]
        else:
            # Fallback: try to find the last named module
            layers = list(encoder.named_modules())
            for name, module in reversed(layers):
                if 'block' in name.lower() or 'layer' in name.lower():
                    return module
    # Fallback
    modules = list(encoder.modules())
    return modules[-2]


class SemanticSegmentationTarget:
    """Target for Grad-CAM that focuses on a specific class."""

    def __init__(self, category: int, mask: torch.Tensor = None):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        if model_output.dim() == 4:
            output = model_output[0]  # Remove batch dim
        else:
            output = model_output

        if output.shape[0] == 1:
            # Binary segmentation
            return torch.sigmoid(output[0]).sum()
        else:
            # Multiclass - focus on specific class
            return output[self.category].sum()


def generate_gradcam(
    model,
    input_tensor: torch.Tensor,
    model_name: str,
    target_class: int = None,
    prediction_mask: np.ndarray = None,
    percentile_clip: float = 95.0,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap of which regions influenced the prediction.

    Ciclo 5.8 (fix O + R):
      - When `prediction_mask` is supplied, the heatmap is multiplied by it
        before normalization. Outside-spine pixels become 0, so the user only
        sees activations INSIDE the detected column. Eliminates the off-spine
        "noise" that confused users in the deployed app.
      - Percentile clipping (default p95) re-normalizes the heatmap so the
        hottest regions stand out. Without this, a few outlier pixels can
        compress the visible range and the heatmap looks washed-out.

    Args:
        model: segmentation model.
        input_tensor: (1, 3, H, W) normalized input.
        model_name: architecture name (selects the target layer).
        target_class: class index to explain. None = spine (binary).
        prediction_mask: optional (H, W) {0, 1} mask. When supplied, the
            heatmap is masked to this region.
        percentile_clip: percentile (0..100) at which to clip the heatmap
            for contrast enhancement. 95 = saturate the top 5% to 1.0.

    Returns:
        (H, W) heatmap in [0, 1].
    """
    target_layer = get_target_layer(model, model_name)

    if target_class is None:
        targets = [SemanticSegmentationTarget(0)]
    else:
        targets = [SemanticSegmentationTarget(target_class)]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    cam_arr = grayscale_cam[0]

    # Mask by predicted spine region (fix O).
    if prediction_mask is not None:
        mask_resized = prediction_mask
        if mask_resized.shape != cam_arr.shape:
            mask_resized = cv2.resize(
                prediction_mask.astype(np.uint8),
                (cam_arr.shape[1], cam_arr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        cam_arr = cam_arr * (mask_resized > 0).astype(cam_arr.dtype)

    # Percentile clip to lift the visible range (fix R). Only useful when the
    # cam has any positive signal — guard against an all-zero array.
    positives = cam_arr[cam_arr > 0]
    if positives.size > 0 and 0 < percentile_clip < 100:
        ceiling = float(np.percentile(positives, percentile_clip))
        if ceiling > 1e-6:
            cam_arr = np.clip(cam_arr / ceiling, 0.0, 1.0)

    return cam_arr.astype(np.float32)


# ============================================================================
# 2. Confidence Maps — Per-pixel prediction certainty
# ============================================================================

def generate_confidence_map(
    model,
    input_tensor: torch.Tensor,
    task: str = "binary",
    prediction_mask: np.ndarray = None,
) -> np.ndarray:
    """Per-pixel confidence map: how sure the model is about each pixel.

    High confidence (≈1) = model is certain. Low confidence (≈0) = model is
    uncertain. Clinical use: low-confidence regions inside the predicted
    spine deserve a human review.

    Ciclo 5.8 (fix O): with `prediction_mask`, the returned map is zero
    everywhere outside the predicted spine. The Gradio render layer then
    only paints color INSIDE the spine; the background stays neutral.
    Without the mask, the background reads as "high confidence" on a
    saturated colormap, which clutters the figure.

    Args:
        model: segmentation model.
        input_tensor: (1, 3, H, W) normalized input.
        task: 'binary' or 'multiclass'.
        prediction_mask: optional (H, W) {0, 1} mask of the predicted spine.

    Returns:
        (H, W) confidence in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)

    if task == "binary":
        probs = torch.sigmoid(logits)
        # Confidence = distance from 0.5 (decision boundary).
        confidence = (2 * torch.abs(probs - 0.5)).cpu().numpy()[0, 0]
    else:
        probs = torch.softmax(logits, dim=1)
        # Confidence = max probability (how sure about the chosen class).
        confidence = probs.max(dim=1)[0].cpu().numpy()[0]

    if prediction_mask is not None:
        mask = prediction_mask
        if mask.shape != confidence.shape:
            mask = cv2.resize(
                prediction_mask.astype(np.uint8),
                (confidence.shape[1], confidence.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        confidence = confidence * (mask > 0).astype(confidence.dtype)

    return confidence.astype(np.float32)


# ============================================================================
# 3. Clinical Explanation Panel — What the doctor sees
# ============================================================================

def generate_explanation_panel(
    model,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    model_name: str,
    task: str = "binary",
    prediction_mask: np.ndarray = None,
    save_path: str = None,
    image_name: str = "",
) -> np.ndarray:
    """
    Generate a complete clinical explanation panel with:
    1. Original radiograph
    2. Model prediction (segmentation)
    3. Grad-CAM heatmap (what the model looked at)
    4. Confidence map (how certain the model is)
    5. Combined overlay with explanations

    This is what the radiologist/clinician sees to understand and trust
    the model's decision.

    Args:
        model: Trained segmentation model
        input_tensor: (1, 3, H, W) preprocessed input
        original_image: (H, W, 3) original RGB image for display
        model_name: Architecture name
        task: 'binary' or 'multiclass'
        prediction_mask: Pre-computed prediction mask (optional)
        save_path: Path to save the panel
        image_name: Name for the title

    Returns:
        Panel image as numpy array
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Get prediction
    with torch.no_grad():
        logits = model(input_tensor.to(device))

    if task == "binary":
        pred_probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        pred_mask = (pred_probs > 0.5).astype(np.uint8)
    else:
        pred_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_mask = pred_probs.argmax(axis=0).astype(np.uint8)

    # 2. Generate Grad-CAM
    try:
        target_class = 0 if task == "binary" else None
        gradcam = generate_gradcam(model, input_tensor.to(device), model_name, target_class)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        gradcam = np.zeros_like(pred_mask, dtype=np.float32)

    # 3. Generate confidence map
    confidence = generate_confidence_map(model, input_tensor.to(device), task)

    # 4. Create the panel
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)

    img_display = original_image if original_image.max() > 1 else (original_image * 255).astype(np.uint8)
    img_float = img_display.astype(np.float32) / 255.0

    # Panel 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_display)
    ax1.set_title('1. Radiografia Original', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Prediction overlay
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = img_display.copy()
    if task == "binary":
        color_mask = np.zeros_like(overlay)
        color_mask[pred_mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    else:
        color_mask = np.zeros_like(overlay)
        for cid, color in MULTICLASS_COLORS.items():
            if cid == 0:
                continue
            color_mask[pred_mask == cid] = color
        overlay = cv2.addWeighted(overlay, 0.6, color_mask, 0.4, 0)
    ax2.imshow(overlay)
    ax2.set_title('2. Prediccion del Modelo', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Grad-CAM
    ax3 = fig.add_subplot(gs[0, 2])
    cam_display = show_cam_on_image(img_float, gradcam, use_rgb=True)
    ax3.imshow(cam_display)
    ax3.set_title('3. Grad-CAM\n(Regiones que influyen en la decision)', fontsize=11, fontweight='bold')
    ax3.axis('off')

    # Panel 4: Confidence map
    ax4 = fig.add_subplot(gs[0, 3])
    conf_display = ax4.imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(conf_display, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title('4. Mapa de Confianza\n(Verde=seguro, Rojo=revisar)', fontsize=11, fontweight='bold')
    ax4.axis('off')

    # Panel 5: Combined clinical view (bottom row, spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, 0:2])
    # Overlay with uncertainty highlighting
    uncertain_mask = confidence < 0.7
    combined = overlay.copy()
    combined[uncertain_mask] = combined[uncertain_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    ax5.imshow(combined.astype(np.uint8))
    ax5.set_title('5. Vista Clinica\n(Zonas rojas = baja confianza, el medico debe revisar)',
                  fontsize=11, fontweight='bold')
    ax5.axis('off')

    # Panel 6: Text report
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.axis('off')

    report_lines = [
        f"REPORTE DE ANALISIS AUTOMATICO",
        f"{'='*40}",
        f"Imagen: {image_name}",
        f"Modelo: {model_name}",
        f"Tarea: {'Segmentacion Binaria' if task == 'binary' else 'Segmentacion Multiclase (Vertebras)'}",
        f"",
        f"RESULTADOS:",
    ]

    if task == "binary":
        spine_pixels = pred_mask.sum()
        total_pixels = pred_mask.size
        report_lines.extend([
            f"  Columna detectada: {'Si' if spine_pixels > 100 else 'No'}",
            f"  Area: {spine_pixels:,} pixeles ({spine_pixels/total_pixels*100:.1f}%)",
        ])
    else:
        class_names = get_class_names("vertebrae_24")
        detected = []
        for c in range(1, 23):
            if (pred_mask == c).sum() > 50:
                detected.append(class_names.get(c, f"C{c}"))
        report_lines.extend([
            f"  Vertebras detectadas: {len(detected)}",
            f"  Lista: {', '.join(detected)}",
        ])

    avg_conf = confidence[pred_mask > 0].mean() if pred_mask.sum() > 0 else 0
    low_conf_pct = (confidence[pred_mask > 0] < 0.7).mean() * 100 if pred_mask.sum() > 0 else 0

    report_lines.extend([
        f"",
        f"CONFIANZA:",
        f"  Confianza promedio: {avg_conf:.1%}",
        f"  Pixeles con baja confianza: {low_conf_pct:.1f}%",
        f"",
        f"NOTA CLINICA:",
        f"  {'Alto' if avg_conf > 0.85 else 'Medio' if avg_conf > 0.7 else 'Bajo'} nivel de confianza.",
    ])

    if low_conf_pct > 20:
        report_lines.append(f"  ATENCION: {low_conf_pct:.0f}% de la segmentacion")
        report_lines.append(f"  tiene baja confianza. Se recomienda revision")
        report_lines.append(f"  manual por el especialista.")
    else:
        report_lines.append(f"  Prediccion estable. Verificar visualmente")
        report_lines.append(f"  la segmentacion antes de uso clinico.")

    report_lines.extend([
        f"",
        f"DISCLAIMER:",
        f"  Este sistema es una HERRAMIENTA DE APOYO.",
        f"  NO reemplaza el criterio del especialista.",
        f"  Toda decision clinica debe ser validada",
        f"  por un profesional de la salud calificado.",
    ])

    report_text = "\n".join(report_lines)
    ax6.text(0.05, 0.95, report_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f'Panel de Explicabilidad Clinica — {image_name}',
                 fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Panel guardado en: {save_path}")

    plt.close()
    return fig


def batch_generate_explanations(
    model,
    test_loader,
    model_name: str,
    task: str = "binary",
    device: str = "cuda",
    max_samples: int = 10,
    output_dir: str = None,
):
    """
    Generate explanation panels for multiple test samples.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        model_name: Architecture name
        task: 'binary' or 'multiclass'
        device: 'cuda' or 'cpu'
        max_samples: Max number of panels to generate
        output_dir: Directory to save panels
    """
    output_dir = Path(output_dir or OUTPUTS_DIR / "explanations")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    count = 0

    for batch in test_loader:
        if count >= max_samples:
            break

        images = batch["image"]
        names = batch["image_name"]

        for i in range(len(images)):
            if count >= max_samples:
                break

            img_tensor = images[i:i+1].to(device)
            img_vis = denormalize_image(images[i])
            name = names[i]

            save_path = str(output_dir / f"explanation_{name.replace('.jpg', '.png')}")

            generate_explanation_panel(
                model=model,
                input_tensor=img_tensor,
                original_image=img_vis,
                model_name=model_name,
                task=task,
                save_path=save_path,
                image_name=name,
            )

            count += 1
            print(f"  [{count}/{max_samples}] {name}")

    print(f"\n{count} paneles de explicabilidad guardados en {output_dir}")
