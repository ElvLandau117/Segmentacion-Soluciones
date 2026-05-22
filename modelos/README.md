# modelos/

Carpeta requerida por la rúbrica de despliegue de la actividad de
Coursera / Universidad de los Andes.

## ¿Por qué esta carpeta está vacía en GitHub?

Los pesos de los modelos (`*.pth`, formato PyTorch FP32) son **archivos
grandes** (102 MB el de DeepLabV3+ multiclase + 124 MB el de U-Net binario
= **226 MB total**). GitHub limita a 100 MB por archivo y desincentiva
blobs binarios grandes en el historial de commits, especialmente cuando
los pesos cambian con cada entrenamiento.

**Decisión arquitectónica (Ciclo 4 — 2026-05-17)**: distribuir los pesos
via **Hugging Face Hub** en un repo separado, manteniendo Git para el
código.

## Dónde están los pesos en realidad

| Modelo | Tarea | Path en HF Hub | Tamaño |
|---|---|---|---|
| **DeepLabV3+ ResNet50** | Segmentación multiclase (24 vértebras) | `ElvLandau/spine-checkpoints/deeplabv3plus_resnet50_multiclass_best.pth` | 102 MB |
| **U-Net ResNet50** | Segmentación binaria (columna completa) | `ElvLandau/spine-checkpoints/unet_resnet50_binary_best.pth` | 124 MB |

Repo público en HF Hub: **https://huggingface.co/ElvLandau/spine-checkpoints**

## ¿Por qué estos 2 modelos en producción?

De los 5 modelos entrenados y comparados (`notebooks/02_training_experiments.ipynb`),
**DeepLabV3+ ResNet50** ganó la comparativa multiclass con Dice 0.3378
(ver tabla en `README.md` raíz sección 9). Para Cobb angle, el método
binary (skeleton del U-Net binario) tiene mejor MAE y correlación que el
método multiclass (endplate), por eso ambos modelos coexisten en producción.

## Cómo descargar los pesos

### Opción A — automática (recomendada en el Space)

La app en producción descarga los pesos automáticamente al primer arranque
via `spine_segmentation/deployment/weights.py`. No requiere acción manual.

### Opción B — manual (para entrenar localmente o reproducir métricas)

```bash
huggingface-cli login
python -c "
from spine_segmentation.deployment.weights import ensure_weights
ensure_weights()
"
```

Los archivos quedan en `checkpoints/` (carpeta gitignored, no se sube al repo).

### Opción C — subir pesos NUEVOS

Si re-entrenas y quieres publicar un peso nuevo:

```bash
python scripts/upload_weights.py --file checkpoints/<nuevo_modelo>.pth
```

(Requiere `HF_TOKEN` con permisos de escritura sobre `ElvLandau/spine-checkpoints`.)

## Más información

- `docs/HUGGINGFACE_SETUP.md` — onboarding completo de HF Hub.
- `README.md` raíz sección 8 — consideraciones del modelo en producción.
- `AGENTS.md` sec 9 — historial de decisiones sobre arquitectura de pesos.
