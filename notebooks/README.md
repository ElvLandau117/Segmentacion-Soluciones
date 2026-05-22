# Notebooks del proyecto

Carpeta con los 4 notebooks que documentan la experimentación, el
entrenamiento y la presentación del proyecto.

## Tabla de notebooks

| Notebook | Tamaño | Propósito | Estado en el pipeline desplegado |
|---|---|---|---|
| `01_EDA.ipynb` | 14 KB | Análisis exploratorio del **MaIA Scoliosis Dataset** (250 radiografías AP, 71 Normal + 179 Scoliosis). Visualizaciones de distribución de clases, ejemplos de máscaras binary + multiclass, estadísticas de tamaño de imagen. | Soporte conceptual — no genera artifacts del deploy. |
| `02_training_experiments.ipynb` | 30 KB | **Entrenamiento de los 5 modelos PyTorch + Segmentation Models PyTorch (SMP)**: U-Net+ResNet50, U-Net+EfficientNet-B4, DeepLabV3+ResNet50, U-Net+MiT-B3, MAnet+MiT-B5. Loss combinada WCE+GDice, AMP, MLflow tracking. | **🟢 DESPLEGADO** — los pesos `deeplabv3plus_resnet50_multiclass_best.pth` (multiclass ganador) y `unet_resnet50_binary_best.pth` (binary) generados aquí son los que sirve la app en HF Spaces. |
| `02b_training_alternativo_unet_keras.ipynb` | 5 MB | Pipeline alternativo en **Keras/TensorFlow** del equipo: U-Net + ResNet50 simple entrenado en Google Colab con GPU T4. Cubre segmentación binaria, multiclase y cálculo de Cobb. Reporta Dice binario ~0.88 y MAE Cobb ~2.3° en sus métricas internas. | Alternativa comparativa — NO se usa en el deploy. Queda como referencia del trabajo del equipo (paths Colab no reproducibles en local). |
| `03_informe_final.ipynb` | 35 KB | Notebook estilo informe: carga los pesos pre-entrenados (sin requerir GPU), reproduce métricas en CPU, genera visualizaciones comparativas de los 5 modelos. **Pensado para que el equipo pueda re-correrlo sin re-entrenar**. | Soporte para auditoría — carga lo que el deploy ya tiene. |

## Convención de nombres

- `0N_*.ipynb` — notebook principal del flujo `01 → 02 → 03`.
- `0Nb_*.ipynb` — variante / alternativa al notebook `0N` (no es el
  desplegado, pero queda como referencia comparativa).

## Cómo abrir los notebooks

### Local (recomendado para `01`, `02`, `03`):
```bash
jupyter notebook notebooks/
```

Requiere `requirements.txt` instalado (ver `README.md` raíz).

### Colab (para `02b`):
El notebook `02b_training_alternativo_unet_keras.ipynb` fue desarrollado
en Google Colab y tiene paths absolutos `/content/drive/MyDrive/...`. Para
re-ejecutarlo subir el archivo a Colab + el dataset en Google Drive
(propiedad U. Andes — solicitarlo al líder).

## Datos y pesos

- **Dataset MaIA Scoliosis**: no incluido en este repo (`.gitignore` por
  decisión, ya que es propiedad de U. Andes — ver `datos/README.md`).
- **Pesos pre-entrenados**: no incluidos en este repo, viven en HF Hub
  `ElvLandau/spine-checkpoints` (ver `modelos/README.md`). El notebook
  `03_informe_final.ipynb` los descarga automáticamente al ejecutarse
  si no están en `checkpoints/` local.
