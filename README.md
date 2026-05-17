# Segmentación Automática de Columna Vertebral y Vértebras

## Proyecto Final — Maestría en Inteligencia Artificial

**Universidad de los Andes**

Sistema de deep learning para la segmentación automática de la columna vertebral y vértebras individuales en radiografías AP de pacientes sanos y con escoliosis, con cálculo automatizado del ángulo de Cobb y explicabilidad clínica integrada.

---

## 🚀 Ruta sugerida de lectura

**¿Eres nuevo en el proyecto?** Sigue esta ruta de onboarding (~85 min total):

| # | Archivo | Para qué |
|---|---------|----------|
| 01 | [`README.md`](README.md) | Visión general, instalación, uso |
| 02 | [`AGENTS.md`](AGENTS.md) | Decisiones, problemas conocidos, métricas |
| 03 | [`WORKFLOW.md`](WORKFLOW.md) | Reglas del repositorio, convenciones |
| 04 | [`docs/CICLO_3_ARTEFACTOS.md`](docs/CICLO_3_ARTEFACTOS.md) | Estado actual del proyecto |
| 05 | [`docs/CICLO_4_DESPLIEGUE_BRIEF.md`](docs/CICLO_4_DESPLIEGUE_BRIEF.md) | Próximo ciclo (despliegue) |
| 06 | [`notebooks/03_informe_final.ipynb`](notebooks/03_informe_final.ipynb) | Notebook ejecutable (CPU) |

Detalles: [`docs/RUTA_LECTURA.md`](docs/RUTA_LECTURA.md)

**¿Vas a continuar el trabajo en un nuevo chat con Claude?**
→ Usa el prompt listo en [`docs/PROMPT_PROXIMO_CHAT.md`](docs/PROMPT_PROXIMO_CHAT.md)

---

## Tabla de Contenido

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Estructura del Repositorio](#estructura-del-repositorio)
5. [Uso Rápido (Sin GPU - para el equipo)](#uso-rápido-sin-gpu---para-el-equipo)
6. [Entrenamiento (Requiere GPU)](#entrenamiento-requiere-gpu)
7. [Evaluación](#evaluación)
8. [Aplicación Web (Gradio)](#aplicación-web-gradio)
9. [Despliegue con Docker](#despliegue-con-docker)
10. [Arquitecturas Implementadas](#arquitecturas-implementadas)
11. [Resultados Obtenidos](#resultados-obtenidos)
12. [Problemas Conocidos del Dominio](#problemas-conocidos-del-dominio)
13. [Flujo de Trabajo del Equipo](#flujo-de-trabajo-del-equipo)

---

## Descripción del Proyecto

### Problema
La escoliosis es una deformidad tridimensional de la columna vertebral que afecta aproximadamente al 2-3% de la población. Su diagnóstico y seguimiento requieren mediciones precisas del ángulo de Cobb sobre radiografías de columna completa, un proceso manual que es:
- Subjetivo (variabilidad inter-observador)
- Lento (5-10 min por radiografía)
- Dependiente de la experiencia del radiólogo

### Solución
Sistema automatizado que:
1. **Segmenta** la columna vertebral y cada vértebra individual (C3-L5, 24 clases) usando deep learning
2. **Calcula** el ángulo de Cobb automáticamente (dos métodos: desde esqueleto binario y desde orientación de placas vertebrales)
3. **Explica** cada predicción con Grad-CAM y mapas de confianza (NO es una caja negra)
4. **Se despliega** en servidor (Docker) y es accesible via web (Gradio) o desde una tablet

### Dataset
**MaIA Scoliosis Dataset** (propiedad Universidad de los Andes):
- 250 radiografías AP de columna completa
- 71 pacientes normales, 179 con escoliosis
- Máscaras de segmentación binaria (columna vs fondo)
- Máscaras multiclase (36 clases originales → 24 clínicamente relevantes)
- Ángulo de Cobb ground truth para los 179 casos de escoliosis

**Split:** 70% entrenamiento (174) / 15% validación (38) / 15% test (38), estratificado por condición, seed=42.

---

## Requisitos

### Hardware
- **Para entrenar:** GPU NVIDIA con al menos 8GB VRAM (recomendado: RTX 3060+ o superior). El proyecto fue entrenado en RTX 4060 Ti 16GB.
- **Para inferencia/evaluación:** CPU suficiente. No se necesita GPU.

### Software
- Python 3.11+
- CUDA 12.4 (si se usa GPU)
- Git

---

## Instalación

### Opción 1: Con GPU (para entrenar)

```bash
# Clonar el repositorio
git clone https://github.com/ElvLandau117/Segmentacion-Soluciones.git
cd Segmentacion-Soluciones

# Crear entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalar dependencias del proyecto
pip install -r requirements.txt

# Verificar GPU
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Opción 2: Solo CPU (para inferencia)

```bash
# Clonar el repositorio
git clone https://github.com/ElvLandau117/Segmentacion-Soluciones.git
cd Segmentacion-Soluciones

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows

# Instalar PyTorch CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias
pip install -r requirements.txt
```

### Dataset y pesos pre-entrenados

> **Importante:** El dataset y los pesos NO están en el repositorio por tamaño y restricciones de propiedad intelectual.

1. **Dataset:** Solicitar al profesor la carpeta `MaIA_Scoliosis_Dataset/` y colocarla en la raíz del proyecto
2. **Pesos pre-entrenados:** Descargar desde el enlace de OneDrive compartido por el líder del equipo:
   - Archivo: `paquete_equipo_onedrive.zip` (~776 MB)
   - Contiene los 5 modelos multiclase + documentación
   - Descomprimir y mover la carpeta `checkpoints/` a la raíz del proyecto

Estructura esperada después:
```
Segmentacion-Soluciones/
├── MaIA_Scoliosis_Dataset/   # ← solicitar al profesor
├── checkpoints/              # ← solicitar al líder
│   ├── unet_resnet50_multiclass_best.pth
│   ├── manet_mit_b5_multiclass_best.pth
│   └── ...
└── ...
```

---

## Estructura del Repositorio

```
Segmentacion-Soluciones/
├── README.md                      ← Este archivo
├── AGENTS.md                      ← Memoria persistente del proyecto (leer para contexto completo)
├── requirements.txt
├── Dockerfile                     ← Deploy en servidor (Hetzner, AWS, etc.)
├── .gitignore
├── data_splits.json               ← División train/val/test fija (seed=42)
│
├── spine_segmentation/            ← Paquete Python principal
│   ├── config.py                  ← Configuración: rutas, modelos, hiperparámetros
│   ├── data/
│   │   ├── dataset.py             ← Datasets PyTorch (binario + multiclase)
│   │   ├── transforms.py          ← Augmentation con Albumentations
│   │   ├── splits.py              ← Split estratificado reproducible
│   │   └── class_mapping.py       ← Esquemas de clases (24 vértebras, 36 full, 5 regional)
│   ├── models/
│   │   ├── smp_models.py          ← Factory: create_model(name, num_classes)
│   │   └── losses.py              ← Weighted CE + Generalized Dice, Focal, etc.
│   ├── training/
│   │   └── trainer.py             ← Loop de entrenamiento + MLflow + AMP + early stopping
│   ├── evaluation/
│   │   ├── metrics.py             ← Dice, IoU, pixel accuracy, per-class
│   │   ├── visualize.py           ← Overlays y gráficas
│   │   ├── cobb_angle.py          ← Cálculo del ángulo de Cobb (2 métodos)
│   │   └── explainability.py      ← Grad-CAM + mapas de confianza + panel clínico
│   ├── postprocessing/
│   │   ├── morphology.py          ← Limpieza de máscaras, esqueletización
│   │   └── vertebra_ordering.py   ← Ordenamiento anatómico, endplates
│   └── deployment/
│       ├── app.py                 ← Aplicación web Gradio con explicabilidad
│       └── inference.py           ← Pipeline end-to-end de inferencia
│
├── scripts/
│   ├── train_multiclass.py        ← Entrenar modelos multiclase (principal)
│   ├── train_binary.py            ← Entrenar modelos binarios (opcional)
│   ├── evaluate.py                ← Evaluación comparativa de todos los modelos
│   ├── compute_cobb_angles.py     ← Cálculo batch de ángulos de Cobb
│   └── explore_data.py            ← EDA del dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb               ← Análisis exploratorio
│   └── 02_training_experiments.ipynb  ← Notebook maestro (carga pesos, evalúa, visualiza)
│
├── checkpoints/                   ← Pesos .pth (NO en git, compartir aparte)
├── outputs/                       ← Figuras, tablas, paneles (se generan)
├── mlruns/                        ← MLflow tracking (se genera)
└── MaIA_Scoliosis_Dataset/        ← Dataset (NO en git, propiedad U. Andes)
```

---

## Uso Rápido (Sin GPU - para el equipo)

Si ya tienes los pesos pre-entrenados en `checkpoints/`:

### 1. Abrir el notebook maestro
```bash
jupyter notebook notebooks/02_training_experiments.ipynb
```

Este notebook permite:
- Cargar cualquier modelo pre-entrenado
- Visualizar predicciones en el test set
- Calcular el ángulo de Cobb
- Generar paneles de explicabilidad
- **NO requiere GPU** (corre en CPU)

**IR DIRECTAMENTE A LA SECCIÓN 4 del notebook** (omitir la sección 3 de entrenamiento).

### 2. Lanzar la aplicación web
```bash
python -m spine_segmentation.deployment.app
```

Abre en el navegador: `http://localhost:7860`

Funciones:
- Subir una radiografía (drag-and-drop)
- Ver segmentación binaria
- Ver segmentación de vértebras individuales
- Ver cálculo del ángulo de Cobb
- Ver Grad-CAM y mapa de confianza (explicabilidad)
- Reporte diagnóstico automático

---

## Entrenamiento (Requiere GPU)

### Entrenar un modelo específico

```bash
# Modelos multiclase (RECOMENDADO — el enfoque principal del proyecto)
python scripts/train_multiclass.py --model unet_resnet50 --scheme vertebrae_24
python scripts/train_multiclass.py --model unet_efficientnet_b4 --scheme vertebrae_24
python scripts/train_multiclass.py --model deeplabv3plus_resnet50 --scheme vertebrae_24
python scripts/train_multiclass.py --model unet_mit_b3 --scheme vertebrae_24
python scripts/train_multiclass.py --model manet_mit_b5 --scheme vertebrae_24
```

### Configuración de entrenamiento

Editar `spine_segmentation/config.py`:

```python
TRAIN_CONFIG = {
    "image_size": 512,
    "batch_size": 8,              # Ajustar según VRAM disponible
    "num_epochs": 150,
    "encoder_lr": 1e-5,           # LR bajo para encoder pre-entrenado
    "decoder_lr": 1e-4,           # LR alto para decoder nuevo
    "early_stopping_patience": 20,
    "use_amp": True,              # Mixed precision (FP16)
}
```

### Tiempo estimado (RTX 4060 Ti 16GB)
- ~30-45 minutos por modelo (con early stopping)
- Total de 5 modelos: ~3-4 horas

### Tracking con MLflow

```bash
# Durante/después del entrenamiento
mlflow ui
# Abrir http://localhost:5000
```

---

## Evaluación

### Evaluación comparativa de todos los modelos

```bash
python scripts/evaluate.py
```

Genera:
- `outputs/model_comparison.csv` — Tabla comparativa de métricas
- `outputs/per_class_dice_*.png` — Gráficas de Dice por vértebra
- `outputs/cobb_*.png` — Bland-Altman plots del ángulo de Cobb

### Cálculo batch de ángulos de Cobb

```bash
python scripts/compute_cobb_angles.py
```

Compara predicciones vs ground truth:
- MAE (grados)
- Correlación de Pearson
- % casos con error < 5° y < 10°

### Generar paneles de explicabilidad

Ver el notebook `02_training_experiments.ipynb`, sección de explicabilidad, o importar:

```python
from spine_segmentation.evaluation.explainability import generate_explanation_panel

generate_explanation_panel(
    model=model,
    input_tensor=img_tensor,
    original_image=img_np,
    model_name="manet_mit_b5",
    task="multiclass",
    save_path="outputs/explanation.png",
)
```

---

## Aplicación Web (Gradio)

### Local

```bash
python -m spine_segmentation.deployment.app
```

Acceder a `http://localhost:7860`.

### Con parámetros

```bash
python -m spine_segmentation.deployment.app \
    --binary-checkpoint checkpoints/unet_resnet50_binary_best.pth \
    --multiclass-checkpoint checkpoints/manet_mit_b5_multiclass_best.pth \
    --binary-model unet_resnet50 \
    --multiclass-model manet_mit_b5 \
    --port 7860
```

### Funcionalidades de la app

- **Tab Binary Segmentation:** Overlay de la columna segmentada
- **Tab Vertebrae Segmentation:** Vértebras individuales coloreadas
- **Tab Cobb Angle:** Visualización con líneas de medición
- **Tab Explainability:**
  - Grad-CAM: qué regiones miró el modelo
  - Mapa de confianza: verde (seguro) → rojo (revisar)
- **Diagnosis Results:** Texto con vértebras detectadas, ángulo de Cobb, severidad

---

## Despliegue con Docker

El proyecto incluye un `Dockerfile` optimizado para deploy en servidor (CPU, ~2-3GB imagen).

### Construir la imagen

```bash
docker build -t spine-segmentation .
```

### Ejecutar el contenedor

```bash
docker run -d -p 7860:7860 --name spine-app spine-segmentation
```

Acceder a `http://<ip-servidor>:7860`.

### Deploy en Hetzner / AWS / GCP

1. Subir el container registry (Docker Hub, GHCR, etc.)
2. SSH al servidor
3. `docker pull <tu-imagen>`
4. `docker run -d -p 7860:7860 <tu-imagen>`

Opcionalmente configurar **Nginx** como reverse proxy con SSL (Let's Encrypt).

---

## Arquitecturas Implementadas

Se comparan **5 arquitecturas** representando diferentes paradigmas:

| # | Modelo | Paradigma | Parámetros | Justificación |
|---|--------|-----------|------------|---------------|
| 1 | U-Net + ResNet50 | CNN clásica | 32.5M | Baseline biomédico estándar |
| 2 | U-Net + EfficientNet-B4 | CNN eficiente | 20.2M | Mejor para deploy en tablet/edge (compound scaling) |
| 3 | DeepLabV3+ + ResNet50 | CNN multi-escala | 26.7M | ASPP captura vértebras de tamaño variable |
| 4 | U-Net + MiT-B3 | Transformer (SegFormer) | 47.4M | Self-attention global para manejar oclusión vertebral |
| 5 | MAnet + MiT-B5 | Transformer + atención dual | 92.2M | Encoder transformer + decoder con Position Attention Module |

### ¿Por qué transformers?
En escoliosis severa, las vértebras rotan axialmente causando **superposición parcial** en la radiografía 2D. Las CNNs puras tienen campo receptivo local (ventanas pequeñas). Los transformers con self-attention pueden ver relaciones globales: si T7 está oculta detrás de T6, el modelo usa el contexto de T5 y T8 para inferir dónde debe estar T7.

---

## Resultados Obtenidos

### Segmentación multiclase (24 clases de vértebras) — Test Set

| Ranking | Modelo | Paradigma | Test Dice | Test IoU | PixAcc | Parámetros |
|---------|--------|-----------|-----------|----------|--------|------------|
| 🥇 | **DeepLabV3+ + ResNet50** | CNN multi-escala (ASPP) | **0.3378** | **0.2556** | **0.9596** | 26.7M |
| 🥈 | MAnet + MiT-B5 | Transformer + atención dual | 0.3271 | 0.2383 | 0.9594 | 92.2M |
| 🥉 | U-Net + MiT-B3 | Transformer (SegFormer) | 0.3157 | 0.2323 | 0.9578 | 47.4M |
| 4° | U-Net + ResNet50 | CNN clásica | 0.2691 | 0.1883 | 0.9541 | 32.5M |
| 5° | U-Net + EfficientNet-B4 | CNN eficiente (tablet-ready) | 0.2189 | 0.1542 | 0.9548 | 20.2M |

### Hallazgo importante
**DeepLabV3+ superó a los transformers**. Su módulo ASPP (convoluciones atrous multi-escala) captura contexto a diferentes zoom en paralelo — exactamente lo que se necesita para vértebras que varían mucho en tamaño (cervicales pequeñas vs lumbares grandes). Este resultado es contrario a la hipótesis inicial de que los transformers ganarían por su self-attention global.

### Ángulo de Cobb (evaluación en casos de escoliosis)

| Modelo | Método | MAE (°) | Correlación Pearson |
|--------|--------|---------|---------------------|
| U-Net + EfficientNet-B4 | Binario (skeleton) | **23.0** | **0.66** |
| U-Net + ResNet50 | Binario (skeleton) | 25.5 | 0.56 |
| U-Net + MiT-B3 | Multiclase (endplate) | 28.2 | 0.27 |
| U-Net + EfficientNet-B4 | Multiclase (endplate) | 26.8 | 0.20 |
| U-Net + ResNet50 | Multiclase (endplate) | 39.4 | -0.12 |

**Observación**: el método binario (basado en esqueletización) da mejor MAE de Cobb que el multiclase, probablemente porque errores en la identificación de vértebras individuales se acumulan al calcular el ángulo desde placas vertebrales.

---

## Problemas Conocidos del Dominio

### 1. Rotación vertebral en escoliosis severa
Las vértebras rotan axialmente, causando superposición en la radiografía 2D. Los transformers ayudan a mitigar esto con self-attention global.

### 2. Casos atípicos (ej: paciente con una sola vértebra visible)
Se mantienen en el dataset como casos clínicos reales. Se documentan como limitaciones en el análisis final.

### 3. Desbalance extremo de clases
- Background: 95.9% de píxeles
- Cervicales (C3-C5): 0.0006% - 0.01% de píxeles
- Mitigación: Weighted Cross-Entropy + Generalized Dice Loss con pesos por frecuencia inversa

### 4. Imágenes de tamaño variable
~259×971 a ~381×1074. Se preserva aspect ratio con resize + padding a 512×512, interpolación nearest-neighbor para máscaras (no corromper IDs de clase).

### 5. Ángulos de Cobb en el límite
Algunos valores ground truth = ~90° exactos (límite del arctan). Se documenta como limitación.

---

## Flujo de Trabajo del Equipo

### Quien tiene GPU
1. Hacer `git pull`
2. Entrenar modelos nuevos o re-entrenar con cambios
3. Guardar `checkpoints/*.pth` y compartir con el equipo (Google Drive, Dropbox, etc.)
4. Hacer `git commit` y `git push` con los cambios de código
5. Actualizar `AGENTS.md` con nuevas métricas y decisiones

### Quien no tiene GPU
1. Hacer `git pull`
2. Descargar `checkpoints/` desde la ubicación compartida
3. Usar el notebook `02_training_experiments.ipynb` para evaluar
4. Usar la app Gradio para demos
5. Contribuir: análisis de resultados, documentación, escritura del artículo

### Metodología Spec-Driven
El proyecto sigue los 6 pilares del trabajo spec-driven:
- **Colaboración sobre delegación** — humanos y herramientas trabajando juntos
- **Procesos trazables** — todas las decisiones en `AGENTS.md` con razón
- **Orquestación y handoff** — entre personas y herramientas
- **Poder del spec** — `AGENTS.md` es la interfaz compartida
- **Spec-driven + test-driven** — verificaciones en cada ciclo
- **Artefacto final como input del siguiente ciclo** — cada ciclo actualiza el AGENTS.md

---

## Licencia y Uso

Proyecto académico de la Universidad de los Andes. El dataset MaIA Scoliosis es propiedad de la Universidad de los Andes y del grupo de Ingeniería Biomédica. El código de este repositorio puede ser usado con fines académicos citando la fuente.

**Disclaimer clínico:** Esta herramienta es de APOYO al diagnóstico. NO reemplaza el criterio del profesional de la salud. Toda decisión clínica debe ser validada por un especialista calificado.

---

## Contacto

**Autor:** Elvis Hernández
**Email:** elvis.hernandez@en-firme.com
**Universidad:** Universidad de los Andes — Maestría en Inteligencia Artificial

**Repositorio:** https://github.com/ElvLandau117/Segmentacion-Soluciones
