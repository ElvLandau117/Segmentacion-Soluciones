---
title: Spine Segmentation for Scoliosis
emoji: 🦴
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Vertebrae segmentation + Cobb angle for scoliosis
suggested_hardware: cpu-basic
---

# Segmentación Automática de Columna Vertebral y Vértebras

**Proyecto Final — Maestría en Inteligencia Artificial · Universidad de los Andes**

> Este repositorio funciona como código fuente en GitHub **y** como un Space
> ejecutable en Hugging Face. El bloque YAML de arriba es la metadata del Space
> (lo lee HF Hub; GitHub lo renderiza como una tabla discreta).

Sistema de deep learning para segmentar la columna vertebral y vértebras
individuales en radiografías AP, calcular el ángulo de Cobb automáticamente y
explicar cada predicción con Grad-CAM + mapas de confianza. Diseñado como
apoyo al radiólogo, no como reemplazo.

---

## 🔗 Enlaces rápidos

| Recurso | URL |
|---------|-----|
| **App desplegada (oficial)** | `https://huggingface.co/spaces/ElvLandau/spine-segmentation` — _por completar tras crear el Space; ver [`docs/HF_SPACES_SETUP.md`](docs/HF_SPACES_SETUP.md)_ |
| **Repositorio GitHub** | https://github.com/ElvLandau117/Segmentacion-Soluciones |
| **Pesos del modelo** | `https://huggingface.co/ElvLandau/spine-checkpoints` — ver [`docs/HUGGINGFACE_SETUP.md`](docs/HUGGINGFACE_SETUP.md) |
| **Dataset** | MaIA Scoliosis (propiedad U. Andes — solicitarlo al líder del proyecto) |
| **Rúbrica oficial** | [`requisitos_universidad/`](requisitos_universidad/) |
| **Memoria del proyecto** | [`AGENTS.md`](AGENTS.md) |
| **Despliegue alternativo (Hetzner / VPS)** | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |

---

## 1. Problema y solución

### El problema
La escoliosis es una deformidad tridimensional de la columna vertebral que
afecta al 2-3 % de la población. El estándar diagnóstico es el **ángulo de
Cobb** medido manualmente sobre radiografías AP, un proceso que es:

- **Subjetivo** — variabilidad inter-observador documentada de 5-10°.
- **Lento** — 5-10 min por radiografía.
- **Dependiente de experiencia** del radiólogo.

### La solución
Una aplicación web (Gradio) que recibe una radiografía y devuelve:

1. **Segmentación binaria** de la columna completa.
2. **Segmentación multiclase** de las 22 vértebras individuales (C3 a L5).
3. **Ángulo de Cobb automático** por dos métodos (binario por esqueleto y
   multiclase por placas vertebrales).
4. **Panel de explicabilidad** con Grad-CAM (qué regiones miró el modelo) y
   mapa de confianza pixel-por-pixel (verde = seguro / rojo = revisar).
5. **Reporte clínico** con disclaimer médico explícito.

Todo corre en CPU (no requiere GPU para inferencia) y se despliega como un
único `docker compose up`.

---

## 2. Arquitectura del despliegue

### Oficial — Hugging Face Spaces (gratis, sin servidor propio)

```
USUARIO (medico, jurado, compañero)
   │ HTTPS gestionado por HF
   ▼
huggingface.co/spaces/ElvLandau/spine-segmentation
   │ runtime: Gradio SDK, CPU Basic gratis (2 vCPU, 16 GB RAM)
   │ entrypoint: app.py (expone `demo`)
   ▼
APP (Gradio + DeepLabV3+ + UNet binario)
   │ boot: ensure_weights() -> snapshot_download desde HF Hub
   ▼
huggingface.co/ElvLandau/spine-checkpoints    ← repo de pesos (mismo HF)
```

- **Modelo en producción:** `DeepLabV3+ ResNet50` (ganador del Ciclo 3 con
  Dice = 0.3378) para vértebras + `UNet ResNet50` binario para Cobb.
- **Pesos:** repo separado en HF Hub. Se descargan al primer arranque del
  Space y se cachean en su storage persistente (50 GB gratis).
- **HTTPS:** automático y transparente, gestionado por HF.
- **URL:** `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- **Sleep:** tras 48 h sin uso el Space se duerme; despierta en 30-60 s al
  primer click. Para evaluación: abrir la URL 1 minuto antes para calentarlo.

### Alternativa — Docker + Caddy en VPS (Hetzner)

Para producción real o si se quiere evitar el sleep:

```
USUARIO ──HTTPS──▶ CADDY (Let's Encrypt) ──HTTP──▶ APP container
                                                       │
                                                       ▼
                                              /data/checkpoints
                                              (cache de HF Hub)
```

Stack en `docker-compose.yml` + `Caddyfile`. Runbook en
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## 3. Instrucciones básicas de uso (usuario final)

1. Abrir la **URL pública** (ver sección Enlaces rápidos).
2. Cargar una radiografía AP de columna completa (formato PNG/JPG, hasta 25 MB).
3. Click en **Analyze**.
4. Esperar ~3-8 segundos (inferencia en CPU).
5. Navegar las pestañas:
   - **Binary Segmentation** — overlay verde de la columna.
   - **Vertebrae Segmentation** — vértebras coloreadas (C3 rojo → L5 azul).
   - **Cobb Angle** — visualización con líneas y ángulo numérico.
   - **Explainability** — Grad-CAM (izquierda) + mapa de confianza (derecha).
6. Leer el bloque **Diagnosis Results** y, sobre todo, el **disclaimer médico**.

---

## 4. Ejemplos de input/output esperados

### Input
Una radiografía AP de columna completa. Ejemplos válidos:

| Característica | Valor |
|----------------|-------|
| Formato | PNG, JPG, JPEG |
| Tamaño típico | 259×971 a 381×1074 px (se redimensiona internamente a 512×512) |
| Color | grayscale o RGB (se convierte automáticamente) |
| Tamaño máximo | 25 MB |

### Output (lo que devuelve la app)

| Campo | Tipo | Ejemplo |
|-------|------|---------|
| Binary overlay | imagen | columna en verde semitransparente |
| Multiclass overlay | imagen | 22 vértebras coloreadas individualmente |
| Cobb visualization | imagen | radiografía con puntos de inflexión + ángulo dibujado |
| Cobb Angle (Binary) | número (°) | `28.4 degrees` |
| Cobb Angle (Multiclass) | número + vértebras | `26.1 degrees (T7-L1)` |
| Vertebrae detected | lista | `[C7, T1, T2, ..., L5]` |
| Assessment | texto | `Moderate scoliosis (25-40 degrees)` |
| Grad-CAM heatmap | imagen | mapa de calor de regiones influyentes |
| Confidence map | imagen | verde = alta confianza / rojo = baja |

### Notas sobre interpretación clínica del Cobb

| Ángulo | Interpretación |
|--------|----------------|
| < 10° | Normal (no es escoliosis) |
| 10-25° | Escoliosis leve |
| 25-40° | Escoliosis moderada |
| > 40° | Escoliosis severa (suele requerir intervención) |

---

## 5. Cómo reproducir el despliegue

### Opción A — Hugging Face Spaces (oficial y recomendada)

Esta es la URL que entregamos al jurado. **Gratis**, sin servidor propio, HTTPS
gestionado por HF, integrado con el repo de pesos.

```bash
# 1. Crear cuenta + token en https://huggingface.co (5 min)
# 2. Subir los pesos al repo de HF Hub (una sola vez):
huggingface-cli login
python scripts/upload_weights.py --file checkpoints/deeplabv3plus_resnet50_multiclass_best.pth
python scripts/upload_weights.py --file checkpoints/unet_resnet50_binary_best.pth

# 3. Crear el Space en https://huggingface.co/new-space
#    SDK: Gradio  |  Hardware: CPU Basic (free)
# 4. Push del codigo al Space:
git remote add hf https://huggingface.co/spaces/ElvLandau/spine-segmentation
git push hf main

# 5. En Settings del Space -> Variables: setear HF_REPO_ID=ElvLandau/spine-checkpoints
```

HF construye el Space, descarga los pesos y la URL queda en
`https://huggingface.co/spaces/ElvLandau/spine-segmentation`.

Runbook detallado paso a paso: [`docs/HF_SPACES_SETUP.md`](docs/HF_SPACES_SETUP.md).
Cualquier evaluador puede **duplicar el Space en 1 click** ("Duplicate Space" en
HF) para reproducirlo bajo su propia cuenta.

### Opción B — Hetzner / VPS propio (alternativa avanzada)

Si en algún momento queremos evitar el "sleep" de HF Spaces o tener control
total del servidor, dejamos preparado un stack `docker compose` con Caddy
para SSL automático.

```bash
git clone https://github.com/ElvLandau117/Segmentacion-Soluciones.git
cd Segmentacion-Soluciones
cp .env.example .env
nano .env                                   # HF_REPO_ID + DOMAIN
bash scripts/deploy_hetzner.sh
```

Runbook completo: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Documentos relacionados

- [`docs/HUGGINGFACE_SETUP.md`](docs/HUGGINGFACE_SETUP.md) — crear cuenta HF,
  subir los `.pth`, intercambiar pesos sin re-deploy.
- [`docs/HF_SPACES_SETUP.md`](docs/HF_SPACES_SETUP.md) — todo el flujo del Space
  paso a paso.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — alternativa Hetzner.

---

## 6. Parametrización (env vars)

Toda la configuración del despliegue se controla con variables de entorno
(no hay que tocar código). Ver `.env.example` para la plantilla.

| Variable | Default | Propósito |
|----------|---------|-----------|
| `APP_HOST` | `0.0.0.0` | Host interno de Gradio |
| `APP_PORT` | `7860` | Puerto interno de Gradio (Caddy hace proxy) |
| `DOMAIN` | `replace-me.nip.io` | Hostname público; Caddy obtiene SSL de Let's Encrypt para él |
| `HF_REPO_ID` | (vacío) | Repo en HF Hub que contiene los `.pth` (ej: `elvis/spine-checkpoints`) |
| `HF_TOKEN` | (vacío) | Solo necesario si el repo de HF es privado |
| `CHECKPOINTS_DIR` | `/data/checkpoints` (en container) | Dónde se cachean los pesos descargados |
| `DEFAULT_MULTICLASS_MODEL` | `deeplabv3plus_resnet50` | Arquitectura multiclase a cargar |
| `DEFAULT_BINARY_MODEL` | `unet_resnet50` | Arquitectura binaria a cargar |
| `DEFAULT_MULTICLASS_WEIGHT` | `<model>_multiclass_best.pth` | Nombre del `.pth` en HF |
| `DEFAULT_BINARY_WEIGHT` | `<model>_binary_best.pth` | Nombre del `.pth` en HF |
| `INFERENCE_IMAGE_SIZE` | `512` | Resize target (coincide con entrenamiento) |
| `MEDICAL_DISCLAIMER` | _texto en inglés_ | Sobrescribible para cambiar el wording sin tocar código |

> **Credenciales:** la app es pública y no requiere login. La única
> credencial es `HF_TOKEN` para repos privados de HF Hub; documentada arriba.

---

## 7. Dependencias y entorno

| Dependencia | Versión mínima | Propósito |
|-------------|----------------|-----------|
| **Python** | 3.11 | Runtime principal |
| **Docker** | 24+ con Compose v2 | Despliegue (recomendado) |
| **PyTorch CPU** | 2.2 | Inferencia (no requiere CUDA) |
| **gradio** | 4.15 | UI web |
| **segmentation-models-pytorch** | 0.3.3 | Familia de modelos de segmentación |
| **huggingface_hub** | 0.20 | Descarga de pesos en runtime |
| **grad-cam** | 1.5 | Explicabilidad (Grad-CAM) |

Lista completa en [`requirements.txt`](requirements.txt).

### Instalación local (desarrollo)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Configurar variables
cp .env.example .env
# Editar .env: poner HF_REPO_ID a tu repo de HF

# Arrancar la app
python -m app.main
# o:  python app/main.py
```

---

## 8. Consideraciones sobre el modelo

| Aspecto | Valor |
|---------|-------|
| **Arquitectura en producción** | DeepLabV3+ ResNet50 (multiclase) + UNet ResNet50 (binario) |
| **Tamaño total de pesos** | ~ 200 MB (los dos `.pth` combinados) |
| **Tamaño imagen Docker** | ~ 3 GB (PyTorch CPU + deps; pesos NO incluidos en la imagen) |
| **RAM en runtime** | ~ 1.5-2 GB con un modelo cargado |
| **Hardware mínimo recomendado** | 2 vCPU + 4 GB RAM + 10 GB disco |
| **Latencia de inferencia (CPU)** | 3-8 s por imagen (depende del tier del server — medido tras el deploy) |
| **Latencia primer boot** | 30-60 s adicionales para descargar pesos de HF (una sola vez) |
| **Soporta GPU** | Sí (detección automática), pero el deploy default es CPU para portabilidad |

### ¿Por qué solo se sirve un modelo y no los 5 entrenados?
DeepLabV3+ ganó la comparativa del Ciclo 3 con margen sólido. Servir los 5
en producción significaría cargar ~838 MB en RAM y un arranque más lento sin
beneficio para el usuario final. Los otros 4 viven en el repo de HF Hub y en
el notebook `02_training_experiments.ipynb` para análisis comparativo.

### ¿Cómo se actualizan los pesos sin re-deploy?
1. Re-entrenas y obtienes un nuevo `.pth`.
2. `python scripts/upload_weights.py --file <nuevo.pth>` (sube a HF Hub).
3. En el server: `docker compose restart app`.
4. La app baja el nuevo peso al arrancar y empieza a servirlo.

**Cero cambios en el repo de código.**

---

## 9. Resultados (Ciclo 3 — entrenamiento)

### Segmentación multiclase (24 clases)

| Ranking | Modelo | Paradigma | Test Dice | Test IoU | Parámetros |
|:-:|---|---|:-:|:-:|:-:|
| 🥇 | **DeepLabV3+ ResNet50** | CNN multi-escala (ASPP) | **0.3378** | **0.2556** | 26.7 M |
| 🥈 | MAnet + MiT-B5 | Transformer + atención dual | 0.3271 | 0.2383 | 92.2 M |
| 🥉 | U-Net + MiT-B3 | Transformer (SegFormer) | 0.3157 | 0.2323 | 47.4 M |
| 4° | U-Net + ResNet50 | CNN clásica | 0.2691 | 0.1883 | 32.5 M |
| 5° | U-Net + EfficientNet-B4 | CNN eficiente | 0.2189 | 0.1542 | 20.2 M |

### Ángulo de Cobb (en casos de escoliosis del test set)

| Método | Mejor modelo | MAE (°) | Correlación Pearson |
|--------|--------------|:-:|:-:|
| Binario (esqueleto) | U-Net + EfficientNet-B4 | **23.0** | **0.66** |
| Multiclase (endplate) | U-Net + EfficientNet-B4 | 26.8 | 0.20 |

**Detalle completo:** [`docs/CICLO_3_ARTEFACTOS.md`](docs/CICLO_3_ARTEFACTOS.md).

> La visualización del ángulo de Cobb en la app (cajas verdes en las vértebras
> de inicio/fin, perpendiculares al endplate, arco del ángulo, mini "Cobb-meter"
> para ángulos pequeños) está inspirada en Fig 1 de **Shi et al. 2025**,
> "Accurate Cobb Angle Estimation via SVD-Based Curve Detection and Vertebral
> Wedging Quantification" (IEEE Journal of Biomedical and Health Informatics,
> [arXiv:2509.24898](https://arxiv.org/abs/2509.24898)). El paper usa landmarks
> anotados por endplate; nuestro pipeline usa la segmentación multiclase como
> sustituto, así que el modelo subyacente no es el mismo — solo la presentación
> visual y el principio de fusionar dos mediciones para la decisión clínica.

---

## 10. Estructura del repositorio

```
.
├── app/                          # Entrypoint estándar (shim al paquete)
│   ├── __init__.py
│   └── main.py
├── spine_segmentation/            # Paquete Python principal
│   ├── config.py                  # Hiperparámetros + env vars de deploy
│   ├── data/                      # Dataset, transforms, splits, class mapping
│   ├── models/                    # SMP factory + losses
│   ├── training/                  # Trainer con MLflow + AMP
│   ├── evaluation/                # Métricas, Cobb, explainability, viz
│   ├── postprocessing/            # Morfología, ordering anatómico
│   └── deployment/                # app.py + inference.py + weights.py
├── scripts/
│   ├── train_multiclass.py        # Entrenar (requiere GPU)
│   ├── evaluate.py                # Evaluación comparativa
│   ├── compute_cobb_angles.py
│   ├── upload_weights.py          # Sube .pth a HF Hub
│   ├── deploy_hetzner.sh          # Deploy reproducible
│   └── reorganize_root_files.ps1  # Limpieza local (una sola vez)
├── tests/                         # Pytest (13 tests)
├── notebooks/                     # EDA, training, informe final
├── docs/                          # Runbooks + artefactos por ciclo
│   ├── DEPLOYMENT.md
│   ├── HUGGINGFACE_SETUP.md
│   ├── RUTA_LECTURA.md
│   ├── CICLO_3_ARTEFACTOS.md
│   ├── CICLO_4_DESPLIEGUE_BRIEF.md
│   └── metodologia/               # Papers de referencia (no en git)
├── requisitos_universidad/        # PDFs oficiales de Coursera / U. Andes
├── archive/                       # Artefactos legacy (no en git)
├── Dockerfile
├── docker-compose.yml             # app + caddy + volúmenes
├── Caddyfile                      # reverse proxy + SSL automático
├── .env.example                   # Plantilla de variables de entorno
├── requirements.txt
├── pytest.ini
├── README.md                      # Este archivo
├── AGENTS.md                      # Memoria persistente del proyecto
├── CLAUDE.md                      # Pointer para Claude Code → AGENTS.md
└── WORKFLOW.md                    # Reglas del repositorio
```

---

## 11. Pruebas

```bash
pytest tests/ -v
```

Cubre: parametrización por env vars (rúbrica 15 %), resolución de pesos vía
HF Hub, construcción de la app Gradio, presencia del disclaimer médico en la
UI, contrato del pipeline de inferencia.

Los tests que requieren `.pth` reales están marcados `requires_checkpoints`
y se saltean automáticamente si no hay pesos locales.

---

## 12. Equipo y metodología

### Autores
- **Elvis Hernández** — autor principal · `elvis.hernandez@en-firme.com`

### Institución
Maestría en Inteligencia Artificial — Universidad de los Andes, Bogotá.

### Metodología
El proyecto sigue **Spec-Driven Work + Work Orchestration** (Leonardo
Gonzalez 2025-2026):

- Cada ciclo cierra con un artefacto en `docs/CICLO_N_ARTEFACTOS.md`.
- `AGENTS.md` es la memoria persistente — todo cambio importante se
  registra ahí con su razón.
- Reglas no negociables del repo en [`WORKFLOW.md`](WORKFLOW.md).
- Las unidades de trabajo son < 8 h y cada una cierra con un commit
  verificable.

---

## 13. Aviso médico

> Esta herramienta es **de apoyo al diagnóstico**. NO reemplaza el criterio
> del profesional de la salud. Toda decisión clínica debe ser validada por
> un radiólogo o especialista calificado. El sistema fue entrenado con un
> dataset académico limitado y no está aprobado por ninguna autoridad
> regulatoria.

---

## 14. Licencia

Proyecto académico. El dataset MaIA Scoliosis es propiedad de la Universidad
de los Andes (no se redistribuye con este repositorio). El código de este
repositorio puede usarse con fines académicos citando la fuente.
