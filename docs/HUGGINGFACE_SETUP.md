# Hugging Face Hub — Setup y gestión de pesos

> **Decidido en Ciclo 4 (2026-05-17):** los pesos de los modelos viven en
> Hugging Face Hub, no en OneDrive ni en git. El container los descarga al boot.

## 1. Por qué Hugging Face Hub

| Razón | Detalle |
|-------|---------|
| **Estándar industria ML** | Es donde se publican modelos open source (BERT, Llama, Stable Diffusion, etc.). Cero fricción para evaluadores familiarizados. |
| **Gratis** | Repos de modelos públicos o privados, sin costo. Los servicios pagos son Inference Endpoints y Spaces con GPU — no los usamos. |
| **Versionado tipo git** | Cada subida es un commit. Puedes rollback a un peso anterior. |
| **Soporta archivos grandes** | LFS automático para `.pth`, `.safetensors`, etc. Hasta 50 GB por archivo en el plan gratuito. |
| **Cambio de pesos sin re-deploy** | Subes nuevo commit a HF → reinicias container → carga el nuevo peso. **Cero cambio en el repo de código.** |

## 2. Estructura: dos repos separados

```
GitHub: ElvLandau117/Segmentacion-Soluciones        ← código (este repo)
            │
            │  En runtime, dentro del container:
            │  snapshot_download("elvis/spine-checkpoints")
            ▼
HF Hub:  elvis/spine-checkpoints                     ← solo los .pth
            ├── deeplabv3plus_resnet50_multiclass_best.pth   (~102 MB)
            └── unet_resnet50_binary_best.pth                 (~100 MB)
```

El repo de HF NO contiene código. El container instala el código del repo de
GitHub (vía `git clone` o capa de Docker) y descarga los pesos del repo de HF.

## 3. Setup inicial (una sola vez)

### 3.1 Crear cuenta en Hugging Face

1. Ir a https://huggingface.co/join
2. Crear cuenta gratis con email (o GitHub).
3. Recordar tu **username** — será parte del repo ID (ej: `elvis/spine-checkpoints`).

### 3.2 Crear un Access Token

1. Ir a https://huggingface.co/settings/tokens
2. Click en **"New token"**
3. Nombre: `spine-deploy`
4. Tipo: **Write** (necesario para subir pesos)
5. Copiar el token (empieza con `hf_...`). **Solo se ve una vez** — si lo pierdes, generas uno nuevo.

### 3.3 Autenticarse localmente

```powershell
pip install huggingface_hub
huggingface-cli login
# Pega el token cuando lo pida
```

El token queda guardado en `~/.cache/huggingface/token` y todas las llamadas
posteriores lo usan automáticamente.

## 4. Subir los pesos del proyecto (primera vez)

Asumiendo que estás en la raíz del proyecto y los `.pth` están en `checkpoints/`:

```powershell
# Reemplaza <tu-username> por tu username de HF
$env:HF_REPO_ID = "<tu-username>/spine-checkpoints"

# Subir el modelo ganador (DeepLabV3+ multiclase, ~102 MB)
python scripts/upload_weights.py --file checkpoints/deeplabv3plus_resnet50_multiclass_best.pth

# Subir el modelo binario (para tab Binary y cálculo de Cobb por skeleton)
python scripts/upload_weights.py --file checkpoints/unet_resnet50_binary_best.pth
```

El script:
1. Crea el repo `<tu-username>/spine-checkpoints` si no existe (tipo "model", público).
2. Sube el `.pth` con LFS automático.
3. Imprime la URL del archivo subido.

Para repos privados: agregar `--private` en la primera subida.

## 5. Re-entrenamiento: subir un peso actualizado

Cuando re-entrenes (ej: con más imágenes, hiperparams diferentes):

```powershell
python scripts/upload_weights.py \
    --file checkpoints/deeplabv3plus_resnet50_multiclass_best.pth \
    --commit-message "v2: trained with augmentation pipeline X"
```

Reinicia el container en Hetzner:

```bash
docker compose restart app
```

Al arrancar, el container detecta que su caché está desactualizada (commit hash
diferente en HF Hub) y descarga el nuevo `.pth`. **Cero cambio en el código.**

## 6. Cómo el container descarga los pesos

En `spine_segmentation/deployment/inference.py` (post-Unidad 4.4):

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    cache_dir="/data/checkpoints",  # volumen persistente
    token=os.getenv("HF_TOKEN"),     # solo necesario si el repo es privado
)
```

- **Primera vez:** baja los .pth (~200 MB total). Tarda 30-60 s según ancho de banda.
- **Reinicios subsiguientes:** verifica el commit hash. Si no cambió, no descarga.
- **Caché:** vive en el volumen Docker `checkpoints_cache`. Sobrevive a `docker compose down/up`.

## 7. Variables de entorno relacionadas

| Variable | Default | Descripción |
|----------|---------|-------------|
| `HF_REPO_ID` | (requerida) | ID del repo, ej: `elvis/spine-checkpoints` |
| `HF_TOKEN` | (vacío) | Solo necesario si el repo es privado |
| `HF_HOME` | `/data/checkpoints` | Directorio de caché dentro del container |

Documentadas en `.env.example` (Unidad 4.3).

## 8. Troubleshooting

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| `RepositoryNotFoundError` al descargar | Repo privado sin `HF_TOKEN` | Generar token de lectura y exportarlo |
| Sube muy lento | LFS sin configurar | `huggingface_hub` lo maneja, pero asegúrate de no estar tras un proxy corporativo |
| `403 Forbidden` al subir | Token sin permiso Write | Re-generar token con permiso correcto |
| El container redescarga cada vez | Volumen no persistente | Verificar que `docker-compose.yml` monta `checkpoints_cache` correctamente |
