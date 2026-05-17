# Hugging Face Spaces — Despliegue oficial (Ciclo 4)

> Esta guía cubre el despliegue **oficial** del proyecto: el Space en Hugging Face.
> Para la alternativa Hetzner/VPS, ver [`DEPLOYMENT.md`](DEPLOYMENT.md).

## 0. Por qué HF Spaces

| Razón | Detalle |
|-------|---------|
| **Gratis** | CPU Basic: 2 vCPU + 16 GB RAM + 50 GB storage. Cero costo recurrente. |
| **Setup trivial** | Sin DNS, sin SSL, sin servidor. Crear Space → `git push` → URL pública. |
| **Integración nativa** | Pesos en HF Hub + Space en HF: una sola plataforma, un solo login. |
| **URL profesional** | `huggingface.co/spaces/<user>/spine-segmentation` con HTTPS automático. |
| **Reproducible para jurados** | Botón "Duplicate Space" → cualquiera lo clona bajo su cuenta en 1 click. |
| **Sleep aceptable** | Tras 48 h sin uso se duerme; despierta en 30-60 s al primer click. |

## 1. Pre-requisitos (una sola vez)

### 1.1 Cuenta + token
1. **Crear cuenta** en https://huggingface.co/join (gratis, 5 min).
2. **Generar token Write** en https://huggingface.co/settings/tokens:
   - Click "New token" → nombre: `spine-deploy` → tipo: **Write** → Create.
   - Copiar el token (`hf_...`). **Solo se muestra una vez.**
3. **Autenticarse en local:**
   ```powershell
   pip install huggingface_hub
   huggingface-cli login
   # Pegar el token cuando lo pida
   ```

### 1.2 Subir los pesos al HF Hub
Antes del Space necesitamos el repo de pesos. Ver
[`HUGGINGFACE_SETUP.md`](HUGGINGFACE_SETUP.md) para el detalle, en resumen:

```powershell
# Reemplaza <usuario> por tu username de HF
python scripts/upload_weights.py `
    --repo <usuario>/spine-checkpoints `
    --file checkpoints/deeplabv3plus_resnet50_multiclass_best.pth

python scripts/upload_weights.py `
    --repo <usuario>/spine-checkpoints `
    --file checkpoints/unet_resnet50_binary_best.pth
```

Confirma que aparecen en `https://huggingface.co/<usuario>/spine-checkpoints`.

## 2. Crear el Space

1. Ir a https://huggingface.co/new-space
2. Llenar el formulario:
   - **Owner:** tu usuario (o una organización si la tienes).
   - **Space name:** `spine-segmentation` (el nombre que verá la URL).
   - **License:** `apache-2.0` (coincide con el YAML del README).
   - **Select the Space SDK:** **Gradio** (la app actual ya es Gradio).
   - **Space hardware:** `CPU basic - 2 vCPU - 16GB - FREE` (suficiente).
   - **Public/Private:** **Public** (para que jurados accedan sin login).
3. Click **Create Space**.

Se crea un repo git vacío en `https://huggingface.co/spaces/<usuario>/spine-segmentation`.

## 3. Configurar variables y secrets del Space

El Space necesita saber dónde están los pesos. En la página del Space:

1. Click en **Settings** (arriba a la derecha).
2. Sección **Variables and secrets** → **New variable**:
   - Name: `HF_REPO_ID`
   - Value: `<usuario>/spine-checkpoints`
3. Si el repo de pesos es **privado** (no recomendado para este proyecto),
   añadir también un **secret**:
   - Name: `HF_TOKEN`
   - Value: el token con permiso Read (puede ser distinto al de Write).

Variables = visibles en logs. Secrets = enmascarados. Para `HF_TOKEN` siempre usa secret.

> Otras variables que el `config.py` respeta (default suelen estar bien, raramente necesitas tocarlas en el Space): `DEFAULT_MULTICLASS_MODEL`, `DEFAULT_BINARY_MODEL`, `DEFAULT_MULTICLASS_WEIGHT`, `DEFAULT_BINARY_WEIGHT`, `INFERENCE_IMAGE_SIZE`, `MEDICAL_DISCLAIMER`.

## 4. Conectar el repo de GitHub con el Space

Hay 2 caminos. El más simple para empezar:

### Camino A — git remote dual (recomendado)

```bash
# En tu clon local del repo de GitHub:
git remote add hf https://huggingface.co/spaces/<usuario>/spine-segmentation
git push hf main
```

La primera vez te pedirá credenciales. Usa tu usuario HF y como password el token Write (el mismo de `huggingface-cli login`).

Cada vez que hagas `git push hf main`, el Space:
1. Recibe el código.
2. Lee el YAML front-matter del `README.md` (sdk, app_file, etc.).
3. Construye el entorno (`pip install -r requirements.txt`).
4. Ejecuta `python app.py`, detecta el `demo` y lo expone.
5. La URL queda accesible en 2-5 min (primer build).

### Camino B — Sync from GitHub (avanzado)

Para automatizar pushes desde GitHub a HF Spaces con GitHub Actions. Útil
si trabajas con un equipo y todos hacen push a GitHub. Documentación:
https://huggingface.co/docs/hub/spaces-github-actions

Para el alcance del Ciclo 4, **camino A es suficiente**.

## 5. Primer arranque

1. Abrir https://huggingface.co/spaces/<usuario>/spine-segmentation
2. Ver la pestaña **Logs** del Space: debería verse el build pip + la descarga
   de pesos (`[weights] Downloading deeplabv3plus_resnet50_multiclass_best.pth ...`).
3. Cuando aparece `Running on local URL: http://0.0.0.0:7860`, la app está lista.
4. Volver a la URL principal: la UI Gradio aparece con sus 4 tabs.

**Tiempo total primer arranque:** 3-8 min (pip install ~2 min + HF download ~1 min + Gradio startup ~30 s).

**Reinicio posterior:** 30-60 s (pesos ya cacheados en storage del Space).

## 6. Acceptance checklist (post-deploy)

| Check | Cómo verificar |
|-------|----------------|
| Space está running | Badge verde "Running" arriba a la izquierda |
| Pesos descargados | Logs muestran `[weights] -> /data/checkpoints/<archivo>.pth` |
| UI carga | Las 4 tabs (Binary, Vertebrae, Cobb, Explainability) son visibles |
| Disclaimer médico visible | Scroll abajo, "Aviso medico / Medical disclaimer" al pie |
| Inferencia funciona | Subir una radiografía → Cobb numérico aparece en 3-10 s |
| HTTPS | URL empieza con `https://`, candado verde |
| Reproducibilidad | "Duplicate Space" en tu URL clona el Space bajo otra cuenta |

## 7. Operaciones comunes

| Acción | Cómo |
|--------|------|
| Ver logs en vivo | Settings → Logs (o la pestaña "Logs" en la página principal) |
| Reiniciar el Space | Settings → Factory reboot |
| Actualizar el código | `git push hf main` (build automático) |
| Actualizar un peso | Subir nuevo `.pth` a `<usuario>/spine-checkpoints` con `scripts/upload_weights.py` + Factory reboot del Space (re-descarga el nuevo) |
| Cambiar `HF_REPO_ID` (apuntar a otro repo de pesos) | Settings → Variables → editar → Factory reboot |
| Cambiar visibilidad (Public ↔ Private) | Settings → Visibility |
| Upgradear hardware (si la inferencia es muy lenta) | Settings → Hardware → CPU Upgrade (~$9/mes) o GPU T4 (~$0.6/h) |
| Despertar antes de la evaluación | Abrir la URL 1 minuto antes de la sesión, esperar a que muestre la UI |

## 8. Troubleshooting

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| Build falla con `ModuleNotFoundError` | Falta dependencia en `requirements.txt` | Añadir el paquete y `git push hf main` |
| `[weights] not found ... HF_REPO_ID is not set` en logs | Variable no configurada en el Space | Settings → Variables → añadir `HF_REPO_ID` + Factory reboot |
| `RepositoryNotFoundError` al descargar pesos | Repo privado sin `HF_TOKEN` | Añadir `HF_TOKEN` como **secret** + Factory reboot |
| App tarda > 30 s en cada inferencia | CPU Basic justa con DeepLabV3+ | Aceptar (es lo esperado) o upgrade a CPU Upgrade |
| El Space duerme y la demo es a las 9:00 AM | Sleep tras 48 h sin uso | Abrir la URL 1 min antes; documentado en la entrega final |
| `git push hf` pide credenciales y rechaza | Token sin permiso Write | Regenerar token con tipo Write |
| YAML del README falla con error de parseo | Edición rompió el front-matter | Verificar que las primeras 3 líneas sean `---`, contenido, `---` y que las keys/values sean YAML válido |
| Mensaje "This Space is sleeping" | Inactividad 48 h+ | Click "Wake up" o cualquier interacción la despierta automáticamente |

## 9. Datos del Space actual

> Esta sección se completa después del primer deploy real.

- **URL del Space:** `https://huggingface.co/spaces/<usuario>/spine-segmentation`
- **Owner:** `<usuario>`
- **Hardware:** CPU Basic (free)
- **Repo de pesos vinculado:** `<usuario>/spine-checkpoints`
- **Fecha primer deploy:** _por completar_
- **Latencia media de inferencia:** _por completar_ s
- **Tamaño del entorno construido:** _por completar_ MB
- **Storage usado (pesos en caché):** _por completar_ MB

## 10. Referencias

- [`HUGGINGFACE_SETUP.md`](HUGGINGFACE_SETUP.md) — gestión de pesos en HF Hub
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — alternativa Hetzner / VPS
- [`../README.md`](../README.md) — overview del proyecto
- [`CICLO_4_ARTEFACTOS.md`](CICLO_4_ARTEFACTOS.md) — cierre del ciclo
- HF Spaces docs: https://huggingface.co/docs/hub/spaces
- HF Spaces Gradio SDK: https://huggingface.co/docs/hub/spaces-sdks-gradio
