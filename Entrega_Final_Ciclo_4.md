# Entrega Final — Despliegue de Soluciones

**Proyecto:** Segmentación automática de columna vertebral y vértebras en
radiografías de pacientes sanos y con escoliosis, para el desarrollo de
herramientas de apoyo a medidas radiológicas.

**Curso:** Maestría en Inteligencia Artificial — Despliegue de Soluciones

**Institución:** Universidad de los Andes

**Integrantes:**
- Elvis Hernández — `elvis.hernandez@en-firme.com`

**Fecha de entrega:** _por completar_

---

## Enlaces

| Recurso | URL |
|---------|-----|
| **Repositorio GitHub** | https://github.com/ElvLandau117/Segmentacion-Soluciones |
| **🚀 Aplicación desplegada** | **https://huggingface.co/spaces/ElvLandau/spine-segmentation** |
| **Pesos del modelo (HF Hub)** | https://huggingface.co/ElvLandau/spine-checkpoints |
| **Documentación de despliegue oficial** | [`docs/HF_SPACES_SETUP.md`](docs/HF_SPACES_SETUP.md) |
| **Documentación despliegue alternativo (Hetzner)** | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| **README principal** | [`README.md`](README.md) |
| **Memoria del proyecto** | [`AGENTS.md`](AGENTS.md) |

> **Acceso:** La aplicación es pública y no requiere credenciales.
> **Nota sobre el sleep:** tras 48 h sin uso el Space se duerme; al primer
> click despierta en 30-60 s. Si la evaluación es a una hora exacta, **abrir
> la URL 1 minuto antes** para garantizar arranque inmediato.

---

## Resumen de la solución

Sistema de segmentación vertebral con deep learning desplegado como una
aplicación web Gradio. Recibe una radiografía AP de columna y devuelve:

1. **Segmentación binaria** de la columna completa.
2. **Segmentación multiclase** de 22 vértebras individuales (C3 a L5).
3. **Ángulo de Cobb** automático por dos métodos.
4. **Explicabilidad** con Grad-CAM + mapa de confianza por pixel.
5. **Reporte clínico** con disclaimer médico explícito.

**Arquitectura del despliegue:**
- Modelo en producción: **DeepLabV3+ ResNet50** (Test Dice = 0.3378, ganador del Ciclo 3).
- Pesos hospedados en **Hugging Face Hub** (descarga automática al boot, caché persistente).
- Stack: **Docker Compose** con `app` (Gradio) + `caddy` (reverse proxy + SSL automático).
- HTTPS gratuito vía **Let's Encrypt** sobre hostname `nip.io`.
- Servidor: **Hetzner** (Linux, CPU-only).

**Tiempo de inferencia:** 3-8 segundos por imagen en CPU (medido tras deploy).

---

## Cumplimiento de la rúbrica

| Criterio | Peso | Cómo se cumple |
|----------|:-:|----------------|
| Funcionamiento | 35 % | App accesible públicamente, healthcheck activo, restart automático, pesos en HF Hub con caché persistente |
| Usabilidad / presentación | 35 % | 4 tabs claras (Binary / Vertebrae / Cobb / Explainability), etiquetas explícitas, disclaimer médico visible, panel de explicabilidad |
| Documentación README | 15 % | README v2 con descripción del problema y solución, instalación, ejemplos input/output, dependencias, pasos de despliegue, consideraciones del modelo |
| Parametrización | 15 % | 12 variables de entorno documentadas en `.env.example`; tests unitarios verifican los overrides |

Detalle exhaustivo del cumplimiento: [`docs/CICLO_4_ARTEFACTOS.md`](docs/CICLO_4_ARTEFACTOS.md).

---

## Instrucciones rápidas para evaluar

### Como usuario final (jurado)
1. Abrir la URL pública.
2. Subir una radiografía AP de columna completa (PNG/JPG, hasta 25 MB).
3. Click en **Analyze**.
4. Esperar 3-8 segundos.
5. Navegar las 4 pestañas; leer el bloque "Diagnosis Results" y el disclaimer.

### Como evaluador técnico
- **Reproducir el deploy:** ver [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) sección 3 (Deploy automatizado).
- **Probar los tests:** `pytest tests/ -v` (13 passing, 1 gated).
- **Revisar parametrización:** ver `.env.example` y `spine_segmentation/config.py` sección "Deployment configuration".
- **Inspeccionar el modelo:** ver `notebooks/03_informe_final.ipynb`.
- **Auditar las decisiones:** ver [`AGENTS.md`](AGENTS.md) sección 9 (Historial de Decisiones).

---

## Tamaño del modelo y consideraciones

| Aspecto | Valor |
|---------|-------|
| Arquitectura en producción | DeepLabV3+ ResNet50 (multiclase) + UNet ResNet50 (binario) |
| Tamaño total de pesos | ~200 MB |
| Tamaño imagen Docker | ~3 GB (pesos NO incluidos en la imagen) |
| RAM en runtime | ~1.5-2 GB |
| Hardware mínimo recomendado | 2 vCPU + 4 GB RAM + 10 GB disco |
| Soporta GPU | Sí (detección automática), pero deploy por defecto es CPU |

### Estrategia de gestión del modelo

La rúbrica menciona que "si el modelo es demasiado grande, debe proporcionarse
enlace de descarga e instrucciones para su carga". Nuestro enfoque:

- Los `.pth` NO viven en GitHub (los repositorios git no son adecuados para
  blobs binarios pesados).
- Viven en **Hugging Face Hub**, el estándar de la industria para distribuir
  modelos ML — gratis, versionado tipo git, soporta archivos grandes con LFS
  automático.
- El container hace `snapshot_download()` al boot y cachea los pesos en un
  volumen persistente. Si los pesos ya están en caché, no se re-descargan.
- **Cambiar de pesos en producción:** subes el nuevo `.pth` al repo de HF,
  reinicias el container con `docker compose restart app`. Cero cambios en
  el código.

Documentación detallada: [`docs/HUGGINGFACE_SETUP.md`](docs/HUGGINGFACE_SETUP.md).

---

## Para convertir este documento a Word o PDF

La rúbrica solicita formato Word o PDF. Este archivo está en Markdown para
facilitar su mantenimiento en git. Para convertir:

```bash
# A PDF (requiere pandoc + LaTeX):
pandoc Entrega_Final_Ciclo_4.md -o Entrega_Final_Ciclo_4.pdf \
    --pdf-engine=xelatex

# A Word:
pandoc Entrega_Final_Ciclo_4.md -o Entrega_Final_Ciclo_4.docx
```

O usar conversores online como https://pandoc.org o https://md-to-pdf.fly.dev.

Los archivos `.pdf` y `.docx` generados están en `.gitignore` (son
artefactos derivados); el .md es la fuente.
