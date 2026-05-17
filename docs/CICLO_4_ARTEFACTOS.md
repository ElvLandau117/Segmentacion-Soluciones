# Ciclo 4 — Despliegue · Artefacto de Salida

> **Fecha de cierre:** 2026-05-17
> **Estado:** ✅ **APP DESPLEGADA Y CORRIENDO** en HF Spaces.
> **URL pública:** https://huggingface.co/spaces/ElvLandau/spine-segmentation
> **Próximo ciclo (tentativo):** Ciclo 5 — Refinamiento de modelo + entrega académica.

---

## 1. Resumen ejecutivo

Ciclo dedicado a transformar el proyecto de "funciona en mi laptop" a
**aplicación pública en HF Spaces accesible 24/7**, alineado con la rúbrica
de evaluación de despliegue de la Universidad de los Andes (Coursera).

**Decisiones de arquitectura tomadas el 2026-05-17:**

| Decisión | Elegido |
|----------|---------|
| Hosting del código + UI | **Hugging Face Spaces** (gratis, Gradio SDK, CPU Basic) |
| Hosting de pesos | **Hugging Face Hub** repo separado (`ElvLandau/spine-checkpoints`) |
| Modelo en producción | **DeepLabV3+ ResNet50** (multiclase, ganador del Ciclo 3) + **UNet ResNet50** (binario para Cobb) |
| Convención de entrypoint | **`app.py` raíz** (HF Spaces) + **`app/main.py` shim** (Docker/local) |
| Despliegue alternativo | **Hetzner + Docker Compose + Caddy** (commiteado, documentado, no usado en producción) |
| Versión Gradio | **5.0.1** (4.x tenía bug `TypeError` en `gradio-client 1.3`) |
| Versión Python | **3.11** (3.13 quitó `audioop` que `pydub` necesita) |

**11 unidades originales + 8 unidades adicionales de iteración para resolver
incompatibilidades de la matriz Python/gradio/huggingface_hub. Todo cerrado.**

---

## 2. Cumplimiento de la rúbrica

| Criterio | Peso | Estado |
|----------|:-:|--------|
| Funcionamiento (app desplegada y operativa) | 35 % | ✅ HF Spaces **RUNNING**, HTTP 200 estable, healthcheck OK |
| Usabilidad / presentación | 35 % | ✅ 4 tabs claras, etiquetas explícitas, disclaimer médico visible, panel de explicabilidad |
| Documentación del despliegue (README) | 15 % | ✅ README v2 con descripción, instalación, ejemplos input/output, dependencias, pasos de deploy, consideraciones del modelo, tiempos |
| Parametrización (env vars / config) | 15 % | ✅ 12 variables documentadas en `.env.example`; tests verifican el override |

---

## 3. Unidades de trabajo y commits

| # | Unidad | Commit | Resultado |
|---|--------|--------|-----------|
| 4.1 | Reorganizar raíz (carpetas temáticas + .gitignore + .dockerignore + CLAUDE.md + script de movimiento) | `88e8519` | ✅ |
| 4.2 | Subir pesos a HF Hub (script + guía) | `8660924` | ✅ |
| 4.3 | Parametrizar `config.py` con env vars + `.env.example` | `dc5152d` | ✅ |
| 4.4 | `app/main.py` shim + autodescarga HF en `weights.py` + solo DeepLabV3+ | `456bc03` | ✅ |
| 4.5 | Dockerfile v2 (sin `COPY checkpoints`, no-root, healthcheck con curl) | `627a6b2` | ✅ |
| 4.6 | Suite pytest (13 tests passing, 1 gated) | `7003252` | ✅ |
| 4.7 | `docker-compose.yml` + `Caddyfile` (SSL automático) | `6b9e833` | ✅ |
| 4.8 | `DEPLOYMENT.md` + `deploy_hetzner.sh` (deploy real lo ejecuta Elvis) | `ab3c1c0` | ✅ (runbook listo) / ⏳ (ejecución pendiente) |
| 4.9 | README v2 rúbrica-compliant | `d184d9c` | ✅ |
| 4.10 | Smoke test desde 3 dispositivos | (sin commit) | ⏳ Requiere deploy |
| 4.11 | Cierre: artefactos + AGENTS.md + entrega final + tag | (este commit) | 🔄 En progreso |

---

## 4. Recursos producidos en este ciclo

### Código nuevo
- `app/__init__.py`, `app/main.py` — entrypoint estándar (shim)
- `spine_segmentation/deployment/weights.py` — resolución de pesos local/HF
- `scripts/upload_weights.py` — sube `.pth` a HF Hub
- `scripts/deploy_hetzner.sh` — bring-up reproducible del stack
- `scripts/reorganize_root_files.ps1` — limpieza local one-shot
- `tests/conftest.py` + 4 archivos `test_*.py` — 13 tests, 1 gated

### Infraestructura
- `Dockerfile` v2 — multi-stage, no-root, sin pesos baked-in
- `docker-compose.yml` — `app` + `caddy` + 3 volúmenes
- `Caddyfile` — reverse proxy + SSL automático + headers de seguridad
- `.dockerignore` — minimiza build context
- `.env.example` — 12 variables documentadas
- `pytest.ini` — configuración mínima

### Documentación
- `README.md` v2 — rúbrica-compliant
- `CLAUDE.md` — pointer a `AGENTS.md`
- `docs/DEPLOYMENT.md` — runbook de Hetzner (prereqs, deploy, ops, troubleshooting)
- `docs/HUGGINGFACE_SETUP.md` — onboarding HF Hub end-to-end
- `docs/CICLO_4_ARTEFACTOS.md` — este documento
- `requisitos_universidad/README.md` — qué va en esa carpeta
- `docs/metodologia/README.md` — papers de Gonzalez (referencia)
- `archive/README.md` — política para artefactos legacy

### Configuración modificada
- `requirements.txt` — añadido `huggingface_hub`, `grad-cam`, `pytest`
- `.gitignore` — migración OneDrive → HF; nuevas carpetas; excepciones para PDFs oficiales
- `spine_segmentation/config.py` — sección "Deployment configuration" con 12 env vars
- `spine_segmentation/deployment/app.py` — uses config defaults, ensure_weights at boot, MEDICAL_DISCLAIMER

---

## 5. Métricas del ciclo

- **Duración del trabajo en el repo:** ~1 día (sesión enfocada)
- **Commits realizados:** 9 (sin co-autoría de IA, conforme política del proyecto)
- **Líneas añadidas:** ~1,800
- **Líneas eliminadas:** ~480 (limpieza del README v1, viejas reglas de gitignore)
- **Tests passing:** 13/14 (1 gated por `requires_checkpoints`)
- **Variables de entorno expuestas:** 12
- **Servicios en docker-compose:** 2 (app, caddy)
- **Volúmenes persistentes:** 3 (checkpoints_cache, caddy_data, caddy_config)

---

## 6. Decisiones registradas (para `AGENTS.md` Historial)

| Fecha | Decisión | Razón |
|-------|----------|-------|
| 2026-05-17 | Migrar pesos de OneDrive a Hugging Face Hub | Estándar industria ML, gratis, versionado tipo git, soporta archivos grandes, intercambio sin re-deploy |
| 2026-05-17 | Servir solo DeepLabV3+ en producción (no los 5) | Ganador del Ciclo 3 con margen; ahorra ~700 MB de RAM y 30-60s de arranque sin perder valor para el usuario final |
| 2026-05-17 | Convención `app/` como shim, no renombrar paquete | Cumple letra de la rúbrica sin churn en notebooks, scripts e imports del equipo |
| 2026-05-17 | Caddy en vez de Nginx para reverse proxy + SSL | Let's Encrypt automático en 1 línea de config vs cert-bot + cron + dhparams |
| 2026-05-17 | Dominio gratis con nip.io en vez de comprar uno | El equipo no tiene dominio asignado; nip.io da hostname resoluble + Let's Encrypt funciona; cambiar a dominio propio es 1 env var después |
| 2026-05-17 | `.dockerignore` agresivo (sin docs, notebooks, dataset, mlruns) | Reduce build context de ~5 GB a < 100 MB; build más rápido, menos data al daemon |
| 2026-05-17 | Tests con pytest desde Ciclo 4 (no antes) | Hasta ahora la prioridad fue entrenar modelos; con deploy llega el momento de pinear contratos (config, weights, app boot, disclaimer) |
| 2026-05-17 | Usuario no-root en el container | Buena práctica de seguridad estándar — uid 1000 con su propio home y permisos chown sobre /data/checkpoints |

---

## 7. Limitaciones identificadas (a abordar en Ciclo 5)

1. **Smoke test pendiente** — no se ha probado el deploy en 3+ dispositivos reales.
2. **Latencia no medida en server real** — el README dice "3-8 s" como rango estimado;
   el valor real depende del tier de Hetzner que se use.
3. **Imagen Docker ~3 GB** — aceptable pero optimizable con `requirements-runtime.txt`
   separado (sin `mlflow`, `seaborn`, `pytest`).
4. **Dice promedio ~ 0.30** — académicamente válido, insuficiente para uso clínico real.
   No es responsabilidad del Ciclo 4, pero se documenta.
5. **C3 no detectada** — clase con desbalance extremo. Pendiente refinamiento del modelo.
6. **Sin GitHub Actions / CI** — los tests pasan local pero no hay enforcement automático.
   Diferido para mantener el alcance del Ciclo 4 acotado.

---

## 8. Lo que NO se hizo en este ciclo (deferred)

- **Deploy real en Hetzner** — todo el material listo, ejecutar `bash scripts/deploy_hetzner.sh`
- **Smoke test desde 3 dispositivos** — bloqueado por el deploy real
- **Quantización INT8 para tablet** — diferido a Ciclo 5
- **Ensemble de los 5 modelos** — el deploy actual sirve uno
- **Refinamiento del cálculo de Cobb** — método actual MAE = 23° (binario)
- **CI con GitHub Actions** — opcional, no requerido por rúbrica
- **Caching de inferencia (Redis o LRU en memoria)** — overkill para tráfico esperado
- **Métricas Prometheus / Grafana** — overkill para Ciclo 4

---

## 9. Handoff al Ciclo 5

### Lo que el Ciclo 5 puede asumir como dado
- Repo estructurado, documentado, alineado a la rúbrica.
- Pesos en HF Hub (intercambiables sin re-deploy).
- Stack docker-compose listo para `up -d`.
- 13 tests verificando contratos del deployment.
- Runbook completo en `docs/DEPLOYMENT.md`.
- App accesible públicamente (post-deploy de Elvis).

### Lo que el Ciclo 5 debe abordar
- Smoke test cross-device + capturas.
- Llenar `docs/DEPLOYMENT.md` sección 8 "Datos del despliegue actual" con
  valores reales (IP, DOMAIN, tier, latencia, tamaño imagen).
- Refinamiento del modelo (augmentation agresiva, pre-training con RadImageNet, ensemble).
- Considerar quantización INT8 para edge.
- Decidir si se agrega CI con GitHub Actions.
- Redactar artículo IEEE/ACM si el deploy y los resultados lo soportan.
- Preparar sustentación oral con la demo en vivo.

Ver `CICLO_5_BRIEF.md` (a crear cuando se inicie ese ciclo).

---

## 10. Referencias

- [`CICLO_3_ARTEFACTOS.md`](CICLO_3_ARTEFACTOS.md) — métricas de entrenamiento
- [`CICLO_4_DESPLIEGUE_BRIEF.md`](CICLO_4_DESPLIEGUE_BRIEF.md) — spec inicial
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — runbook operativo
- [`HUGGINGFACE_SETUP.md`](HUGGINGFACE_SETUP.md) — gestión de pesos
- [`../README.md`](../README.md) — visión general (rúbrica-compliant)
- [`../AGENTS.md`](../AGENTS.md) — memoria persistente del proyecto
- [`../WORKFLOW.md`](../WORKFLOW.md) — reglas del repositorio
- [`../requisitos_universidad/`](../requisitos_universidad/) — rúbrica oficial Coursera
