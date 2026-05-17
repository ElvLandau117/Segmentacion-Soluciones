# Ciclo 4 — Despliegue (Brief Inicial)

> **Estado:** 🔜 Por iniciar
> **Fecha estimada de inicio:** próximo chat
> **Input:** [`CICLO_3_ARTEFACTOS.md`](CICLO_3_ARTEFACTOS.md)
> **Output esperado:** `docs/CICLO_4_ARTEFACTOS.md`

---

## 1. Objetivo del ciclo

Desplegar la aplicación web de segmentación vertebral en un servidor accesible públicamente, integrando los 5 modelos pre-entrenados del Ciclo 3, con explicabilidad clínica y disclaimer médico.

---

## 2. Spec inicial (a refinar con la rúbrica)

### 2.1 Componentes a desplegar

```
┌────────────────────────────────────────────────────────────┐
│  USUARIO (medico, investigador, jurado)                    │
│  via browser desde cualquier dispositivo                    │
└────────────────────────────┬───────────────────────────────┘
                             │ HTTPS
                             ▼
┌────────────────────────────────────────────────────────────┐
│  NGINX (reverse proxy + SSL via Let's Encrypt)             │
│  Puerto 443 publico                                         │
└────────────────────────────┬───────────────────────────────┘
                             │ HTTP interno
                             ▼
┌────────────────────────────────────────────────────────────┐
│  GRADIO APP (Docker container)                              │
│  - Carga modelo MAnet+MiT-B5 (mejor precision) o            │
│    DeepLabV3+ (mejor ASPP) como default                     │
│  - 4 tabs: Binary | Vertebrae | Cobb | Explainability       │
│  - Inferencia CPU (~2-5s por imagen)                        │
│  - Puerto 7860 interno                                      │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Modelo en producción (decisión inicial, sujeta a revisión)

| Opción | Modelo | Tamaño | Ventaja |
|--------|--------|--------|---------|
| A | **DeepLabV3+** (ganador) | 102 MB | Mejor Dice (0.3378), tamaño moderado |
| B | MAnet + MiT-B5 | 352 MB | Mejor explicabilidad (atención) |
| C | **Ambos** (selector en UI) | 454 MB | Usuario elige |

**Recomendación inicial:** Opción C (selector en UI), ya implementado en `deployment/app.py`.

### 2.3 Infraestructura

- **Servidor:** Hetzner (el usuario tiene cuenta y server existente)
- **OS:** Linux (Ubuntu / Debian)
- **Docker:** instalado
- **Dominio:** por definir (¿usar IP por ahora?)
- **SSL:** Let's Encrypt (gratis) si hay dominio
- **Almacenamiento:** ~5 GB (Docker image + checkpoints + logs)

---

## 3. Acceptance Criteria (a refinar con la rúbrica)

### Funcionales
- [ ] App accesible vía URL pública (http o https)
- [ ] Sube imagen → recibe segmentación + Cobb + explicabilidad
- [ ] Tab de explicabilidad muestra Grad-CAM + mapa de confianza
- [ ] Reporte clínico se genera con disclaimer médico
- [ ] Latencia <5s por imagen (CPU inference)

### No funcionales
- [ ] Container reinicia automáticamente si crashea (`--restart=unless-stopped`)
- [ ] Health check endpoint (`/health`) responde 200 OK
- [ ] Logs persistentes (no se pierden al reiniciar container)
- [ ] HTTPS habilitado (si hay dominio)

### Documentación
- [ ] `docs/DEPLOYMENT.md` con pasos exactos para re-desplegar
- [ ] Diagrama de arquitectura
- [ ] Lista de comandos útiles (logs, restart, update)

---

## 4. Plan tentativo de tareas

> ⚠️ **Este plan es tentativo.** Se debe refinar después de leer la rúbrica de evaluación de la universidad.

### Unidad 4.1: Validar Dockerfile actual (2h)
- Construir imagen localmente: `docker build -t spine-app .`
- Correr container: `docker run -p 7860:7860 spine-app`
- Verificar que carga checkpoints, responde a inferencia, mide tiempo
- Si falla: refinar Dockerfile

### Unidad 4.2: Optimizar tamaño de la imagen (2h)
- Multi-stage build (separar dependencias de runtime)
- Usar `python:3.11-slim` como base (no `python:3.11`)
- Excluir checkpoints del build (montar como volumen)
- Meta: <3 GB de imagen Docker

### Unidad 4.3: Preparar archivos para Hetzner (2h)
- `docker-compose.yml` con servicio Gradio + (opcional) Nginx
- Script de deploy: `scripts/deploy_hetzner.sh`
- Documentación de configuración inicial del servidor
- Archivo `.env.example` con variables necesarias

### Unidad 4.4: Desplegar en Hetzner (3h)
- SSH al servidor
- Clonar repo + descargar checkpoints (desde OneDrive)
- Construir imagen + run container
- Configurar firewall (abrir puerto 80, 443 si SSL)
- Verificar accesibilidad pública

### Unidad 4.5: Configurar Nginx + SSL (opcional, 3h)
- Solo si hay dominio disponible
- Nginx como reverse proxy
- Let's Encrypt con certbot
- Renovación automática
- Headers de seguridad básicos

### Unidad 4.6: Smoke test end-to-end (2h)
- Acceder desde dispositivos diferentes (móvil, otro PC)
- Subir 3-5 radiografías de muestra
- Verificar: segmentación correcta, Cobb dentro de rango, panel explicabilidad
- Medir latencia real

### Unidad 4.7: Documentación + cierre del ciclo (2h)
- Crear `docs/DEPLOYMENT.md`
- Actualizar `AGENTS.md` con URL de despliegue + decisiones
- Crear `docs/CICLO_4_ARTEFACTOS.md`
- Commit + push

**Total estimado:** ~16 horas de trabajo enfocado.

---

## 5. Rúbrica de Evaluación

> ⚠️ **PLACEHOLDER**
>
> La universidad compartirá una rúbrica de evaluación específica para esta entrega
> (despliegue / componente de despliegue del proyecto final).
>
> Cuando se reciba, **pegar aquí** el contenido y re-priorizar el plan de tareas
> según los pesos de cada criterio.
>
> **Esperado en el próximo chat:**
> - Lista de criterios con peso porcentual
> - Acceptance criteria oficiales
> - Formato de entrega (URL, video, informe, código)
> - Fecha límite

```
[ESPACIO RESERVADO PARA LA RÚBRICA OFICIAL]
```

---

## 6. Dependencias del Ciclo 3

| Recurso | Ubicación | Estado |
|---------|-----------|--------|
| Código del paquete | `spine_segmentation/` | ✅ En git |
| Modelos pre-entrenados | `checkpoints/*.pth` | ✅ En local (no git) |
| App Gradio | `spine_segmentation/deployment/app.py` | ✅ En git |
| Dockerfile | `Dockerfile` | ✅ En git, **a validar** |
| Splits reproducibles | `data_splits.json` | ✅ En git |
| Notebook ejecutable | `notebooks/03_informe_final.ipynb` | ✅ En git |

---

## 7. Riesgos identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Hetzner no tiene suficientes recursos | Baja | Alto | Probar con imagen mínima primero |
| Dockerfile rompe en build | Media | Medio | Build local primero, iterar |
| Inferencia muy lenta en CPU del servidor | Media | Alto | Quantización INT8 si necesario |
| Sin dominio = sin HTTPS = navegadores bloquean cámara | Baja | Bajo | App no usa cámara, solo upload |
| Disclaimer médico insuficiente | Baja | Alto | Revisar texto con perspectiva legal |

---

## 8. Definición de "Done" para este ciclo

- [ ] App accesible públicamente vía URL (http o https)
- [ ] Test desde 3+ dispositivos diferentes exitoso
- [ ] Documentación de deployment escrita y verificada
- [ ] `docs/CICLO_4_ARTEFACTOS.md` creado
- [ ] `AGENTS.md` actualizado con URL + decisiones
- [ ] Commit con tag `v1.0-deploy` (opcional)
- [ ] Si aplica: video corto de demo grabado

---

## 9. Próximos ciclos (futuros, post-Ciclo 4)

### Ciclo 5 (tentativo) — Refinamiento de modelo
- Entrenar con augmentation más agresiva
- Probar pre-training con RadImageNet
- Ensemble de los 5 modelos
- Quantización INT8

### Ciclo 6 (tentativo) — Artículo + sustentación
- Redactar artículo IEEE/ACM
- Preparar slides de defensa
- Demo en vivo durante sustentación

---

## Referencias

- [`CICLO_3_ARTEFACTOS.md`](CICLO_3_ARTEFACTOS.md) — qué tenemos disponible
- [`../WORKFLOW.md`](../WORKFLOW.md) — reglas del repositorio
- [`../AGENTS.md`](../AGENTS.md) — contexto completo del proyecto
