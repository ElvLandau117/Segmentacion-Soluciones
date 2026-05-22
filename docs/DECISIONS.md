# DECISIONS.md — Indice navegable de decisiones del proyecto

> **Fuente completa**: [`AGENTS.md`](../AGENTS.md) sec 9 (Historial de
> Decisiones). Este archivo es una vista resumida por ciclo + por tema,
> para que jurados, compañeros médicos y nuevos agentes puedan navegar
> las decisiones sin leer las ~820 líneas del AGENTS.md completo.
>
> Mantenido desde el Ciclo 5.12 (2026-05-22). Cada cierre de ciclo
> añade entradas tanto aquí como en AGENTS.md sec 9.

---

## Por ciclo (resumen ejecutivo)

| Ciclo | Tema central | Decisión clave |
|---|---|---|
| **1-2** | Infraestructura + Training | PyTorch + SMP (no TensorFlow); batch_size escalado 4→12; esquema 24 clases (no 36). |
| **3** | 5 modelos comparados | DeepLabV3+ gana sobre transformers (Dice 0.3378); pesos via OneDrive (legacy). |
| **4** | Despliegue público | Pivote a HF Spaces (no Hetzner); Gradio 5.50.0 (fix de gradio-client bug `api_info`); script `upload_to_space.py` (no `git push hf`). |
| **5** | UX clínica del Cobb | Assessment basado en binary (MAE 23°, r=0.66), no multiclass (MAE 26-45°, r negativa). |
| **5.1** | Cobb viz | Perpendiculares al endplate (no a lo largo) + arco del ángulo + speedometer fallback. Convención RGB en cv2.line (no BGR). |
| **5.2** | Multi-curva (rotoescoliosis) | `cobb_from_binary` devuelve `list[curves]`. Multiclass solo para naming Tn-Lm. |
| **5.3** | Coverage del binary | Umbral binary 0.5→0.3; cierre morfológico vertical 3×25 ANTES del filtro por componente; multi-pass adaptativo en smoothing. |
| **5.4** | Rotación de captura | SVD del skeleton → tilt_deg; ROTATION WARNING en UI (no auto-rotate). Filtro de curvas degeneradas `MIN_IP_Y_DISTANCE_PX=30`. |
| **5.5** | Manual rotation UI | Slider -180..180 + 5 botones rápidos (rechazo de auto-rotate por riesgo de signo). `BORDER_REPLICATE` + deadband 0.5°. |
| **5.6** | Live preview rotación | `gr.State` para imagen original; rotación dispara en `slider.change`, no en Analyze. Predict() pierde `rotation_deg`. |
| **5.7** | i18n + multiclass cleanup | ES/EN toggle (default Español, Nivel B = Diagnosis + header). Quitar bloque CROSS-CHECK confuso del frontend. |
| **5.8** | Explainability polish | Mask Grad-CAM y Confidence por `binary_mask` predicha; percentile clip p95; títulos in-image + colorbars verticales. |
| **5.9** | Reference image fija | gr.Image bilingüe ARRIBA del panel dinámico; recreación programática del mockup (PSD original no disponible). |
| **5.10** | Lateralidad clínica | Convención anatomía del paciente (no perspectiva del viewer). Swap del ternario en `_curve_direction`. Sample base S_22 → S_200. |
| **5.11** | Sample-invariant arrows | Posiciones de blobs y arrows DERIVADAS del spine bbox (no hardcoded). Helpers `_derive_visual_anchors` + `_pixel_to_figure_coords`. |
| **5.12** | Coord centering bug | Fix de `_pixel_to_figure_coords` para `aspect='equal'` centering. `_imshow_bbox_in_figure` computa el rect real. Constante `FIG_SIZE_IN`. |
| **5.12** | DECISIONS.md | Este archivo. AGENTS.md sec 9 sigue como source-of-truth completa. |

---

## Por tema clínico

### Convención de lateralidad (Ciclo 5.10)
La app reporta SIEMPRE en **anatomía del paciente** (estándar radiológico).
En radiografía AP, el lado derecho del paciente aparece a la izquierda del
viewer. Caso de regresión: `S_158` debe reportar "convexidad derecha"
porque la columna se curva hacia la derecha del paciente (= a la izquierda
del viewer en la imagen). Implementado en `cobb_angle.py:_curve_direction`.

### Cobb Severity (Ciclo 5)
Calculado sobre Cobb binary (más robusto: MAE 23°, r=0.66) — NO sobre
multiclass (MAE 26-45°, correlación negativa en el peor caso). El
multiclass se mantiene para:
- Naming de vértebras (label transfer de IPs a Tn-Lm).
- Visualización (cajas verdes en end vertebrae).
- Fallback cuando binary FALLA completamente.

### Multi-curva (Ciclo 5.2)
`cobb_from_binary` devuelve TODAS las curvas detectables (pares
adyacentes de inflection points del spline). UI reporta "Curva
principal / secundaria / Curva N" con direcciones y nombres. Caso
pivote: `S_100` (S-shape severa) reporta 2 curvas (84° torácica + 65°
lumbar), antes habría salido como un solo ángulo.

### Coverage warning (Ciclo 5.3)
Se emite bloque `=== COBERTURA ===` cuando `binary_mask` cubre <70%
de la altura O segmenta <15 vértebras. Assessment cambia a
"Inconclusive — insufficient coverage" cuando coverage parcial +
Cobb ~0° (evita falsos negativos tipo S_22 pre-5.3).

### Rotación de captura (Ciclos 5.4-5.6)
- **Detección** (5.4): SVD del skeleton → tilt_deg vs vertical;
  warning si `abs(tilt) > 12°`.
- **Control manual** (5.5): slider -180..180 + 5 botones rápidos
  (no auto-rotate por riesgo de signo ambiguo).
- **Live preview** (5.6): `gr.State` preserva original; slider.change
  rota la mostrada en vivo (<300ms en CPU).

### Explicabilidad (Ciclos 5.8-5.12)
- **Masking** (5.8): Grad-CAM y Confidence enmascarados por
  `binary_mask` predicha — sólo se colorea DENTRO del spine detectado.
- **Reference image** (5.9): mockup educativo fijo arriba del panel
  dinámico, bilingüe, con 5 callouts numerados + colorbars + disclaimer.
- **Sample base** (5.10): cambiado de S_22 (modesto, Cobb 24.9°) a
  S_200 (S-curve claramente visible).
- **Sample-invariant arrows** (5.11): blobs simulados y arrow targets
  derivados del spine bbox del sample que se cargue.
- **Coord centering fix** (5.12): aspect='equal' de matplotlib centra
  la imagen 512×512 dentro de un ax rect taller-than-wide, dejando
  márgenes verticales. El converter ahora compensa este offset.

---

## Por tema arquitectónico

### Hosting y despliegue
- **Producción**: HF Spaces (`ElvLandau/spine-segmentation`), pivote
  del Ciclo 4. Gratis, sin servidor propio, HTTPS gestionado por HF.
- **Alternativo**: Docker + Caddy en Hetzner (deploy alternativo
  commiteado, no usado en producción).
- **Pesos**: HF Hub `ElvLandau/spine-checkpoints` (público, 2 .pth =
  226 MB).

### Deploy mechanism
- `scripts/upload_to_space.py` con **`--path-in-repo` SIEMPRE
  explícito** (lección Ciclo 5.5: sin esto, el archivo va a basename
  raíz y sobrescribe el shim).
- `HfApi.upload_file()` / `create_commit()`, no `git push hf` (no
  queremos tocar la historia LFS del Space).
- Commit atómico por ciclo. Rebuild ~90s cuando sólo cambian .py;
  ~3 min cuando también cambian requirements.

### Tests
- `pytest` desde Ciclo 4. Suite actual: **66 passed + 1 skipped**
  (gated por `requires_checkpoints` cuando no hay .pth locales).
- Cobertura: config env vars, weights resolution, app boot, disclaimer,
  pipeline contracts, build_results_text helpers, draw_cobb_*, i18n,
  reference image (anchors + bbox).

### i18n
- Dict-based en `spine_segmentation/deployment/i18n.py`.
- Default Español (target audience U. Andes / Colombia).
- Nivel B: Diagnosis Results + Header markdown + Explainability
  markdown + reference image son sensibles al toggle.
- Nivel C (tabs, slider, botones rápidos en mayúsculas) NO implementado
  porque requeriría recrear el Blocks completo.

### Reference image generator
- `scripts/generate_explain_reference.py` (no se despliega al Space,
  vive solo en el repo).
- Sample-invariant desde Ciclo 5.11 (anchors derivados del spine bbox).
- Coord-corrected desde Ciclo 5.12 (compensa centering de aspect=equal).
- Sample oficial actual: **S_200** (cambio del 5.10).

---

## Known issues / known limitations

### Gradio `FileNotFoundError /tmp/gradio/*.jpg` (Ciclo 5.12)
**Status**: known upstream issue, no patch en nuestro código.

**Síntoma**: en los logs del Space ocasionalmente aparece:
```
FileNotFoundError: [Errno 2] No such file or directory:
  '/tmp/gradio/<hash>/<filename>.jpg'
```

**Causa**: el error ocurre en `gradio.Image.preprocess()` (línea 254
de `gradio/components/image.py`) ANTES de invocar nuestro callback
`predict()`. El stack trace pasa por `gradio/blocks.py:preprocess_data`
y nunca llega a nuestro código.

Razones esperadas:
- Cold start / worker restart purga `/tmp` automáticamente.
- Gradio cleanup periódico de archivos viejos en `/tmp/gradio/`.
- Doble-click rápido en Analyze (race condition con cleanup).

**Mitigación**: el usuario refresca el browser y vuelve a subir la
imagen. Es esporádico y no afecta a la mayoría de sesiones.

**No-action por ahora**: un patch puro en Python no es viable (el
crash es upstream). Opciones futuras si la frecuencia sube:
1. Upgrade de Gradio (riesgo: la 5.50.0 fue elegida específicamente
   para evitar el bug `gradio-client api_info` del Ciclo 4.10).
2. Pre-copiar el upload a una ubicación persistente en el handler de
   `input_image.upload`.

### Otros limitations menores
- **Threshold de tilt 12° es borderline** en escoliosis severa (S_100,
  S_150 disparan ROTATION WARNING aunque la rotación es la curvatura
  patológica real). Candidato para recalibrar en Ciclo 6.
- **Per-class Dice de C3 y C4 es 0.0 / 0.11** — vértebras cervicales
  muy raras en el dataset. Reentrenamiento con augmentation lumbar
  agresivo (Ciclo 6 candidato).
- **Multiclass MAE 26-45°** — usado sólo para naming, no para Cobb.
  Mejoras futuras vía SVD sobre centroides + constraint biomecánico.

---

## Referencias

- [`AGENTS.md`](../AGENTS.md) sec 9 — historial completo (single
  source of truth).
- `docs/CICLO_N_ARTEFACTOS.md` — detalle por ciclo (motivación,
  diagnóstico, cambios, tests, smoke remoto, limitaciones honestas).
- [`../WORKFLOW.md`](../WORKFLOW.md) — reglas no negociables del repo.
- [`../README.md`](../README.md) — visión general y URL pública.
