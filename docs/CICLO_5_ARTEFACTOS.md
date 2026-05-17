# Ciclo 5 — UX clinica del Cobb · Artefacto de Salida

> **Fecha de cierre:** 2026-05-17 noche (Ciclo 5) + 5.1 polish la misma noche
> **Estado:** ✅ COMPLETO
> **URL pública:** https://huggingface.co/spaces/ElvLandau/spine-segmentation
> **Próximo ciclo (tentativo):** Ciclo 6 — Refinamiento del modelo, entrega académica final, sustentación.
>
> **Addendum 5.1** (mismo día): polish de la visualización del Cobb. Ver sección 11.

---

## 1. Resumen ejecutivo

Ciclo de **mejora de UX clínica del Cobb angle sin reentrenamiento**. Motivado
por pruebas manuales del Ciclo 4 que revelaron 3 problemas:

1. Algunos casos de escoliosis se reportaban como "Normal" (Assessment se
   calculaba sobre el `cobb_multiclass`, que tiene MAE 26-45° y correlación
   negativa en el peor caso).
2. La visualización del Cobb era solo texto sobreimpreso — no mostraba las
   líneas geométricas del ángulo.
3. El usuario no tenía forma de saber si confiar en la medida (¿binary?
   ¿multiclass? ¿coinciden?).

Inspirado parcialmente en **Shi et al. 2025** (`archive/2509.24898v1.pdf`,
"Accurate Cobb Angle Estimation via SVD-Based Curve Detection and Vertebral
Wedging Quantification"). Su modelo HRNet+Swin con dual-task heatmap+vector
y métrica VWI lograron MAE 2.55° con 630 imágenes adolescentes anotadas con
landmarks dual-rater. **Replicar su modelo no es viable en nuestro contexto**
(no tenemos landmarks anotados, solo 174 imgs train, single-rater, dataset
MaIA mezcla edades). Extrapolamos solo dos ideas del paper:

- **Idea A** — Visualización con cajas en upper/lower end vertebrae + líneas
  tangentes a los endplates (su Fig 1).
- **Idea B** — Combinar múltiples mediciones para una decisión clínica
  (en nuestro caso, fusionar binary + multiclass + concordancia).

---

## 2. Decisiones de arquitectura

| Decisión | Elegido |
|----------|---------|
| Severity Assessment | **Cobb binary** (más robusto: MAE 23°, r=0.66). Multiclass como referencia anatómica + fallback si binary falla. |
| Visualización del Cobb | Cajas verdes en end vertebrae + líneas tangentes rojas a los endplates + header con ambos ángulos. Función `draw_cobb_angle_visualization` en `evaluation/visualize.py`. |
| UI panel de resultados | Texto multi-bloque: Cobb (binary + multi), CONCORDANCE (con umbrales 5°/15°), Assessment (basado en binary), Vertebrae detected. |
| Test builder | Refactor de `predict()` closure → `build_results_text()` puro a nivel de módulo (testeable sin Gradio ni modelo). |
| Reentrenamiento | **NO**. Todo el upside salió de post-procesamiento + visualización + UI. |
| Mecanismo de deploy | `scripts/upload_to_space.py` con `--file` repetido (commit atómico al Space). Rebuild ~90s porque solo cambiaron .py (no `requirements.txt`). |

---

## 3. Unidades de trabajo y commits

| # | Unidad | Commit | Resultado |
|---|--------|--------|-----------|
| 5.1 | Branch + worktree nuevos (`cycle5-cobb-ui` desde main) | n/a | ✅ |
| 5.2 | Assessment via Cobb binary (fix en `deployment/app.py`) | `0993182` | ✅ |
| 5.3 | `draw_cobb_angle_visualization` + reescritura de `_create_cobb_visualization` | `ba9ed31` | ✅ |
| 5.4 | UI dual-Cobb con indicador de concordancia | `65a2d4d` | ✅ |
| 5.5 | Tests (4 nuevos) + refactor `build_results_text` testeable | `00b015d` | ✅ 17 passed + 1 skipped |
| 5.6 | Deploy: subida atómica de 3 archivos al Space + smoke remoto | (sin commit nuevo) | ✅ Space `RUNNING`, smoke verde |
| 5.7 | Cierre: AGENTS.md + este artefacto + reciclado PROMPT_PROXIMO_CHAT.md | (este commit) | 🔄 En progreso |

---

## 4. Recursos producidos

### Código modificado
- `spine_segmentation/deployment/app.py`
  - Helper module-level `build_results_text()` (pure, testeable).
  - Assessment usa `cobb_binary` como fuente primaria; multiclass es fallback.
  - Panel de resultados reorganizado en 4 bloques.
- `spine_segmentation/deployment/inference.py`
  - `_create_cobb_visualization` ahora delega en
    `draw_cobb_angle_visualization`; fallback text-only si no hay
    `multiclass_mask`.
- `spine_segmentation/evaluation/visualize.py`
  - Nueva función `draw_cobb_angle_visualization` con cajas + tangentes + header.

### Tests nuevos (`tests/test_app_smoke.py`)
- `test_build_results_text_uses_binary_for_assessment` — pin de regresión.
- `test_build_results_text_includes_concordance_when_both_succeed` — cubre
  rangos "High agreement" y "Significant discrepancy".
- `test_build_results_text_falls_back_to_multiclass_when_binary_fails` —
  ruta multiclass-only sin bloque CONCORDANCE.
- `test_draw_cobb_angle_visualization_modifies_image` — verifica que la
  función modifica los pixeles (no es no-op).

---

## 5. Métricas del ciclo

- **Duración:** ~1 sesión enfocada (~16 h estimadas, ejecutadas en 1 día).
- **Commits:** 5 (sin co-autoría de IA, conforme política).
- **Líneas añadidas:** ~430
- **Líneas eliminadas:** ~135
- **Archivos modificados:** 4 (`app.py`, `inference.py`, `visualize.py`,
  `tests/test_app_smoke.py`)
- **Tests passing:** 17/18 (1 gated por `requires_checkpoints`).
- **Rebuild del Space:** ~90 s (solo .py, sin tocar deps).

---

## 6. Smoke test remoto (caso S_21 — escoliosis del dataset)

Llamada via `gradio_client.Client.predict()` con `S_21.jpg` (latencia 11.9 s):

```
=== COBB ANGLE ===
Binary method:      15.1 deg  (more robust, recommended)
Multiclass method:   0.4 deg  (anatomical info: Upper=C5, Lower=T10)

CONCORDANCE: Review recommended - methods differ slightly
    |Binary - Multiclass| = 14.7 deg

=== ASSESSMENT (based on Binary) ===
Mild scoliosis (10-25 degrees)

=== VERTEBRAE DETECTED (18) ===
C5, C6, C7, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, L1, L2, L3
```

**Antes del Ciclo 5** este caso habría sido reportado como **"Normal (<10°)"**
porque el Assessment se calculaba sobre el multiclass=0.4°. **Después del
Ciclo 5** se reporta correctamente como **"Mild scoliosis"** (basado en
binary=15.1°). Resuelve el false-negative que Elvis observó.

---

## 7. Lo que NO entró en este ciclo (deferred)

- **Replicar el modelo del paper Shi et al. 2025** — requiere landmarks
  anotados, no viable.
- **VWI (Vertebral Wedging Index)** — requiere upper/lower endplate angles
  por vértebra, que nuestro multiclass no da.
- **SVD sobre matriz de ángulos** — mismo problema.
- **SVD alternativo sobre centroides de vértebras** — experimento; podría
  ir en Ciclo 6 como mejora del Cobb multiclass.
- **Constraint biomecánico como post-procesamiento** — filtrar vértebras
  con ángulos fuera del rango de sus vecinas; experimento Ciclo 6.
- **Enmascarar el confidence map por la predicción** — la viz actual es
  poco informativa (casi todo verde); fix simple para Ciclo 6.
- **Seg-Grad-CAM en vez de Grad-CAM vanilla** — mejor activación espacial
  para tareas densas. Ciclo 6.
- **Reentrenamiento** — explícitamente excluido. Espera al Ciclo 6 o posterior.

---

## 8. Handoff al Ciclo 6

### Lo que el Ciclo 6 puede asumir como dado
- App pública corriendo con UX clínica del Cobb (este ciclo).
- 17 tests verificando contratos del deployment (incluyendo dual-Cobb UI).
- Mecanismo de deploy reproducible (`scripts/upload_to_space.py`).
- 5 modelos entrenados (Ciclo 3) + 2 en producción (DeepLabV3+ multi, UNet binary).
- Memoria del paper Shi et al. 2025 documentada: qué se extrapoló y qué no.

### Candidatos para el Ciclo 6 (a priorizar con Elvis)
1. Mejorar Cobb multiclass (SVD sobre centroides + constraint biomecánico
   post-proc + votación robusta).
2. Mejorar explicabilidad (enmascarar confidence map + Seg-Grad-CAM).
3. Refinamiento del modelo (augmentation agresiva, ensemble, pre-training
   RadImageNet).
4. Quantización INT8 para edge (tablet).
5. CI con GitHub Actions (opcional).
6. Sustentación oral + slides + demo en vivo + smoke test cross-device.
7. Artículo IEEE/ACM (si los resultados lo soportan).

---

## 9. Referencias

- [`CICLO_4_ARTEFACTOS.md`](CICLO_4_ARTEFACTOS.md) — despliegue + fix gradio.
- [`PROMPT_PROXIMO_CHAT.md`](PROMPT_PROXIMO_CHAT.md) — onboarding del próximo chat (Ciclo 6).
- `archive/2509.24898v1.pdf` — Shi et al. 2025 (HKU + Renji, IEEE J-BHI).
- [`../AGENTS.md`](../AGENTS.md) — memoria persistente del proyecto.
- [`../WORKFLOW.md`](../WORKFLOW.md) — reglas del repositorio.

---

## 11. Addendum 5.1 — Polish de la visualización del Cobb

> **Fecha:** 2026-05-17 noche (misma sesión, posterior al cierre original).
> **Motivación:** Elvis subió 2 screenshots de la pestaña Cobb Angle y
> reportó "solo veo dos líneas separadas... visualmente yo veo la desviación
> pero la imagen no la comunica". Diagnóstico: la viz dibujaba líneas A LO
> LARGO del endplate en lugar de PERPENDICULARES, y no tenía arco ni punto
> de intersección.

### Cambios

| # | Mejora | Implementación |
|---|--------|----------------|
| A | Perpendiculares correctas al endplate | `_endplate_vectors` retorna `(tangent, perpendicular)`. La función principal usa el `perpendicular` para las líneas largas y el `tangent` para los marcadores cortos. |
| B | Punto de intersección + arco del ángulo | `_line_intersection` + `_draw_angle_arc` con `cv2.ellipse`. Solo dibuja el arco si la intersección cae dentro del frame Y el ángulo es > 1°. |
| C | Marcadores cyan sobre los endplates | `_draw_endplate_marker`, ancho 70% del bbox de la vértebra. Análogo a las "ticks" que en Fig 1 del paper marcan el endplate. |
| D | Mini "Cobb-meter" en esquina | `_draw_speedometer` con escala 4x en la aguja para que ángulos pequeños se vean. Triggers: ángulo < 8° o intersección fuera de frame. |
| E | Overlay del binary: spline + inflection points | `_draw_binary_overlay` consume `cobb_binary["spline_x"]`, `cobb_binary["spline_y"]`, `cobb_binary["inflection_points"]` — datos que ya devolvía `cobb_from_binary` y que la viz del Ciclo 5 no usaba. |

Modularización: 6 helpers privados al mismo nivel que la función pública,
todos testeables aislados.

### Bug encontrado durante el smoke remoto

El primer smoke remoto reportó "0 red pixels" en la imagen final. Causa:
las llamadas `cv2.line(..., (0, 0, 255), ...)` asumían convención BGR de
OpenCV, pero el array `vis` viene de Gradio como **RGB**. OpenCV no
transforma color spaces — solo escribe los 3 valores en orden. Resultado:
las "líneas rojas" eran AZULES en la imagen final, los "círculos amarillos"
eran CYAN, etc.

Fix: cambiar todas las llamadas que esperaban rojo a `(255, 0, 0)`, e
inflection points a `(255, 255, 0)` (amarillo en RGB, distinto del cyan
de los endplate markers). Documentado como decisión en AGENTS.md sec 9.

### Smoke remoto post-fix

| Color | N_1 | S_21 | Significado |
|---|:-:|:-:|---|
| Red (BGR check) | 1295 | 929 | perpendiculares + arco + speedometer needle |
| Green | 890 | 579 | end-vertebra fills + boxes |
| Cyan | 2900 | 2975 | endplate markers + header multiclass |
| Yellow | 2567 | 2609 | inflection points + header binary |

Todos los colores presentes con conteos coherentes.

### Tests añadidos (Ciclo 5.1)

- `test_line_intersection_handles_parallel_and_crossing` — pin del contrato
  parallel→None + cross math.
- `test_endplate_vectors_are_perpendicular_unit_vectors` — verifica que
  orientation=π/2 da tan=(1,0) y perp=(0,-1) y son ortogonales.
- `test_speedometer_draws_inside_the_image` — el gauge realmente pinta
  píxeles en la mitad inferior.
- `test_binary_overlay_renders_spline_and_inflection_points` — spline +
  círculos amarillos + skip-paths para None y failed.
- Actualizado `test_draw_cobb_angle_visualization_modifies_image` para la
  nueva signature (`cobb_binary_result` dict en vez de `cobb_binary_deg` float).

Suite final: **21 passed + 1 skipped** (era 17 + 1 antes del polish).

### Commits del Ciclo 5.1

- `85186d9` fix(eval): replace cobb viz with perpendiculars + arc + endplate markers + binary overlay
- `4b612a1` test: cover new cobb visualization helpers
- `0b97bad` fix(viz): use RGB color tuples (not BGR) — the image array is RGB
- `<este>` docs(cycle5): polish cobb visualization + credit to Shi et al. 2025

### Crédito al paper

La visualización del Cobb angle está inspirada en Fig 1 de Shi et al. 2025
("Accurate Cobb Angle Estimation via SVD-Based Curve Detection and
Vertebral Wedging Quantification", IEEE J-BHI, arXiv:2509.24898).
Documentado en el docstring de `draw_cobb_angle_visualization` + README sec 9.
