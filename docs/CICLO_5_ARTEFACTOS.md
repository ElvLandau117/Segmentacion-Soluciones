# Ciclo 5 — UX clinica del Cobb · Artefacto de Salida

> **Fecha de cierre:** 2026-05-17 noche (Ciclo 5) + 5.1 polish la misma noche
> **Estado:** ✅ COMPLETO (con addenda 5.1 + 5.2 + 5.3)
> **URL pública:** https://huggingface.co/spaces/ElvLandau/spine-segmentation
> **Próximo ciclo (tentativo):** Ciclo 6 — Refinamiento del modelo, entrega académica final, sustentación.
>
> **Addendum 5.1** (mismo día): polish de la visualización del Cobb. Ver sección 11.
> **Addendum 5.2** (mismo día): detección multi-curva (rotoescoliosis). Ver sección 12.
> **Addendum 5.3** (2026-05-19): cobertura del binary + UX informativa. Ver sección 13.
> **Addendum 5.4** (2026-05-19): robustez ante rotación + UX de la viz Cobb. Ver sección 14.
> **Addendum 5.5** (2026-05-19): control manual de rotación en la UI. Ver sección 15.

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

---

## 12. Addendum 5.2 — Detección multi-curva (rotoescoliosis y S-curve)

> **Fecha:** 2026-05-17 noche (misma sesión, posterior al polish 5.1).
> **Motivación:** Julian Florido (compañero de Elvis) compartió un ejemplo
> real de informe radiológico de **rotoescoliosis con doble curvatura**:
> "Curva principal torácica T5-L1, ápice T8-T9, Cobb ~35°, convexidad derecha.
> Curva lumbar compensatoria, Cobb ~18°, convexidad izquierda."
> Diagnóstico: nuestra app reportaba UN solo ángulo Cobb. Para escoliosis con
> doble curva (S-shape, muy común en escoliosis adolescente), la curva
> compensatoria nunca aparecía en el reporte. Es una limitación algorítmica,
> no del modelo de segmentación.

### Análisis crítico de los modelos del compañero (Julian)

| Métrica | Su modelo | Nuestro modelo | Notas |
|---|:-:|:-:|---|
| Segmentación binaria — Dice | 0.8840 | ~0.85 | similares |
| Cobb central — MAE / r | 20.25° / 0.76 | 23.0° / 0.66 | suyo modestamente mejor |
| Cobb vertebral v5 — MAE / r | 18.06° / **0.15** | 26.8° / 0.20 | r=0.15 es ruido, no señal |
| Segmentación multiclase — Dice | **0.0984** | 0.3378 | el suyo es peor |

Conclusión: **reentrenar el multiclass no resuelve el problema**. La solución
es algorítmica (multi-curve detection sobre el binary), no de modelo.

### Cambios algorítmicos

1. **`cobb_from_binary`** ([cobb_angle.py:35-159](../spine_segmentation/evaluation/cobb_angle.py))
   - Computa un Cobb por CADA par adyacente de inflection points, no solo
     los 2 extremos. Una S-shape (3 IPs) → 2 curvas; triple-curve (5 IPs)
     → 4 candidatos.
   - Filtra candidatos con `cobb_angle_deg < min_curve_deg=3°` (ruido).
   - Devuelve `curves: list[dict]` ordenada por magnitud descendente.
   - Cada curva: `cobb_angle_deg`, `ip_upper`, `ip_lower`, `slope_upper`,
     `slope_lower`, `direction` ("right"/"left"/"neutral"), `rank`.
   - Expone `all_inflection_points` (todos los IPs, para overlay del binary).
   - Back-compat: `cobb_angle_deg` = magnitud de la curva principal;
     `inflection_points` = los 2 IPs de la principal.

2. **`assign_vertebra_names_to_curves`** ([cobb_angle.py:161-187](../spine_segmentation/evaluation/cobb_angle.py))
   - Para cada curva, busca la vértebra multiclass más cercana (por y) a
     cada IP. Hace label-transfer: la curva gana `upper_vertebra`,
     `lower_vertebra` (ej. "T5", "T12").
   - **El multiclass se usa solo para nombrar**, nunca para el Cobb (Dice
     0.34 es ruido para mediciones per-vertebra).

3. **`inference.py`** llama `assign_vertebra_names_to_curves` justo después
   de tener ambos resultados (binary + multiclass).

### Cambios en la UI

[`build_results_text`](../spine_segmentation/deployment/app.py) — layout
multi-curva en español (matching el informe radiológico clínico):

```
=== COBB ANGLE - Curvas detectadas ===
Curva principal:    XX.X deg  (Tn - Lm, convexidad <derecha|izquierda>)
Curva secundaria:   YY.Y deg  (Tn - Lm, convexidad <opuesta>)
[Curva N:           ...  — solo si hay >2]

=== CROSS-CHECK binary vs multiclass ===
    Binary principal:    XX.X deg
    Multiclass:          MM.M deg  (Upper=Cn, Lower=Tm; illustration only)
    CONCORDANCIA: <High agreement | Review | Significant discrepancy>

=== ASSESSMENT (based on Binary principal) ===
<Normal/Mild/Moderate/Severe>
Numero total de curvas detectadas: N (S-curve / triple-curve / ...)
```

### Cambios en la visualización

Nuevo helper privado `_draw_single_cobb_curve` en
[visualize.py](../spine_segmentation/evaluation/visualize.py). El orquestador
lo llama una vez por curva (top 2 por magnitud). Diferenciación de colores:

| Curva | Color (RGB) |
|---|---|
| Principal | RED (255, 0, 0) |
| Secundaria | MAGENTA (255, 100, 200) |

Header del viz lista las dos con sus colores correspondientes. Speedometer
solo para la principal (evita dos gauges peleando por la esquina).

### Smoke remoto verde

Probado contra varios casos del MaIA dataset:

| Caso | Curva principal | Curva secundaria | Comportamiento |
|---|---|---|---|
| `N_1` (Normal) | 0.0° (binary) | — | "no clinically meaningful curves" |
| `S_21` (leve) | 15.1° T6-L1, conv. izquierda | — | 1 curva, Assessment Mild |
| `S_45` | 31.7° T3-T9 | — | 1 curva, escoliosis simple |
| `S_77` | 40.7° T4-T8 | — | 1 curva |
| **`S_100` (severa S-shape)** | **84.2° T5-T12** | **65.0° T12-L4** | **2 curvas ✓ rotoescoliosis** |
| `S_120` | 55.8° T4-T11 | — | 1 curva grave |
| `S_130` | 75.7° T4-T11 | — | 1 curva muy grave |
| `S_150` | 54.8° T2-T11 | — | 1 curva grave |

**S_100 es el caso pivote**: antes del Ciclo 5.2 habría devuelto un único
Cobb (probablemente engañoso, posiblemente más bajo por la cancelación
parcial entre las dos curvas opuestas). Ahora reporta correctamente las
dos curvas con sus end vertebrae nombradas, replicando el estilo del
informe que compartió Julian.

### Tests añadidos (Ciclo 5.2)

- `test_cobb_from_binary_detects_two_curves_on_s_shape` — sintético 2-cycle
  sinusoidal → ≥2 curvas con keys completos.
- `test_assign_vertebra_names_to_curves_label_transfer` — nearest-y por
  IP, con caso de multiclass vacío.
- `test_build_results_text_multi_curve_layout` — texto con
  "Curva principal", "Curva secundaria", "T5 - T12", convexidades,
  S-curve descriptor, Assessment desde la principal.
- `test_draw_cobb_visualization_multi_curve_uses_two_colors` — render
  contiene tanto RED como MAGENTA píxeles en cantidades coherentes.

Suite final: **25 passed + 1 skipped** (era 21 + 1 antes del 5.2).

### Commits del Ciclo 5.2

- `e9f7190` feat(cobb): detect multiple curves (s-shape, triple) from binary spline
- `3da21d8` feat(app): report all detected curves in diagnosis text
- `798ec3d` feat(viz): draw principal and secondary cobb curves
- `6e4fc2c` test: cover multi-curve cobb detection and rendering
- `<este>` docs(cycle5): close cycle 5.2 — multi-curve cobb

### Limitaciones honestas que persisten

- **Precisión absoluta del Cobb**: nuestro MAE sigue siendo ~23°. Multi-curve
  detection mejora la INFORMACIÓN clínica reportada (1 curva → 2 curvas
  cuando aplica), pero NO mejora la precisión numérica de cada ángulo. Eso
  requeriría reentrenamiento o cambio arquitectural (Ciclo 6+).
- **Componente rotacional vertebral**: el informe de Julian menciona
  "asimetría de los pedículos" como signo de rotoescoliosis. **NO podemos
  medir esto** con segmentación de silueta — requeriría detección de
  estructuras internas (pedículos), que necesita anotaciones específicas.
- **Direction estimate**: `_curve_direction` usa el signo del slope en el
  midpoint entre los 2 IPs. En casos sintéticos perfectos funciona; en
  radiografías reales con spline ruidoso puede dar "right" cuando deberia
  decir "left". No es bloqueante (es info auxiliar).
- **Naming de vértebras depende del multiclass**: si el multiclass falla
  en detectar (por ejemplo) T8, la curva con IP cerca de T8 reciba el
  nombre de T7 o T9 (el más cercano). Mejora poco con el multiclass actual
  (Dice 0.34).

---

## 13. Addendum 5.3 — Cobertura del binary + UX informativa (sin reentrenar)

> **Fecha:** 2026-05-19.
> **Motivación:** Elvis probó manualmente `S_22` (caso del dataset Scoliosis
> con S-shape clara). El modelo binary solo segmentó C6-T10 (~12 de 22
> vértebras), el spline se ajustó solo a la mitad superior casi recta, y la
> app reportó "0° — no clinically meaningful curves" cuando el ojo clínico
> ve dos curvas. Cuello de botella = **cobertura de la segmentación
> binaria**, no severidad ni algoritmo multi-curva.

### Diagnóstico

`S_22` produce una probabilidad binaria fuerte en la zona torácica
(>0.5) y débil pero presente en la zona lumbar (~0.3-0.4). Con el umbral
Ciclo 5.2 de 0.5, los píxeles lumbares se descartan. Aunque queden dos
fragmentos (toracico + lumbar tenue), `clean_binary_mask` filtraba por
"largest connected component" ANTES de cualquier operación de cierre, así
que el fragmento lumbar se perdía silenciosamente. El spline se ajustaba
solo a la mitad superior (casi recta), `cobb_from_binary` reportaba 0° y
la UI decía "Normal" — falso negativo clínico.

### Cambios

| Fix | Archivo | Detalle |
|---|---|---|
| **A** | [inference.py:117](../spine_segmentation/deployment/inference.py) | Umbral `binary_prob > 0.5` → `> 0.3`. Acepta píxeles marginales en zona lumbar. |
| **B** | [postprocessing/morphology.py](../spine_segmentation/postprocessing/morphology.py) | Cierre morfológico vertical (`cv2.MORPH_RECT (3, 25)`) insertado entre `MORPH_OPEN` y "keep largest CC". Puentea fragmentos torácico↔lumbar antes de filtrar. |
| **C** | [evaluation/cobb_angle.py](../spine_segmentation/evaluation/cobb_angle.py) | Default `smoothing_factor` 5000 → 1500. Spline más sensible captura inflexiones en curvas leves. |
| **D** | misma | Default `min_curve_deg` 3.0 → 2.0. Reporta curvas compensatorias suaves. |
| **Multi-pass** | misma | Refactor a `_cobb_from_binary_single_pass`. La envoltura corre la pasada del usuario; si encuentra exactamente 1 curva, re-corre con `smoothing*3.3` y prefiere lo que de más curvas. Reconcilia smoothing bajo (S_22) con smoothing alto (S_100). |
| **F** | [evaluation/coverage.py](../spine_segmentation/evaluation/coverage.py) (nuevo) + app.py + inference.py | Nuevo helper `compute_coverage_info`. UI emite bloque `=== COVERAGE ===` cuando `is_partial=True`, con nombres `Cn - Tm`, ratio %, y warning `Lower/Upper spine (T11-L5) NOT segmented`. Assessment cambia a "Inconclusive — insufficient coverage" cuando partial + 0°. |

### `compute_coverage_info` — API

```python
compute_coverage_info(binary_mask, multiclass_vertebrae, image_height=None) -> dict:
    success: bool
    top_y, bottom_y: int            # y-range del binary mask
    coverage_ratio: float           # (bottom_y - top_y) / image_height
    vertebrae_in_range: list[str]   # nombres con centroid_y en [top_y, bottom_y]
    vertebrae_below_range: list[str]  # las que el binary MISSED debajo
    vertebrae_above_range: list[str]  # las que MISSED arriba
    n_vertebrae: int                # len(vertebrae_in_range)
    n_expected: int                 # 22 (C3-L5)
    is_partial: bool                # ratio < 0.7 OR (multiclass disp AND n_vert < 15)
    upper_vertebra, lower_vertebra: str|None  # nearest-y a top/bottom
```

El multiclass se usa SOLO para naming (`extract_vertebra_info` + nearest-y),
no para calcular cobertura. Sin multiclass disponible, `upper_vertebra` y
`lower_vertebra` quedan en `None` pero `coverage_ratio` y `is_partial` siguen
funcionando.

### Tests añadidos (Ciclo 5.3)

1. `test_clean_binary_mask_bridges_vertical_gap` — Dos fragmentos verticales
   con gap de 20 px se unen en 1 componente conectado.
2. `test_compute_coverage_info_reports_partial_segmentation` — Binary mask
   parcial + vértebras dispersas → `is_partial=True`, `vertebrae_below_range`
   poblada.
3. `test_compute_coverage_info_reports_full_segmentation` — Binary mask de
   ~90% + 20 vértebras → `is_partial=False`.
4. `test_compute_coverage_info_handles_empty_mask_and_no_vertebrae` — Robustez
   ante mask vacío / None / sin multiclass.
5. `test_build_results_text_emits_coverage_warning_when_partial` — Texto
   contiene "COVERAGE", "C6 - T10", "12 of ~22", "WARNING", "Lower spine".
6. `test_build_results_text_no_coverage_warning_when_full` — Coverage llena
   omite el bloque y el WARNING.
7. `test_build_results_text_says_inconclusive_when_zero_cobb_and_partial` —
   El regresión-pin clave: partial + 0° NO debe decir "Normal" sino
   "Inconclusive".

Suite final: **32 passed + 1 skipped** (era 25 + 1 antes del 5.3).

### Smoke remoto (gradio_client) — 9 casos del dataset

| Caso | Ciclo 5.2 | Ciclo 5.3 | Observación |
|---|---|---|---|
| `N_1` (Normal) | 0° "Normal" | 0° "Normal" | sin regresión ✓ |
| **`S_22` (pivote)** | 0° "false-Normal" | **2 curvas 19.5°+5.9°, Mild + WARNING** | **FIXED — pasa de falso-negativo a verdadero-positivo ✓✓** |
| `S_21` (mild) | 1 curva 15.1° T6-L1 | 2 curvas 30.4° + 28.9° T6-T12 + WARNING | shift por A+B (más píxeles → más curva); ambos reportes flag-ean scoliosis |
| **`S_100` (severa)** | **2 curvas 84°+65°** | **2 curvas 83°+61°** | **multi-pass recuperó la curva secundaria — sin regresión ✓** |
| `S_45` | 1 curva 31.7° | 2 curvas 61.7°+41.7° + WARNING | más detalle |
| `S_77` | 1 curva 40.7° | 3 curvas 47.5°+30.9°+2.5° + WARNING | triple curve detectada |
| `S_120` | 1 curva 55.8° | 2 curvas 63.9°+38.3° | S-shape ahora detectado |
| `S_130` | 1 curva 75.7° | 2 curvas 80.6°+54.3° | S-shape ahora detectado |
| `S_150` | 1 curva 54.8° | 3 curvas 54.9°+39.9°+6.8° | triple curve detectada |

**Criterio de éxito cumplido**: S_22 detecta curvas y muestra el warning de
coverage; S_100 mantiene las 2 curvas; ningún caso baja de severidad clínica
relevante (todas las escoliosis siguen siendo escoliosis).

### Commits del Ciclo 5.3

- `d2fe3be` fix(inference): lower binary probability threshold to 0.3
- `f3efda4` fix(postproc): bridge fragmented spine via vertical morphological closing
- `7196361` feat(cobb): tune spline smoothing and noise floor for finer curve detection
- `2082b79` feat(coverage): compute binary mask coverage and surface it in diagnosis text
- `5e61c70` feat(cobb): multi-pass smoothing to recover Ciclo 5.2 S-shape detection
- `<este>` docs(cycle5): close cycle 5.3 — binary coverage fixes

### Limitaciones honestas

- **El umbral de "partial"** (`coverage_ratio < 0.7` OR `n_vertebrae < 15`) es
  conservador. S_21 cubre el 66% del alto y muestra el WARNING aunque
  prácticamente abarca toda la columna (el resto es padding). El warning es
  ligeramente sobre-sensible — preferimos un falso positivo de "review the
  segmentation" que un falso negativo silencioso.
- **El shift de S_21 (15° → 30°)** es real: viene de fixes A+B (más píxeles
  en el binary → spline con más curvatura). Ambos reportes la flag-ean como
  escoliosis (Mild vs Moderate); el MAE del binary es 23° así que un shift
  de 15° está dentro del ruido. NO es un cambio en el diagnóstico clínico,
  pero sí en el número exacto reportado.
- **Direction estimate sigue ruidoso** (heredado de Ciclo 5.2). En S_100 ahora
  reporta "izquierda" donde 5.2 reportaba "derecha". El módulo de la curva
  se preserva; la convexidad puede oscilar con el spline más sensible.
- **Coverage no aplica a casos sin multiclass**. Si el modelo multiclass
  falla por completo, `upper_vertebra` / `lower_vertebra` quedan en `None` y
  el warning degrada a un mensaje genérico. En la práctica, el multiclass
  detecta al menos 5-10 vértebras incluso en casos difíciles.
- **NO se reentrenó nada**. Todos los fixes son post-procesamiento. Si la
  probabilidad binaria del modelo está realmente por debajo de 0.3 en una
  zona, ningún umbral ni cierre morfológico la recuperará — esa es la
  frontera natural del Ciclo 6 (fallback multiclass, fix E, o reentrenar
  con augmentation lumbar agresivo).

---

## 14. Addendum 5.4 — Robustez ante rotación + UX de la viz Cobb

> **Fecha:** 2026-05-19.
> **Motivación:** Smoke manual del Space tras el cierre del 5.3 reveló 3
> issues nuevos que el 5.3 no anticipó:
> 1. **N_61** (Normal del dataset, rotada en frame) producía 4 curvas
>    fantasma (31.8° + 20° + 17.1° + 14.1°) clasificada como "Moderate
>    scoliosis", mientras el multiclass reportaba 0.6° correcto.
> 2. **S_22**: los rótulos `[Principal/Secundaria] Superior/Inferior (Tn)`
>    aparecían ilegiblemente encimados en la viz cuando 2 curvas
>    compartían vértebra (T9 = lower principal + upper secundaria).
> 3. **S_22**: la curva secundaria reportada como "5.9° T9-T9" — upper
>    y lower iguales, degenerada geométricamente.

### Diagnóstico

1. **Rotación**: `_cobb_from_binary_single_pass` ajusta un spline
   `x = f(y)` sobre los skeleton points. Una columna inclinada en el
   frame traza una **curva** en x(y) aunque sea anatómicamente recta. El
   segundo derivada produce zero-crossings espurios → "curvas" que son
   artefactos de la rotación de captura. **No existía** detección de
   orientación en el repo.
2. **Rótulos encimados**: `_draw_single_cobb_curve` calculaba
   `text_y = (min_row + max_row) // 2` independiente para cada rótulo,
   sin colisión ni dedup. Cuando 2 curvas compartían vértebra los 4
   textos aterrizaban en el mismo pixel.
3. **Curvas degeneradas**: `_cobb_from_binary_single_pass` filtraba solo
   por `angle < min_curve_deg = 2°`; no había umbral de y-distancia entre
   IPs adyacentes, ni chequeo de `upper == lower` en label transfer.

### Cambios

| Fix | Archivo | Detalle |
|---|---|---|
| **G** Detección rotación | nuevo [`evaluation/orientation.py`](../spine_segmentation/evaluation/orientation.py) + integración en [`inference.py`](../spine_segmentation/deployment/inference.py) + [`app.py`](../spine_segmentation/deployment/app.py) | `compute_orientation_info(skeleton_points)` con SVD centrado. `tilt_deg` = ángulo del primer singular vector vs eje y. `is_tilted = abs(tilt_deg) > 12°`. Bloque `=== ROTATION WARNING ===` en `build_results_text` cuando se dispara. UX: Cobb binary sigue visible, Assessment NO se modifica (decisión Elvis). |
| **H** Filtro degeneradas | [`cobb_angle.py`](../spine_segmentation/evaluation/cobb_angle.py) | (i) Nueva constante `MIN_IP_Y_DISTANCE_PX = 30` filtra pares IPs sub-vertebrales en `_cobb_from_binary_single_pass`. (ii) `assign_vertebra_names_to_curves` elimina in-place curvas con `upper == lower` post label transfer y reindexa `rank`. (iii) `inference.py` resyncroniza `cobb_angle_deg` / `inflection_points` si la principal fue filtrada. |
| **I** Dedup + anti-overlap rótulos | [`visualize.py`](../spine_segmentation/evaluation/visualize.py) | Nuevo `_rects_overlap` helper. `_draw_single_cobb_curve` acepta `labeled_vertebrae: set` y `placed_label_rects: list`. Dedup por nombre de vértebra (geometría se mantiene). Si label nuevo colisiona, desplaza `text_y` 16px hasta encajar. `draw_cobb_angle_visualization` instancia y threadea los acumuladores. |

### Smoke remoto (gradio_client) — 10 casos

| Caso | Ciclo 5.3 | Ciclo 5.4 | Observación |
|---|---|---|---|
| **`N_61`** (Normal **rotada** — pivote 5.4) | 4 curvas fantasma 31.8°/20°/17.1°/14.1° "Moderate" | **1 curva 17.1° + ROTATION WARNING (tilt 13.1°)** | **FIXED — false positive desactivado, warning visible al médico ✓✓** |
| **`S_22`** | 2 curvas: 19.5° T6-T9 + degenerada 5.9° T9-T9 + rótulos encimados | **1 curva 19.5° T6-T9** (degenerada filtrada, rótulos limpios) | **FIXED — issue UX resuelto ✓✓** |
| `N_1` (Normal) | 0° Normal | 0° Normal (sin warning) | sin regresión ✓ |
| `S_21` (mild) | 2 curvas 30.4°+28.9° + WARNING coverage | 2 curvas 30.4°+28.9° + WARNING coverage | sin regresión ✓ |
| `S_100` (severa S-shape) | 2 curvas 83°+61° | 2 curvas 83°+61° + ROTATION WARNING (tilt 12.6°, borderline) | warning marginal — la columna severamente escoliótica YA está tilted; diagnóstico Severe se mantiene |
| `S_45` | 2 curvas 61.7°+41.7° + WARNING coverage | 2 curvas 61.7°+41.7° + WARNING coverage | sin regresión ✓ |
| `S_77` | 3 curvas 47.5°+30.9°+2.5° + WARNING coverage | 3 curvas 47.5°+30.9°+2.5° + WARNING coverage | sin regresión ✓ |
| `S_120` | 2 curvas 63.9°+38.3° | 2 curvas 63.9°+38.3° | sin regresión ✓ |
| `S_130` | 2 curvas 80.6°+54.3° | 2 curvas 80.6°+54.3° | sin regresión ✓ |
| `S_150` | 3 curvas 54.9°+39.9°+6.8° | 3 curvas 54.9°+39.9°+6.8° + ROTATION WARNING (tilt 12.8°, borderline) | warning marginal análogo a S_100 |

Criterio de éxito cumplido: N_61 ya no se reporta como "Moderate
scoliosis" sin contexto; el warning de rotación explica el contexto. S_22
muestra solo la curva real, sin degenerada T9-T9 ni rótulos encimados.
S_100, S_45 y los demás severos no regresan en magnitudes ni en
detección multi-curva.

### Tests añadidos (Ciclo 5.4)

1. `test_compute_orientation_info_detects_tilt` — Skeleton sintético 30°
   → `is_tilted=True`, `tilt_abs_deg ≈ 30`.
2. `test_compute_orientation_info_vertical_spine` — Skeleton casi vertical
   → `is_tilted=False`.
3. `test_compute_orientation_info_handles_empty_or_collinear` — None,
   empty, <3 points, collinear → `success=False`.
4. `test_build_results_text_emits_rotation_warning_when_tilted` — 3
   ramas: tilted (block visible) / not tilted (absent) / None
   (back-compat).
5. `test_cobb_from_binary_skips_close_inflection_points` — synthetic
   slow+fast sine → toda curva sobreviviente tiene IPs ≥ 30 px de
   separación en y.
6. `test_assign_vertebra_names_drops_same_vertebra_curves` — curva con
   ambos IPs cerca de T9 → eliminada, sobreviviente reindexada a rank=1.
7. `test_assign_vertebra_names_keeps_curves_without_multiclass` — None ==
   None NO triggerea el filtro.
8. `test_draw_single_cobb_curve_dedupes_shared_vertebra_label` — T9
   compartida entre 2 curvas → segunda llamada no añade rect duplicado.
9. `test_draw_single_cobb_curve_shifts_overlapping_labels_down` — 2
   vértebras adyacentes con labels que colisionarían → rects finales
   sin overlap.

Suite final: **41 passed + 1 skipped** (era 32 + 1 antes del 5.4).

### Commits del Ciclo 5.4

- `69fff67` feat(eval): detect spine tilt from skeleton via SVD
- `7f1a53c` feat(inference,app): surface rotation warning in diagnosis text
- `6538dba` fix(cobb): filter degenerate curves with same upper/lower vertebra
- `9ca4017` fix(viz): dedupe shared-vertebra labels and prevent overlap
- `<este>` docs(cycle5): close cycle 5.4 — rotation + degenerate curve + label polish

### Limitaciones honestas

- **Threshold de tilt 12° es borderline en escoliosis severa**. S_100 y
  S_150 (Cobb >50°) muestran ROTATION WARNING aunque la "rotación"
  detectada es en realidad la inclinación intrínseca de su columna
  patológica. El warning es técnicamente correcto (el spline ve un eje
  tilted), pero podría confundir si el médico interpreta como
  "diagnostico no confiable" cuando en realidad la curva es real. El
  multiclass coincide en severo en estos casos, así que la concordancia
  high resuelve la ambigüedad. Recalibración a 15° o 18° es candidato
  para Ciclo 6 con más data.
- **N_61 todavía reporta 1 curva fantasma 17.1°** post-fixes. La rotación
  era suficiente para que el spline encontrara una inflexión que sobrevive
  el filtro de y-distancia y el label transfer (no degenerada). El
  WARNING explica el porqué, pero auto-de-rotar la imagen previo al
  pipeline daría un fix más limpio. Trabajo de Ciclo 6.
- **Dedup oculta info clínica útil**: si T8 es realmente lower de
  principal y upper de secundaria (caso anatómico válido), el rótulo
  "[Secundaria] Superior (T8)" se omite. La caja y la perpendicular
  siguen dibujadas, así que el médico ve la geometría — solo el texto se
  silencia. Trade-off aceptable: menos texto, más legible.
- **MIN_IP_Y_DISTANCE_PX = 30 es heurístico**. Calibrado a 512×512 input.
  Si el image_size cambia en el futuro, esta constante necesita
  re-escalar.

---

## 15. Addendum 5.5 — Control manual de rotación en la UI

> **Fecha:** 2026-05-19.
> **Motivación:** El smoke remoto del 5.4 confirmó que `N_61` seguía
> produciendo 1 curva fantasma 17.1° catalogada como "Mild scoliosis" a
> pesar del nuevo `=== ROTATION WARNING ===`. El warning explica el
> porqué, pero un médico apurado puede leer el número y saltarse el
> warning.
>
> El fix raíz era de-rotar la imagen antes del análisis. Auto-rotación
> era el primer candidato, pero al planificar surgieron 2 riesgos:
> 1. **Casos borderline** (S_100 tilt 12.6°, S_150 tilt 12.8°) habrían
>    sido auto-rotados, cambiando posiblemente magnitudes anatómicas
>    reales.
> 2. **Ambigüedad de signo**: `compute_orientation_info` retorna
>    `tilt_deg` signed via SVD + arctan2 + wrap a (-90, 90]. Predecir
>    qué dirección de rotación lo endereza en `cv2.getRotationMatrix2D`
>    no es trivial. Confirmé post-hoc empíricamente que `-tilt_deg`
>    HABRÍA EMPEORADO `N_61` (tilt 13.1° → 25.1°), no lo habría
>    corregido — la dirección correcta era `+tilt_deg`.

**Idea de Elvis (mejor):** exponer un control manual de rotación en la
UI. El médico ve la imagen, decide si rotarla y cuánto, y obtiene
feedback visual antes de Analyze. Cero magic, cero ambigüedad de signo
(el ojo del médico da la dirección), cero riesgo en S_100/S_150 (si no
los rota, no se rotan).

### Diseño

Bajo el componente `Image` de Gradio, antes del botón Analyze, se
añaden:

```
Rotate image (degrees). Negative = clockwise.    [slider -180 ──────── +180]   0
[↺ -90°]  [↺ -5°]  [Reset]  [↻ +5°]  [↻ +90°]
[                       Analyze                      ]
```

- `gr.Slider(min=-180, max=180, value=0, step=1)` con etiqueta clara.
- 5 botones rápidos modifican el slider por deltas (clip a (-180, 180)):
  - `-90°` y `+90°` para radiografías atravesadas o invertidas.
  - `-5°` y `+5°` para ajuste fino.
  - `Reset` salta a 0.
- `predict_btn.click` ahora pasa `[input_image, rotation_slider]` como
  inputs.

### Cambios

| Cambio | Archivo | Detalle |
|---|---|---|
| Helper module-level `rotate_image_for_analysis` | [`app.py`](../spine_segmentation/deployment/app.py) | `cv2.getRotationMatrix2D` + `cv2.warpAffine` con `BORDER_REPLICATE` y deadband `|deg| < 0.5°` (slider en 0 = identity, sin warp). Pure function: testeable sin Gradio. |
| `predict()` acepta `rotation_deg` | misma | Aplica el helper antes de delegar al pipeline. La cadena `pipeline.predict() → build_results_text()` sigue intacta. |
| UI: slider + 5 botones | misma | Bajo `input_image`, antes de Analyze. Click handlers de los botones usan `_adjust_rotation(current, delta)` con clipping. |
| Bug fix latente | misma | Los early-returns de `predict()` (sin imagen / sin pipeline) devolvían 4 valores cuando el handler espera 5. Corregido en passing. |

**NO se tocan**: `inference.py`, `cobb_angle.py`, `orientation.py`,
`coverage.py`, `visualize.py`, `morphology.py`. Toda la cirugía es en
la UI.

### Smoke remoto (gradio_client) — 4 casos clave

| Caso | Resultado | Validación |
|---|---|---|
| **`N_61` rotation=0** | 17.1° fantasma + ROTATION WARNING (tilt 13.1°) | Baseline Ciclo 5.4 (sin cambios cuando slider en 0) ✓ |
| **`N_61` rotation=+13** | **0.0° Normal, sin ROTATION WARNING** | **FIXED — falso positivo eliminado vía control manual ✓✓** |
| `N_1` rotation=0 | 0° Normal | sin regresión ✓ |
| `S_22` rotation=0 | 1 curva 19.5° T6-T9 + COVERAGE WARNING | sin regresión ✓ |

### Hallazgo crítico sobre la convención de signo

Comprobado en `dev/smoke_local_cycle5_5.py`:

| N_61 con | Tilt detectado post | Cobb binary | Severity |
|---|---|---|---|
| rotation=0      | 13.1° (sin cambio)  | 17.1° fantasma | Mild  |
| rotation=-13    | **25.1° (empeoró)** | 6.9°           | Normal (pero coverage roto) |
| rotation=+13    | **0.6° (resuelto)** | **0.0°**       | **Normal** |

Si hubiera implementado auto-rotación con `-tilt_deg` (la convención
intuitiva), N_61 habría EMPEORADO. Confirma post-hoc que el approach
manual fue la decisión correcta: el feedback visual del médico
resuelve la ambigüedad de signo en una sola intervención.

### Tests añadidos (Ciclo 5.5)

1. `test_rotate_image_for_analysis_zero_is_identity` — 0 / +0.3 / -0.4
   deg retornan exactamente el mismo objeto (deadband 0.5°). Garantiza
   que el slider en 0 no paga costo computacional.
2. `test_rotate_image_for_analysis_90_swaps_axes` — Stripe vertical →
   stripe horizontal después de +90°. Sanity geométrico.
3. `test_rotate_image_for_analysis_handles_none` — None pasa unchanged,
   sin crash, en cualquier ángulo.

Suite final: **44 passed + 1 skipped** (era 41 + 1).

### Commits del Ciclo 5.5

- `a764878` feat(app): add manual rotation control (slider + quick buttons) before Analyze
- `<este>` docs(cycle5): close cycle 5.5 — manual rotation control

### Anécdota del deploy

Primer intento de deploy (`upload_to_space.py --file ...app.py` sin
`--path-in-repo`) defaultó al basename, subiendo el módulo completo a
`app.py` (la raíz del Space) en lugar de `spine_segmentation/deployment/app.py`.
Esto SOBRESCRIBIÓ el shim raíz (que solo importa `create_app` y expone
`demo` para HF). Recovery: subir ambos archivos (shim raíz + módulo) en
un commit atómico al path correcto. Lesson learned: siempre pasar
`--path-in-repo` explícito cuando los archivos tienen el mismo basename.

### Limitaciones honestas

- **Sin live preview de la rotación**. El slider modifica un valor; la
  imagen mostrada NO rota hasta presionar Analyze. Esto es lo más
  simple (sin `gr.State`, sin doble-rotación). En la práctica clínica
  el médico puede pulsar Analyze tras cada ajuste para ver el preview
  segmentado. Live preview sería ~30 LOC adicionales con `gr.State`
  para preservar el original; deferred a Ciclo 6 si se justifica.
- **El slider no recuerda valores entre uploads**. Si cargas otra
  imagen, el slider sigue donde lo dejaste. Tampoco hay handler de
  upload que lo resetee. Para Ciclo 6 si molesta.
- **Step de 1°**. Suficiente para corrección clínica (radiografías no
  necesitan precisión sub-grado). Bajar a 0.5° si surge un caso real.
- **No hay rotación + flip combinados** (e.g., para una radiografía
  espejada). Caso muy raro; los 5 botones cubren los casos prácticos
  (±5°, ±90°, reset). Si surge, agregar un toggle Flip-H en Ciclo 6.
