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
> **Addendum 5.6** (2026-05-19): live preview de la rotación. Ver sección 16.
> **Addendum 5.7** (2026-05-19): limpieza multiclass del frontend + toggle ES/EN. Ver sección 17.
> **Addendum 5.8** (2026-05-20): polish del tab Explainability. Ver sección 18.
> **Addendum 5.9** (2026-05-20): imagen fija de referencia clínica bilingüe en Explainability. Ver sección 19.
> **Addendum 5.10** (2026-05-20): fix de convención de lateralidad (anatomía del paciente) + sample S_200. Ver sección 20.
> **Addendum 5.11** (2026-05-20): fix de arrows del reference image (sample-invariant). Ver sección 21.
> **Addendum 5.12** (2026-05-22): fix coord centering (aspect='equal') + DECISIONS.md + Gradio FileNotFound known-issue. Ver sección 22.
> **Addendum 6.1** (2026-05-22): fix de lateralidad por chord signed-area (post-sustentación). Ver sección 23.

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

---

## 16. Addendum 5.6 — Live preview de la rotación

> **Fecha:** 2026-05-19.
> **Motivación:** El slider del 5.5 actualizaba un número pero la
> imagen mostrada NO rotaba hasta presionar Analyze (10s en CPU). Cita
> de Elvis (paráfrasis): "la idea es que te muestre cómo queda al rotar
> la imagen para que él pueda decidir cómo dejarla y ahí sí hacer el
> análisis". Sin feedback visual, dialing el ángulo correcto requería
> múltiples ciclos de Analyze, cada uno al costo completo de inferencia.

### Flujo antes vs después

**Antes (Ciclo 5.5):**
1. Mover slider → solo cambia un valor numérico.
2. Presionar Analyze (10s CPU).
3. Ver resultado.
4. Si la rotación quedó mal, repetir.

**Después (Ciclo 5.6):**
1. Mover slider → **la imagen rota en vivo (~50-150ms)**.
2. Ajustar hasta que la columna se vea vertical.
3. **Un solo Analyze al final** (10s, una vez).

### Pieza nueva clave: `gr.State` para imagen original

Sin un state separado, el preview rotaría la imagen MOSTRADA (que ya
puede ser producto del cambio anterior del slider), acumulando
rotaciones cuando el usuario arrastra el slider por valores
intermedios. La solución estándar de Gradio:

```python
original_image_state = gr.State(value=None)
```

- `input_image.upload(...)` guarda la imagen recién subida en este
  state y resetea el slider a 0. Solo dispara para uploads de usuario
  (no para escrituras programáticas del preview pipeline).
- `rotation_slider.change(...)` rota la **original** (no la mostrada)
  por el valor actual del slider, escribiendo el resultado en
  `input_image`.

### Cambios

| Cambio | Detalle |
|---|---|
| Helper module-level `preview_rotation_for_display(original, deg)` | Delega a `rotate_image_for_analysis`. Retorna None cuando original es None (slider movido antes de cualquier upload). |
| `gr.State(value=None)` para imagen original | Stash en upload + reset slider a 0. |
| `rotation_slider.change` handler | Rota original → escribe en `input_image`. |
| 5 botones rápidos atómicos | Cada uno retorna `(new_slider, rotated_image)` en un handler único. Más robusto que esperar a que `slider.change` dispare después de update programático. |
| Reset button | Retorna `(0.0, original_unrotated)` — slider Y display vuelven a estado uploaded. |
| `predict()` pierde `rotation_deg` | Por que `input_image` al momento de Analyze ya está rotada visualmente. Predict solo delega al pipeline. Cero riesgo de doble-rotación. |
| `predict_btn.click` inputs | Simplificado a `[input_image]` (no slider). |
| `rotate_image_for_analysis` | Se queda como helper puro; ahora también lo usa `preview_rotation_for_display`. |

### Tests añadidos (Ciclo 5.6)

1. `test_preview_rotation_for_display_handles_none` — None original a
   cualquier ángulo retorna None sin crash. Cubre "slider movido antes
   de upload".
2. `test_preview_rotation_for_display_returns_rotated` — Imagen
   sintética + 90° → resultado distinto al original. Sanity de que
   delega al helper.
3. `test_predict_callback_no_longer_takes_rotation_deg` — Regression
   pin contra accidentalmente poner `rotation_deg` de vuelta en el
   closure. Verifica que la signature del callback es de 1 argumento.

Los 3 tests del 5.5 sobre `rotate_image_for_analysis` siguen pasando —
el helper no cambió.

Suite final: **47 passed + 1 skipped** (era 44 + 1).

### Smoke remoto

Sin live preview testeable via `gradio_client` (es UI-only). Lo que sí
se testea via API:

| Caso | rotation slider | Resultado | Validación |
|---|---|---|---|
| `N_61` | 0 (sin tocar) | 17.1° fantasma + ROTATION WARNING | igual al 5.5 ✓ |
| `N_1`  | 0 (sin tocar) | 0° Normal                          | igual al 5.5 ✓ |
| `S_22` | 0 (sin tocar) | 1 curva 19.5° + COVERAGE WARNING   | igual al 5.5 ✓ |

Cero regresión a la ruta del Analyze. El live preview se valida
manualmente en browser:

- Subir N_61 → slider en 0 → spine se ve tilted.
- Mover slider a +13° lentamente → spine rota en vivo y queda vertical
  (visible en <300ms en CPU basic free).
- Presionar Analyze → "0° Normal" en menos de un round-trip de prueba-y-error.

### Commits del Ciclo 5.6

- `ca0989e` feat(app): live preview of rotation — slider drags rotate the image in-place
- `<este>` docs(cycle5): close cycle 5.6 — live rotation preview

### Limitaciones honestas

- **Performance del preview en imágenes muy grandes** (~3000×4000):
  `cv2.warpAffine` puede tomar 100-200ms + round-trip http. En CPU
  basic free, podría sentirse menos fluido. Si surge, cambiar
  `rotation_slider.change` → `rotation_slider.release` (preview solo al
  soltar). No es bloqueante en imágenes típicas del dataset MaIA.
- **`gr.State` se mantiene en el cliente del navegador**: si el usuario
  recarga la página, pierde el state. Pero también pierde el upload
  (input_image se limpia). Comportamiento consistente.
- **Si el usuario sube una imagen muy diferente sin presionar Reset**:
  el upload handler resetea el slider a 0 automáticamente, evitando que
  la nueva imagen aparezca rotada por el valor previo. Probado y
  funcional.
- **Sin live preview de la segmentación**: Cobb/overlays siguen
  apareciendo solo después de Analyze. Eso es por diseño — la
  segmentación es el cómputo caro (10s). El preview solo afecta la
  decisión "¿cuánto rotar?", no "¿cuál es el diagnóstico?".

---

## 17. Addendum 5.7 — Limpieza multiclass del frontend + toggle ES/EN

> **Fecha:** 2026-05-19.
> **Motivación:** Tras probar el live preview del 5.6, Elvis identificó
> dos issues simultáneos:
> 1. "La info del multiclass no es bueno que aparezca al final, el
>    binary es el que tiene mayor peso... no le aporta al usuario y lo
>    termina confundiendo". El bloque `=== CROSS-CHECK ===` mostraba
>    "Multiclass: 90.0 deg" (a menudo degenerado por el clamp del arctan
>    sobre el multiclass ruidoso, Dice 0.34) al lado del binary
>    coherente (e.g., 4.2°). Sin contexto algorítmico, parecían
>    contradictorios.
> 2. La UI era inconsistente: el reporte estaba en español ("Curva
>    principal", "convexidad derecha") pero el header markdown y los
>    tabs en inglés. Solicitó toggle ES/EN para usuarios de ambos
>    idiomas.

### Parte 1 — Multiclass cleanup del frontend (Fix M)

**Eliminado de la UI** (sin perder funcionalidad backstage):

- `build_results_text` ya no emite el bloque `=== CROSS-CHECK binary vs
  multiclass ===` (con sus 3 líneas Binary/Multiclass/CONCORDANCIA).
- `draw_cobb_angle_visualization` ya no dibuja la línea cyan
  "Multiclass (illustration only): X.X deg" en el header.

**Conservado** (uso interno del multiclass):

- `assign_vertebra_names_to_curves`: el label transfer Tn-Lm para
  nombrar end vertebrae de las curvas binary sigue funcionando
  silenciosamente.
- Cajas verdes en end vertebrae: se siguen dibujando con datos del
  multiclass mask.
- Lista `=== VERTEBRAS DETECTADAS ===` (en ES) / `=== VERTEBRAE
  DETECTED ===` (en EN): se mantiene — informa cuántas vértebras
  individuales detectó el modelo.
- Tab "Vertebrae Segmentation": viewer del multiclass mask coloreado,
  útil educativamente, intacto.
- Multi-fallback line: cuando el binary FALLA totalmente (e.g., empty
  mask), el multiclass se vuelve a mostrar como `Multiclass method
  (fallback): X.X deg` porque allí sí es la única señal disponible.

### Parte 2 — i18n module + toggle ES/EN (Fix N, Nivel B)

**Nuevo módulo** `spine_segmentation/deployment/i18n.py`:

```python
DEFAULT_LANG = "es"  # default audience = U. Andes / Colombia
DIAGNOSIS_STRINGS = {"es": {...}, "en": {...}}     # all UI strings
MARKDOWN_HEADER  = {"es": "...", "en": "..."}      # top-of-page intro

def t(key, lang="es"): ...
def header_markdown(lang="es"): ...
def label_to_lang(label): ...  # "Español"/"English" -> "es"/"en"
```

`t()` cae a Español si el lang es desconocido y al key si la clave no
existe (placeholder visible para regresiones de traducción durante
desarrollo).

**Refactor `build_results_text`**:

- Nuevo parámetro `language: str = "es"`.
- TODOS los strings de UI ahora vienen de `t(key, lang)`: `cobb_block_
  header_curves`, `principal_label`, `convex_right/left/neutral/unknown`,
  `binary_method`, `binary_no_curves`, `multi_fallback_prefix`,
  `coverage_header`, `coverage_covers`, `coverage_of`,
  `coverage_vertebrae_word`, `warning_lower_spine_prefix`,
  `warning_not_segmented_suffix`, `warning_partial_generic`,
  `rotation_header`, `rotation_line_{1,2,3}`,
  `assessment_header_{principal,binary,fallback}`,
  `assessment_{normal,mild,moderate,severe,inconclusive}`,
  `curves_total_prefix`, `shape_{double,triple,n_curves}`,
  `vertebrae_header_prefix`.

**UI (Gradio)**:

```python
language_radio = gr.Radio(
    choices=["Español", "English"],
    value="Español",
    label="Idioma / Language",
    interactive=True,
)
header_md = gr.Markdown(header_markdown(DEFAULT_LANG))
# ...
predict_btn.click(predict, [input_image, language_radio], [...])
language_radio.change(
    fn=lambda lbl: header_markdown(label_to_lang(lbl)),
    inputs=[language_radio],
    outputs=[header_md],
)
```

El header markdown se re-renderiza al instante al togglear. El reporte
de diagnóstico se re-traduce cuando el usuario presiona Analyze otra
vez (no se re-ejecuta el modelo solo para retraducir).

**Texto del ROTATION WARNING actualizado**: la línea final ahora dice
"Re-capture with the patient straight, or use the rotation slider to
straighten the image" (en lugar del obsoleto "trust the multiclass
measurement" que describía el path de auto-rotación que abandonamos en
el 5.5).

### Tests añadidos (Ciclo 5.7)

1. `test_t_returns_spanish_by_default` — sin lang, ES.
2. `test_t_returns_english_when_lang_is_en` — `lang='en'`, EN.
3. `test_t_falls_back_to_key_when_key_missing` — key returned visible.
4. `test_t_falls_back_to_spanish_when_lang_unknown` — defensa.
5. `test_label_to_lang_maps_radio_choices` — labels Gradio → códigos.
6. `test_build_results_text_renders_in_spanish` — verifica strings ES.
7. `test_build_results_text_renders_in_english` — verifica strings EN.
8. `test_header_markdown_has_both_languages` — sanity de ambos blocks.

Además, los 6 tests de Ciclos previos que asercionaban strings en
inglés (e.g. "Mild scoliosis", "CONCORDANCIA") ahora pasan
`language='en'` explícito o `language='es'` según corresponda — la
intención original se preserva.

Y el test `test_build_results_text_includes_concordance_when_both_succeed`
del 5.5 se transformó a `test_build_results_text_no_longer_emits_cross_check_block`
(regression pin de Fix M).

Suite final: **55 passed + 1 skipped** (era 47 + 1).

### Smoke remoto (gradio_client) — verificación bilingüe

| Test | Resultado |
|---|---|
| API signature | `input_image` + `language_label: enum["Español","English"]` ✓ |
| S_22 con `language="Español"` | "ANGULO COBB", "Curva principal: 19.5 deg", "COBERTURA", "Escoliosis leve", "VERTEBRAS DETECTADAS" ✓ |
| S_22 con `language="English"` | "COBB ANGLE", "Principal curve: 19.5 deg", "COVERAGE", "Mild scoliosis", "VERTEBRAE DETECTED" ✓ |
| Ningún output contiene "CROSS-CHECK" | ✓ |
| Ningún output contiene "Multiclass:" en ambos idiomas | ✓ |

### Commits del Ciclo 5.7

- `0d344f3` refactor(app,viz): remove multiclass cross-check from user-facing UI
- `28872b8` feat(app): add i18n module + ES/EN language toggle for diagnosis results
- `<este>` docs(cycle5): close cycle 5.7 — multiclass UI cleanup + bilingual report

### Limitaciones honestas

- **Tabs / slider / botones rápidos siguen en inglés**. Cambiar labels
  de componentes Gradio dinámicamente requiere recrear el Blocks. Para
  símbolos universales (Reset, Analyze, ↺ -90°, "Binary Segmentation")
  el valor de traducir no justifica la complejidad. Si surge necesidad,
  promover a Ciclo 6 con full UI re-render.
- **El reporte no se re-traduce automáticamente al togglear el
  idioma** — el usuario debe presionar Analyze otra vez. Es por diseño:
  el modelo cuesta 10s en CPU; ejecutarlo solo para retraducir strings
  fijos sería desperdicio. El header markdown sí cambia instantáneamente
  porque es texto fijo sin cómputo.
- **Default Español puede sorprender** a un evaluador anglófono que
  abre el Space por primera vez. Mitigación: el radio "Idioma /
  Language" es bilingüe, visible al tope, y un click resuelve.
- **i18n basado en dict no es escalable** a más de 2-3 idiomas. Para
  expansión futura, considerar `gettext` o un sistema de archivos `.po`
  por idioma. Para 2 idiomas el dict es lo más simple y mantenible.
- **Multi-fallback line cuando binary FALLA** se conserva. Si en
  producción aparecen casos donde el binary falla pero el multiclass
  reporta un Cobb decente, el usuario sí ve un número multiclass — pero
  con el prefijo "fallback" claro y el binary marcado como ERROR, no
  como contradicción.

---

## 18. Addendum 5.8 — Polish del tab Explainability (Grad-CAM + Confidence)

> **Fecha:** 2026-05-20.
> **Motivación:** Tras el push del 5.7, Elvis probó la pestaña
> Explainability y reportó: "las zonas que marca no son... o no son
> claras... la idea es que marque bien las zonas... y adicional el mapa
> de confianza de una mejor lectura, de más claridad para que el usuario
> sepa cómo leer la info".

### Diagnóstico

Tres issues distintos en el panel side-by-side:

1. **Grad-CAM pintaba toda la imagen** (incluyendo zonas fuera de la
   columna detectada). Sin masking, las activaciones del último conv
   layer del encoder cubrían 512×512 completo, y muchas regiones
   "calientes" estaban en costillas, pelvis o fondo — no en vértebras.
2. **Mapa de confianza con cmap RdYlGn sin masking** pintaba el fondo
   en rojo intenso (porque 0 → rojo en esa cmap). Visualmente parecía
   "baja confianza en todo el fondo", cuando en realidad esas zonas NO
   se evaluaron. El médico no podía distinguir "no analizado" de
   "analizado y con baja confianza".
3. **Sin títulos ni colorbars** en los subpaneles. Cada uno era una
   imagen 512×512 plana sin indicación de qué representaba la escala de
   color. El Markdown debajo explicaba con texto, pero requería leer
   antes de mirar.

### Cambios

| Fix | Archivo | Detalle |
|---|---|---|
| **O** Mask Grad-CAM + Confidence por `binary_mask` | [`explainability.py`](../spine_segmentation/evaluation/explainability.py) + [`app.py`](../spine_segmentation/deployment/app.py) | `generate_gradcam` y `generate_confidence_map` aceptan parámetro opcional `prediction_mask`. Cuando se pasa, multiplican el output por el mask (fuera = 0). El render en app.py mezcla con la imagen original (`cv2.where(outside, original, colored)`) → el fondo es la radiografía en grises, no negro plano. |
| **R** Percentile clip p95 en Grad-CAM | misma | Tras masking, `np.clip(cam / np.percentile(cam[cam>0], 95), 0, 1)` renormaliza para que los hot-spots reales destaquen. Sin esto, outliers comprimían la escala. |
| **P** Anotaciones in-image | `app.py` | Nuevo helper `annotate_explainability_panel(cam, conf, language_label)` añade strip oscuro 32px arriba con título + colorbar vertical 18px a la derecha con etiquetas "Alta/Baja" o "High/Low". El panel final pasa de 1024×512 a **1100×544**. |
| **Q** Markdown bilingüe del tab Explainability | `i18n.py` + `app.py` | Nuevo `EXPLAIN_MARKDOWN` dict + `explain_markdown(lang)`. Wording mejorado con sección "Cómo leerlo / How to read it" que enseña qué es un resultado clínicamente confiable vs sospechoso. El handler `language_radio.change` ahora actualiza header markdown Y explain markdown en un solo round-trip. |

### Tests añadidos (Ciclo 5.8)

1. `test_annotate_explainability_panel_adds_titles_and_colorbars` —
   output más alto (header strip) + más ancho (colorbar margins) +
   pixels brillantes en el header (texto).
2. `test_annotate_explainability_panel_localizes_titles` — Tanto ES
   como EN producen pixels de título no-triviales.
3. `test_generate_gradcam_applies_prediction_mask` — Con `prediction_mask`
   suplido, pixels fuera del mask son 0; dentro son > 0.
4. `test_generate_confidence_map_applies_prediction_mask` — Mismo
   comportamiento para el confidence map.
5. `test_explain_markdown_has_both_languages` — ES y EN existen,
   difieren, y ambos contienen "Grad-CAM" + guía de lectura.

Suite final: **60 passed + 1 skipped** (era 55 + 1).

### Smoke remoto

Via `gradio_client.Client.predict(handle_file(path), 'Español')` y
`'English'`:

- Imagen explainability resultado: **1100 × 544** (vs 1024 × 512 antes
  del 5.8) — matemática: 2 × (512 + 18 colorbar + 20 márgenes) = 1100
  ancho, 512 + 32 header = 544 alto. ✓
- Funciona en ambos idiomas — sin crashes ni regresión.
- Visualmente en el browser: títulos legibles, colorbar visible,
  Grad-CAM/Confidence masked solo dentro del spine, fondo en grises.

### Commits del Ciclo 5.8

- `037d182` feat(explain): mask gradcam + confidence by predicted spine + annotate panel
- `62e5e80` feat(i18n,app): bilingual explainability panel description
- `<este>` docs(cycle5): close cycle 5.8 — explainability polish

### Limitaciones honestas

- **Grad-CAM target layer sigue siendo el último conv del encoder**.
  Cambiar a Seg-Grad-CAM "auténtico" requeriría agregar
  `pytorch_grad_cam.GradCAM.compute_cam_per_layer` con upsampling para
  el caso de segmentación densa — tema mayor, deferred a Ciclo 6 si
  surge necesidad.
- **El colorbar es decorativo**: no muestra valores numéricos exactos
  (solo "Alta/Baja"). Para una versión clínicamente más completa
  habría que añadir labels intermedios ("0.25", "0.5", "0.75"). Trade-
  off: con un colorbar de 18px de ancho, no caben más etiquetas
  legibles.
- **El Grad-CAM percentile-clip puede oscurecer hot-spots débiles**
  cuando todo el cam es de baja intensidad (e.g., en una imagen muy
  poco característica). Si surge un caso real donde no se ve nada,
  bajar el clip a p99 o quitarlo. No es bloqueante.
- **El masking depende de `binary_mask`**: si el binary mask falla por
  completo, el render queda con la imagen original sin overlay
  (degradación graceful). El multiclass mask no se usa aquí — solo el
  binary. Coherente con el resto del pipeline (Cobb se calcula desde
  el binary).

---

## 19. Addendum 5.9 — Imagen fija de referencia clínica en Explainability

> **Fecha:** 2026-05-20.
> **Motivación:** Tras el polish del 5.8 (masking + colorbars +
> markdown bilingüe), el médico colaborador de Elvis preparó un
> **mockup educativo** del panel Explainability con 5 anotaciones
> numeradas + colorbars con interpretación + disclaimer. Elvis quería
> dejarlo **fijo encima del panel dinámico** del Space, en ambos
> idiomas — para que cualquier médico que abra la app aprenda a leer
> Grad-CAM + Confidence ANTES de ver su propio caso. Es una mejora
> puramente pedagógica, no algorítmica.

### Diagnóstico

El panel dinámico del 5.8 ya es claro para alguien con contexto, pero
el primer usuario que abre la app sin haberlo visto antes:

1. No sabe que **rojo en Grad-CAM = alta influencia** (no necesariamente
   un hallazgo clínico).
2. Confunde **rojo en Confidence = baja confianza** con "alarma" (es
   indicación de revisar, no de patología).
3. No tiene referencia visual para distinguir "hot-spot legítimo en
   la columna" de "activación espuria fuera de la columna".

El markdown del 5.8 explica esto con texto, pero un médico apurado
salta lectura. Una imagen anotada con flechas es más eficiente
clínicamente.

### Cambios

| Fix | Archivo | Detalle |
|---|---|---|
| **S** Generador one-shot bilingüe | nuevo [`scripts/generate_explain_reference.py`](../scripts/generate_explain_reference.py) | Compone un panel side-by-side estilo mockup: sample radiograph (S_22) + binary mask del dataset → simulación de Grad-CAM (jet) y Confidence (RdYlGn) sobre la silueta del spine. Dibuja 5 callouts numerados con flechas leader + 2 colorbars centradas + caption + disclaimer footer. Acepta `--lang es|en|both`. No se invoca en runtime — solo para regenerar los assets. |
| **T** Assets bilingües committeados | nueva carpeta `spine_segmentation/deployment/assets/` | Dos PNGs (~165 KB cada uno, 1126×716): `explainability_reference_es.png` y `_en.png`. Subidos al Space via LFS automático del `HfApi`. |
| **U** Helper i18n | [`i18n.py`](../spine_segmentation/deployment/i18n.py) | Nuevo `EXPLAIN_REFERENCE_FILES` dict + `explain_reference_path(lang)` retorna path absoluto. Fallback a Español para lang desconocido. |
| **V** UI wiring | [`app.py`](../spine_segmentation/deployment/app.py) | Nuevo `reference_image = gr.Image(interactive=False, height=300)` ARRIBA del `explain_output` dinámico. Extender `_on_language_change` para que `language_radio.change` actualice también la `reference_image` en un solo round-trip (junto al header_md y explain_md del 5.7/5.8). |

### Restricciones técnicas resueltas durante el ciclo

- **`show_download_button` removido en Gradio 6.0**: el local dev corre
  Gradio 6.0 (donde el kwarg ya no existe), aunque el Space corre 5.50
  (donde sí existe). Para mantener `pytest tests/` verde en ambos
  entornos, se omite el kwarg y se acepta el default `True` — una
  imagen estática educativa es seguro que el usuario descargue.

### Tests añadidos (Ciclo 5.9)

1. `test_explain_reference_images_exist_and_are_nonempty` — ambos PNGs
   resuelven a un path existente con tamaño > 1 KB. Catch
   regression-pin contra commits accidentales de archivos vacíos o
   typos en el mapping del i18n.
2. `test_explain_reference_path_distinct_per_lang` — ES y EN devuelven
   paths distintos (sin esto, el toggle ES/EN sería un no-op
   silencioso) + unknown lang colapsa al default (ES).

Suite final: **62 passed + 1 skipped** (era 60 + 1).

### Smoke remoto

Tras el push al Space, verificación via `gradio_client` + HEAD HTTP:

| Test | Resultado |
|---|---|
| `runtime.stage` post-rebuild | `RUNNING` ✓ |
| HEAD `/spaces/ElvLandau/spine-segmentation` | HTTP 200 estable ✓ |
| Reference image visible al cargar el Space (default ES) | ✓ |
| Toggle a English en el radio → reference image cambia a `_en.png` | ✓ |
| Reference image arriba del panel dinámico (orden correcto) | ✓ |
| Sin regresión: predict() sigue devolviendo 4 overlays + texto | ✓ |

### Commits del Ciclo 5.9

- `71a5494` feat(assets): generate bilingual explainability reference images
- `a44c50f` feat(i18n,app): wire static reference image into Explainability tab with language toggle
- `5510a79` test(app): cover bilingual explainability reference assets
- `<este>` docs(cycle5): close cycle 5.9 — bilingual explainability reference image

### Decisión honesta documentada

La imagen original del médico colaborador no estaba disponible como
path de archivo en el sistema en el momento de generación — solo como
adjunto visual al chat. Los dos PNGs son **recreación programática del
mockup** con matplotlib + cv2, no copia binaria. Esto:

- ✅ Garantiza consistencia visual ES↔EN al 100% (mismo template).
- ✅ Es reproducible y versionable (el generador queda commiteado).
- ⚠️ Puede no coincidir píxel a píxel con el diseño original del médico.

Mitigación documentada en AGENTS.md sec 9: si Elvis quiere usar el PNG
exacto del médico, basta dejarlo en `assets/explainability_reference_es.png`
y re-correr `python scripts/generate_explain_reference.py --lang en`
para regenerar solo el EN con la misma plantilla.

### Limitaciones honestas

- **El sample radiograph es S_22**, que es un caso real de S-shape del
  dataset MaIA — visualmente representativo pero específico. Si en el
  futuro la app se especializa a una población distinta (e.g.,
  pediátricos con curvas distintas), el sample podría regenerarse con
  otro caso del dataset.
- **Las activaciones del Grad-CAM y los valores de confidence son
  simulados** (no salen del modelo real). El propósito es educativo —
  enseñar qué SIGNIFICAN los colores, no qué saldría en este caso
  exacto. Para una versión "anatómicamente exacta" habría que correr
  inferencia real del modelo deployment y congelar el resultado, lo
  cual añade complejidad sin upside pedagógico (el médico ya ve el
  resultado real del modelo en el panel dinámico debajo).
- **El generador depende del dataset MaIA local** (`Scoliosis/S_22.jpg`
  + `LabelBinaryJPG/Label_S_22.jpg`), que no está en git. Para
  regenerar en otra máquina sin el dataset, hay que ajustar los args
  `--sample-xray` y `--sample-mask` a otras rutas.
- **Anotaciones numéricas sobre los hot-spots del CAM real** (e.g.,
  "esta zona dispara X% de activación") quedan pendientes para el
  Ciclo 6 si Elvis lo prioriza tras probar — el plan original lo
  excluyó explícitamente como over-scope del 5.9.
- **El layout es fijo 1126×716**: si Elvis cambia el `height=300` de
  Gradio, la imagen se downscale-eará proporcionalmente; las
  proporciones del texto interno seguirán legibles porque las anotaciones
  son `fontsize 6-10pt` (representan ~12-20px en el render final).

---

## 20. Addendum 5.10 — Fix de convención de lateralidad clínica + sample S_200

> **Fecha:** 2026-05-20.
> **Motivación:** Feedback de la compañera médica (especialista, mensajes
> WhatsApp del 2026-05-20 8:30–8:47 PM) tras probar el Space deployed del
> Ciclo 5.9. Dos puntos:
>
> 1. **Bug clínico (alta prioridad).** Las radiografías AP siguen la regla
>    del espejo: el lado derecho del paciente aparece en el lado izquierdo
>    de la imagen para quien la mira. Los informes radiológicos SIEMPRE
>    describen lateralidad en anatomía del paciente. La compañera identificó
>    el caso `S_158.jpg`: la curva es anatómicamente right-convex pero la
>    app la reportaba como "convexidad izquierda". Cita: *"la encontré
>    mira por ejemplo esta la convexidad es derecha y se reporta como
>    izquierda."*
> 2. **UX educativo.** La compañera no reconoció la radiografía base del
>    reference image del Ciclo 5.9 (S_22) como un caso del dataset MaIA
>    — S_22 es modesto (Cobb 24.9°) y la columna se ve casi recta. Pidió
>    cambiarlo por un caso más demostrativo. Elvis eligió `S_200`.

### Diagnóstico del bug de lateralidad

El helper `_curve_direction` ([cobb_angle.py:58-73](../spine_segmentation/evaluation/cobb_angle.py))
opera sobre el spline `x = f(y)` (coordenadas de imagen, no anatomía).
El código anterior `return "right" if mid_slope < 0 else "left"`
asignaba labels en perspectiva del viewer:

- Pendiente negativa en el midpoint → la columna se desplaza a la
  izquierda del VIEWER al bajar → app reportaba "right".
- Pero la convención radiológica dice que "right" debe referirse a la
  derecha del PACIENTE = izquierda del viewer (regla del espejo).

Resultado: para el caso S_158 (anatómicamente right-convex), el
algoritmo computaba un mid_slope > 0 → return "left" → app mostraba
"convexidad izquierda". Exactamente el bug reportado.

### Cambios

| Fix | Archivo | Detalle |
|---|---|---|
| **W** Convención clínica | [`cobb_angle.py:_curve_direction`](../spine_segmentation/evaluation/cobb_angle.py) | Swap del ternario en línea 73: `return "left" if mid_slope < 0 else "right"`. Docstring reescrita documentando explícitamente la convención del espejo + referencia a este ciclo + regression test. Comentario inline también deja claro que el swap es intencional. NO se tocan `i18n.py`, `app.py`, ni `visualize.py` — los strings "derecha"/"izquierda" siguen funcionando, sólo cambia su SIGNIFICADO (anatomía del paciente, no perspectiva del viewer). |
| **X** Sample S_22 → S_200 | [`scripts/generate_explain_reference.py`](../scripts/generate_explain_reference.py) + 2 PNGs en `assets/` | Defaults `--sample-xray` y `--sample-mask` bumped a `S_200.jpg` y `Label_S_200.jpg` con comentario explicativo. Las 2 PNGs regeneradas (`~187 KB` ES, `~183 KB` EN). S_200 muestra una S-curve clínicamente clara, mejor para pedagogía. Layout invariante — sólo cambia la radiografía de fondo + simulación de overlays. |

### Tests añadidos (Ciclo 5.10)

1. `test_curve_direction_uses_patient_anatomy_convention` — Pinea el
   mapping post-fix:
   - `mid_slope < 0` → returns `"left"` (era "right" pre-Ciclo 5.10).
   - `mid_slope > 0` → returns `"right"` (era "left" pre-Ciclo 5.10).
   - `mid_slope ≈ 0` → `"neutral"` (sin cambio).
   - Bad indices → `"unknown"` (sin cambio).

   El test documenta la convención clínica en su docstring y referencia
   el caso `S_158` como evidencia clínica. Cualquier refactor futuro que
   re-invierta el ternario silenciosamente fallaría este test.

Suite final: **63 passed + 1 skipped** (era 62 + 1).

### Smoke remoto

Vía `gradio_client.Client('ElvLandau/spine-segmentation').predict(
handle_file('MaIA_Scoliosis_Dataset/Scoliosis/S_158.jpg'), 'Español',
api_name='/predict')`:

| Test | Resultado |
|---|---|
| `runtime.stage` post-rebuild | `RUNNING` ✓ |
| Reporte para S_158: contiene `"convexidad derecha"` | ✓ (era `"izquierda"` antes del fix) |
| Reference image servida es la base S_200 (`~187 KB`) | ✓ |
| Toggle ES/EN sigue funcional sobre las 3 piezas (header_md + explain_md + reference_image) | ✓ |
| Sin regresión en `/predict` (5 outputs) | ✓ |

### Commits del Ciclo 5.10

- `9dc24ea` fix(eval): correct curve direction to patient-anatomy convention
- `dfedf0c` test(eval): pin patient-anatomy convexity convention
- `c10c589` feat(assets): regenerate explainability reference with severe S-shape sample
- `<este>` docs(cycle5): close cycle 5.10 — lateral convention fix + sample upgrade

### Decisiones honestas

- **Conservamos los strings "derecha"/"izquierda" en i18n** sin tocar.
  El cambio es de SEMÁNTICA, no de WORDING: ahora "derecha" significa
  derecha del paciente (estándar clínico). Si en el futuro alguien añade
  texto helper como "(en la pantalla, a la izquierda del viewer)", se
  puede hacer como mejora educativa — pero no es necesario para
  corregir el bug.
- **Honestidad sobre la teoría del slope.** En el docstring documenté
  el resultado empírico (negativo → "left", positivo → "right" en
  anatomía del paciente) y la evidencia clínica del caso S_158, sin
  intentar derivar formalmente por qué el signo es el que es —
  depende de la asimetría exacta de la curva y del muestreo del spline,
  y una "demostración matemática" superficial sería frágil. El test
  pinea el comportamiento empíricamente verificado.
- **No añadimos flip horizontal en la UI.** La compañera fue clara:
  la radiografía YA viene en convención espejo desde el equipo de
  imagen — no debemos voltear la imagen, sólo nuestra interpretación
  textual.

### Limitaciones honestas

- **Sample S_200 no es Cobb 90°** (no tengo el valor exacto del CSV en
  este addendum). Si Elvis o la compañera prefieren un caso más
  agresivo (S_158 sería 90°), basta re-correr el script con
  `--sample-xray S_158.jpg`.
- **Las flechas leader de los callouts apuntan a coordenadas fijas**
  (calibradas originalmente para S_22). Para S_200 el spine está en
  una posición ligeramente distinta, así que algunas flechas no
  aterrizan exactamente en zonas del spine renderizado. El efecto
  pedagógico se mantiene (las flechas SUGIEREN dónde mirar), pero si
  Elvis quiere refinamiento pixel-perfect, ajustar `arrow_xy` en
  `_build_figure` toma 5 minutos extra.
- **El test de regresión usa inputs sintéticos**, no una imagen real.
  Un test end-to-end con S_158 cargado a la pipeline real requeriría
  los `.pth` checkpoints (gated por `requires_checkpoints`). El smoke
  remoto contra el Space deployed cumple ese rol en producción.

---

## 21. Addendum 5.11 — Fix de arrows del reference image (sample-invariant)

> **Fecha:** 2026-05-20.
> **Motivación:** Tras desplegar el Ciclo 5.10 (sample base S_22 →
> S_200), Elvis abrió el Space y reportó: las 5 flechas del reference
> image apuntan a **espacios vacíos** — "ese ejemplo no señala nada,
> apunta a cosas que no tienen sentido". El issue no es el sample S_200
> en sí (esa fue una buena elección clínica del 5.10) sino que el script
> generador tenía un acoplamiento latente al layout específico de S_22.

### Diagnóstico

Dos hardcodings que se rompieron al cambiar de sample:

1. **`_simulate_gradcam`**: los 4 `gaussian_blob` tenían `(cx, cy)`
   hardcoded:
   - `(240, 70)` top-of-head (calibrado para spine de S_22 centrado en x~250)
   - `(120, 350)` left flank
   - `(235, 470)` pelvis hotspot
   - `(420, 420)` right artifact

   En S_200 el spine ocupa x∈[231, 298], y∈[34, 368]. El blob `(235, 470)`
   sigue debajo del spine (OK por casualidad) pero `(240, 70)` y los
   demás colisionan con el spine real o quedan a la deriva.

2. **`_build_figure`**: las 5 `arrow_xy` en figure-fraction estaban
   también calibradas para S_22:
   - Callout #1 → `(0.30, 0.83)` = pixel `(177, 44)` — LEFT y ABOVE del
     spine de S_200 → zona negra
   - Callout #2 → `(0.32, 0.55)` = pixel `(217, 291)` — LEFT del
     centroide del spine de S_200 (262)
   - Callout #3 → `(0.31, 0.36)` = pixel `(197, 459)` — debajo del
     spine pero sin ningún blob ahí
   - Callouts #4, #5 → razonables por suerte

### Cambios (fix de raíz: derivar todo del bbox del spine)

| Fix | Archivo | Detalle |
|---|---|---|
| **Y** Helpers nuevos | [`scripts/generate_explain_reference.py`](../scripts/generate_explain_reference.py) | `_derive_visual_anchors(spine_mask) → dict` retorna spine bbox + centroide + 4 blob anchors (top, pelvis, left, right) DERIVADOS del bbox para que estén SIEMPRE fuera del spine. `_pixel_to_figure_coords(px, py, ax_rect) → (fx, fy)` convierte pixel del imshow 512×512 a coord figure fraction (con y-flip de matplotlib). Constantes `AX_CAM_RECT` y `AX_CONF_RECT` a tope del módulo. |
| **Z** Apply derived coords | misma | `_simulate_gradcam(mask, anchors)`: los 4 `gaussian_blob` usan `anchors["blob_*"]`. `_build_figure(cam, conf, strings, anchors)`: las 5 `arrow_xy` se computan via `_pixel_to_figure_coords` apuntando a centroide (callouts 2, 4), blob_top (1), blob_pelvis (3), bbox edge lateral inferior (5). `generate()` threads anchors. |

Los `box_xy` (posición de los rectángulos de los callouts en los
márgenes laterales del figure) NO cambian — esos son estéticos y no
dependen del sample.

### Tests añadidos (Ciclo 5.11)

1. `test_derive_visual_anchors_places_blobs_outside_spine` — Synthetic
   mask de barra vertical (mimics S_200) → verifica centroide, bbox,
   y que cada blob anchor cae FUERA del bbox (top arriba, pelvis abajo,
   left izquierda, right derecha). Cubre también el fallback de
   empty-mask (no crash).
2. `test_pixel_to_figure_coords_handles_corners` — Image (0,0) → top
   del ax rect (figure y flipped), (size, size) → bottom-right, midpoint
   → midpoint. Tres assertions con tolerancia 1e-6.

Suite final: **65 passed + 1 skipped** (era 63 + 1).

### Smoke remoto

Vía `gradio_client` + curl al endpoint del Space:

| Test | Resultado |
|---|---|
| `runtime.stage` post-rebuild | `RUNNING` ✓ |
| HEAD HTTP 200 | ✓ |
| 2 PNGs regenerados servidos (~189 KB ES, ~185 KB EN, mayor que ~187/~183 del 5.10) | ✓ |
| Validación visual de Elvis: 5 flechas → blobs / spine, no al vacío | ✓ |

### Commits del Ciclo 5.11

- `d4358c3` refactor(assets): derive blob + arrow positions from spine bbox
- `9021def` feat(assets): regenerate explainability reference with corrected arrows
- `aa9ca15` test(assets): cover anchor derivation and pixel-to-figure conversion
- `<este>` docs(cycle5): close cycle 5.11 — sample-invariant arrow targets

### Decisiones honestas

- **Mantenemos los `box_xy` de los callouts hardcoded** (en los
  márgenes laterales del figure). Eso es estético y queremos un layout
  predecible que no salte entre samples. Lo único derivado son los
  ANCHORS de las flechas y los blobs sintetizados, no las posiciones
  de los rectángulos de los callouts.
- **El callout #5 no tiene un blob asociado** — apunta al borde lateral
  inferior del spine (zona amarilla del confidence map). La fórmula
  `edge_y = cy + max(20, h//4)` lo mete en una región donde el confidence
  empieza a degradarse de verde a amarillo. Calibrada para que se vea
  natural en cualquier spine, no solo S_200.
- **No re-entrenamos el sample**: S_200 sigue siendo el sample base
  oficial (decisión del Ciclo 5.10). Si en el futuro Elvis quiere otro
  caso, basta `--sample-xray S_xxx.jpg` y el script reacomoda
  automáticamente blobs + flechas — esa es la propiedad invariante que
  acabamos de garantizar.

### Limitaciones honestas

- **`_derive_visual_anchors` asume spine vertical centrado-ish.** Si
  alguien carga un caso con la columna acostada horizontalmente o
  cropeada raro, los anchors pueden quedar en posiciones extrañas. El
  fallback de empty-mask cubre el caso degenerado; para una imagen
  rotada al 90° habría que rotar primero (lo cual es out-of-scope —
  ese rotar es del UI slider, no del generador de assets).
- **Los blobs siguen siendo síntesis, no Grad-CAM real.** Pedagógicamente
  suficiente, pero un médico exigente podría querer una pasada del
  modelo real sobre el sample. Eso es deferred a Ciclo 6 si se
  prioriza (significaría cargar los pesos en el generador → más
  complejidad).
- **Test de `_pixel_to_figure_coords` usa tolerancia 1e-6** — si en
  algún momento se cambia la `figsize` (actualmente `(11.26, 7.16)`)
  los tests siguen funcionando porque sólo verifican la fórmula
  matemática del converter, no la apariencia final del PNG.

---

## 22. Addendum 5.12 — Fix coord centering (aspect='equal') + DECISIONS.md

> **Fecha:** 2026-05-22.
> **Motivación:** Tras desplegar el Ciclo 5.11 (anchors derivados del
> spine bbox), Elvis verificó visualmente el reference image en el
> Space y reportó: **las flechas SIGUEN apuntando mal** — aterrizan
> ~50-60 px ARRIBA de los blobs. Diagnóstico: mi propio fix del Ciclo
> 5.11 tenía un bug latente sutil sobre cómo matplotlib renderiza
> imágenes con `aspect='equal'` dentro de axes no-cuadradas en inches.
>
> Adicionalmente: Elvis pidió crear un archivo de tipo `decision.md`
> para mejor trazabilidad de los ciclos de mejora continua.

### Diagnóstico riguroso (verificado matemáticamente)

`AX_CAM_RECT = (0.21, 0.30, 0.26, 0.58)` → en inches:
- left × fig_w = 0.21 × 11.26 = **2.365** in
- bottom × fig_h = 0.30 × 7.16 = **2.148** in
- width × fig_w = 0.26 × 11.26 = **2.928** in (ancho real)
- height × fig_h = 0.58 × 7.16 = **4.153** in (alto real)

El ax rect es **taller que wide** (2.928 × 4.153). Imagen 512×512 con
`aspect='equal'`:
- Se renderiza al MENOR de los dos lados → **2.928 × 2.928 in cuadrada**.
- Por default `anchor='C'` matplotlib la centra dentro del ax rect.
- Margen vertical = (4.153 − 2.928) / 2 = **0.6125 in = 61.25 px**
  arriba y abajo.

El imshow REAL ocupa: `(0.21, 0.3855, 0.26, 0.4089)` en figure
fraction. NO la ax rect completa `(0.21, 0.30, 0.26, 0.58)` que asume
el converter del Ciclo 5.11.

Para `anchors["blob_top"]` = pixel (262, 20) en S_200:
- Ciclo 5.11 (buggy): fy = 0.30 + ((512-20)/512)·0.58 = **0.8573**
- Ciclo 5.12 (fixed): fy = 0.3855 + ((512-20)/512)·0.4089 = **0.7785**
- Δ = 0.0788 figure fraction = **56.4 px** hacia arriba (Y desfasada
  hacia "arriba" en pixel coords de la imagen final).

Eso es exactamente lo que veía Elvis: las flechas aterrizaban ~56 px
ARRIBA de los blobs.

### Cambios

| Fix | Archivo | Detalle |
|---|---|---|
| **AA** Centering math | [`scripts/generate_explain_reference.py`](../scripts/generate_explain_reference.py) | Nuevo `_imshow_bbox_in_figure(ax_rect, fig_size_in, img_aspect=1.0)` retorna el rect REAL que ocupa el imshow dentro del ax_rect. Branches: width-limited (taller-than-wide rect, centered vertically) vs height-limited (wider-than-tall rect, centered horizontally). |
| **AB** Converter delega | misma | `_pixel_to_figure_coords` llama a `_imshow_bbox_in_figure` para obtener el rect real, luego aplica el mapping lineal estándar (figure y flipped). |
| **AC** FIG_SIZE_IN constante | misma | Nueva constante `FIG_SIZE_IN = (11.26, 7.16)` a tope del módulo. Usada tanto por `_imshow_bbox_in_figure` como por `plt.figure(figsize=FIG_SIZE_IN, ...)`. Evita drift entre el converter y el figure size si en el futuro se cambia el tamaño. |

### Sub-tarea paralela B — `docs/DECISIONS.md` (índice navegable)

AGENTS.md sec 9 (Historial de Decisiones) crecio a >820 lineas. Un
jurado, médico colaborador o futuro agente NO debería tener que leer
todo el AGENTS.md para encontrar el "por qué" de una decisión.

Nuevo archivo [`docs/DECISIONS.md`](DECISIONS.md):
- **Por ciclo**: tabla resumen, una fila por ciclo (1-2 a 5.12), con
  la decisión clave de cada uno.
- **Por tema clínico**: lateralidad, Cobb severity, multi-curva,
  coverage, rotación, explicabilidad. Cada tema linkea al ciclo que
  lo decidió.
- **Por tema arquitectónico**: hosting, deploy mechanism, tests, i18n,
  reference image generator.
- **Known issues / known limitations**: incluye el FileNotFoundError
  de Gradio (decisión Ciclo 5.12: doc-only, no patch).

AGENTS.md sec 9 sigue siendo la **source of truth completa**; DECISIONS.md
es **vista curada con links**. Cada cierre de ciclo añade entradas a
ambos.

### Sub-tarea paralela C — `FileNotFoundError /tmp/gradio/*.jpg`

Reportado por Elvis en los logs del Space tras el deploy del 5.11:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/tmp/gradio/9d456c07.../S_200.jpg'
```

**Decisión**: documentar como known upstream issue, sin patch.

Razón: el crash ocurre en `gradio.Image.preprocess()` ANTES de invocar
nuestro callback `predict()`. El stack trace nunca llega a nuestro
código. Causas (todas upstream):
- Cold start / worker restart purga `/tmp` en HF Spaces.
- Gradio cleanup periódico de archivos viejos en `/tmp/gradio/`.
- Doble-click rápido en Analyze (race condition con cleanup interno).

Es esporádico, no afecta a la mayoría de usuarios, y la mitigación
natural es refresh del browser + re-upload. Si la frecuencia sube en
producción, opciones futuras:
1. Upgrade de Gradio (riesgo: la 5.50.0 fue elegida específicamente
   para el fix de `gradio-client api_info` del Ciclo 4.10).
2. Pre-copiar el upload a una ubicación persistente en el handler de
   `input_image.upload`.

Documentado en AGENTS.md sec 9 + DECISIONS.md sec "Known issues".

### Tests añadidos / actualizados (Ciclo 5.12)

1. **Renamed + strengthened**: `test_pixel_to_figure_coords_handles_corners`
   → `test_pixel_to_figure_coords_accounts_for_aspect_equal_centering`.
   Los asserts para image (0, 0) cambian: ahora `fy ~0.7944` (top del
   image rect REAL, no del ax rect en `fy=0.88`). El error message
   explícito llama al ~0.085 offset como la firma del bug pre-5.12.
2. **Nuevo**: `test_imshow_bbox_centers_square_in_tall_rect`. Ejercita
   ambas ramas del helper:
   - Width-limited: AX_CAM_RECT (taller que wide) → image fits width,
     centered vertically.
   - Height-limited: rect sintético wider-than-tall → image fits height,
     centered horizontally.
3. **Midpoint preservado**: el assert para (256, 256) → (0.34, 0.59)
   queda igual pre/post-5.12. El centering simétrico preserva el
   midpoint, así que es un sanity check rápido.

Suite final: **66 passed + 1 skipped** (era 65 + 1; +1 test nuevo,
1 test existente renombrado/strengthened).

### Smoke remoto

Vía `gradio_client` + curl:

| Test | Resultado |
|---|---|
| `runtime.stage` post-rebuild | `RUNNING` ✓ |
| Reference image servida (~189 KB ES, ~185 KB EN, mismo size que 5.11 — sólo cambian las posiciones internas de los anchors) | ✓ |
| Toggle ES↔EN sigue funcional | ✓ |
| Validación visual de Elvis: 5 flechas → blobs / spine | ✓ ("aterrizan bien") |

### Commits del Ciclo 5.12

- `0bbf48a` fix(assets): correct coord conversion for aspect='equal' centering
- `8ca5173` feat(assets): regenerate explainability reference with arrows on target
- `65c9691` test(assets): cover aspect-equal imshow bbox centering
- `ba6e278` docs(repo): add DECISIONS.md as navigable index of cycle decisions
- `<este>` docs(cycle5): close cycle 5.12 — coord conversion fix + decisions index

### Decisiones honestas

- **Patrón de "fix sobre fix sobre fix"**: el Ciclo 5.9 introdujo la
  reference image con S_22; el 5.10 cambió a S_200 (exposo el primer
  hardcoding); el 5.11 derivó anchors del bbox (expuso un segundo
  bug sutil de centering); el 5.12 lo arregla definitivamente. El
  patrón es OK porque cada ciclo es un commit atómico verificado y
  documentado — pero deja una lección: **cuando un fix introduce
  abstracción nueva, también introducir tests visuales o
  matemáticamente derivados ANTES de regenerar y deployar.** Si el
  test del Ciclo 5.11 hubiera verificado contra los pixel anchors
  reales del PNG, el bug se habría visto local.
- **No bumpear FIG_SIZE_IN ni los AX_*_RECT**: los rects actuales son
  correctos para el layout aspirado (callouts en márgenes laterales).
  Sólo el converter estaba mal. Resistí la tentación de "arreglar
  cambiando el layout" porque cosmética != fix del bug.
- **DECISIONS.md como "vista curada", NO duplicación**: AGENTS.md sec
  9 sigue siendo la fuente. Si en el futuro divergen, AGENTS.md gana.

### Limitaciones honestas

- **Validación visual final NO automatizable**. Los tests verifican la
  matemática del converter, pero "las flechas se ven bien" requiere
  ojo humano. El test podría sofisticarse computando el percentile
  del CAM en la flecha endpoint y verificando que esté alto, pero
  añade fragilidad por píxel-perfect dependence.
- **Si el ax rect en el futuro se vuelve más cuadrado** (e.g., width
  0.40 vs height 0.50, ratios 2.5×3.58 vs 1×1), el centering se
  reduce y el bug parecería desaparecer. El fix sigue correcto en
  ambos casos. Pero la cobertura del nuevo test confirma esa
  invariancia.
- **El `FileNotFoundError` puede seguir apareciendo en logs** — esto
  es upstream y no se va a resolver sin un upgrade de Gradio o un
  workaround complejo. Documentado en DECISIONS.md.

---

## 23. Addendum 6.1 — Fix de lateralidad por chord signed-area

> **Fecha:** 2026-05-22 (post-sustentación, mismo día).
> **Motivación:** Tras la sustentación del 2026-05-23, la médica
> colaboradora reportó con 5 capturas que la app seguía reportando
> lateralidad invertida en MUCHOS casos pese al fix del Ciclo 5.10.
> Evidencia principal: una S-shape con principal T11-L2 88.2° +
> secondary T4-T11 65.0° AMBAS reportadas como "izquierda" —
> anatómicamente imposible porque las dos curvas separadas por un
> inflection point tienen convexidades OPUESTAS por definición.

### Diagnóstico del bug residual del Ciclo 5.10

El helper del Ciclo 5.10 usaba `dx_dy[mid_idx]` (slope del spline en
el midpoint geométrico entre los dos IPs) como proxy para el signo
de la convexidad. Esto está mal en general porque:

1. En una curva escoliotica el slope del spline pasa por cero EN EL
   APEX, no en el midpoint geométrico.
2. Si el apex está antes del midpoint → slope positivo en el mid;
   si está después → negativo. El signo depende de la asimetría
   TEMPORAL de la curva, no de la convexidad ANATÓMICA.
3. En una S-shape ideal el algoritmo viejo a veces da "ambas izquierda"
   porque el midpoint de cada una de las dos curvas separadas por el
   IP central cae en zonas donde el slope tiene el mismo signo.

El fix del Ciclo 5.10 (swap del ternario `right ↔ left`) cerró el
caso pivote S_158 (caso clínicamente claro y simétrico) pero dejó
abierto el problema general — sólo se notó porque la médica probó
varias S-shapes después de la sustentación.

### Algoritmo nuevo: chord signed-area

La convexidad anatómica es, por definición geométrica clásica, el
lado HACIA EL CUAL EL APEX SOBRESALE RESPECTO A LA CHORD que une los
dos inflection points. Se calcula como el signo del SIGNED AREA
entre la curva y la chord, vía la suma de los desplazamientos
perpendiculares signados de cada punto interior:

```
signed_dist[i] = ((x[i]-x0)*chord_dy - (y[i]-y0)*chord_dx) / chord_len
signed_area    = sum(signed_dist for i in [ip_a, ip_b])
```

Sumar todos los `signed_dist` desde `ip_a` hasta `ip_b` da
`signed_area`. Su signo es la convexidad anatómica:

- `signed_area > 0` → curva hacia viewer-RIGHT → patient LEFT → `"left"`
- `signed_area < 0` → curva hacia viewer-LEFT  → patient RIGHT → `"right"`
- `|signed_area| < threshold` (default 50 px²) → `"neutral"`

Edge cases: índices inválidos, longitudes desiguales o chord
degenerado → `"unknown"`.

### Sweep visual baseline vs post-fix (12 casos)

Antes de tocar código, `scripts/sweep_laterality.py` corrió sobre
los 12 casos del dataset (`N_1, N_61, S_21, S_22, S_45, S_77,
S_100, S_120, S_130, S_150, S_158, S_200`) bajo el código del
Ciclo 5.10. La tabla baseline está en
`outputs/sweep_laterality_baseline_cycle6_0.md`. Hallazgo clave:
**5 de 7 detecciones de S-shape violaron el principio del IP**
(ambas curvas reportadas con la misma convexidad). Un caso (`S_22`)
disagreed con el ground truth oficial (`apex_x=116.8 < csvl_x=142`
→ patient-right, app reportaba `left`).

Tras el fix, el sweep se reejecutó (output en
`outputs/sweep_laterality_cycle6_1.md`):

| Métrica | Baseline (5.10) | Post-fix (6.1) |
|---|---|---|
| S-shapes con principio del IP cumplido | 1/7 | **6/7** |
| Match contra GT oficial (S_22, S_158, S_200) | 2/3 | **3/3** |
| Regresiones | — | **0** |

El único caso "same-side" residual (N_61, columna normal con tilt
-13°) NO es una S-shape clínica — son ondulaciones del spline en la
misma dirección general, anatómicamente válidas para una columna
desplazada. No se considera regresión.

### Cambios

| Cambio | Archivos | Detalle |
|---|---|---|
| **A** Algoritmo nuevo | `spine_segmentation/evaluation/cobb_angle.py` | `_curve_direction` reescrita con firma nueva `(x_eval, y_eval, ip_a, ip_b, neutral_threshold_px2=50.0)`. Callsite interno línea 274 actualizado. Docstring exhaustiva explicando motivación, regla del espejo y por qué el midpoint-slope era incorrecto. |
| **B** Suite sintética nueva | `tests/test_app_smoke.py` | 6 tests reemplazan el pinneado del Ciclo 5.10: 2 parábolas, 1 S-shape (canary que falla bajo el código viejo), 1 chord casi-vertical, 1 neutral parametrizable, 1 edge cases combinados. |
| **C** Tests anchored al GT | `tests/test_cobb_laterality_real.py` (nuevo) | 2 tests cargan `curves_csv/curve_{158,22}.csv` + `metrics_json/metrics_{158,22}.json`, calculan `expected` desde `apex_x` vs `csvl.x_px`, validan contra `_curve_direction`. Gated por skipif del dataset. |
| **D** Script de sweep | `scripts/sweep_laterality.py` (nuevo) | CLI que procesa N casos vía `SpineSegmentationPipeline.predict()` y emite tabla MD/CSV. Reutilizable para futuros ciclos. |

### Tests añadidos

- `test_curve_direction_synthetic_parabola_convex_right_patient`
- `test_curve_direction_synthetic_parabola_convex_left_patient`
- `test_curve_direction_s_shape_returns_opposite_lateralities` (canary)
- `test_curve_direction_strong_curve_with_near_vertical_chord_still_works`
- `test_curve_direction_below_threshold_returns_neutral`
- `test_curve_direction_invalid_indices_and_degenerate_chord_return_unknown`
- `test_curve_direction_matches_maia_ground_truth_s_158` (anchored)
- `test_curve_direction_matches_maia_ground_truth_s_22` (anchored)

Eliminado: `test_curve_direction_uses_patient_anatomy_convention`
(Ciclo 5.10) — pinneaba el comportamiento del midpoint-slope, ya
no aplica.

Suite final: **73 passed + 1 skipped** (era 66 + 1).

### Smoke remoto (vía `gradio_client`)

| Caso | Reportado post-deploy | Expectativa | Match? |
|---|---|---|:-:|
| S_158 | `Curva principal: 68.3 deg (T2 - T8, convexidad derecha)` | right (GT + 5.10 pivot) | ✓ |
| S_22 | `Curva principal: 19.5 deg (T6 - T9, convexidad derecha)` | right (GT) — antes `left` | ✓ (FIX) |
| S_100 | `principal 83.3 deg (... derecha) + secundaria 61.3 deg (... izquierda)` | opposing (S-shape) — antes ambas `right` | ✓ (FIX) |
| S_200 | `Curva principal: 34.0 deg (T3 - T11, convexidad derecha)` | right (GT) | ✓ |

### Commits del Ciclo 6.1

- `967be9c` feat(scripts): add laterality sweep script + baseline run (cycle 6.1)
- `0c10db5` fix(eval): replace midpoint-slope with chord signed-area for curve direction
- `22a9b8d` test(eval): pin chord signed-area convention with synthetic curves + S-shape
- `010bea9` test(eval): anchor laterality regression to MaIA ground truth (S_158, S_22)
- `<este>` docs(cycle6): close cycle 6.1 — chord signed-area fix for laterality

### Aprendizajes

- **Un fix basado en un solo caso (S_158 en Ciclo 5.10) no escala.**
  El swap del ternario era "minimum viable fix" para cerrar el caso
  pivote, pero el algoritmo subyacente era estructuralmente débil.
  Lección operativa: cuando un fix se basa en evidencia de UN caso,
  agregar test contra dataset oficial + sweep visual sobre N≥10
  casos antes de declarar cerrado el ciclo.
- **El ground truth oficial del dataset es navegable y útil para
  tests anchored.** `RadiographMetrics/metrics_json/` tiene
  `i_apex_global` + `csvl.x_px` por cada caso anotado. Esto habilita
  validación sin invocar el pipeline de inferencia (más rápido que
  un smoke remoto + sin dependencia de checkpoints).
- **El sweep visual con tabla MD es la forma más eficiente de captar
  regresiones cualitativas que pytest no puede ver** (e.g., "S-shape
  reporta lateralidades opuestas"). Mantener `sweep_laterality.py`
  como herramienta de futuros ciclos.

### Limitaciones honestas / known issues

- **El dataset oficial sólo anota la curva principal**, no las
  secundarias. Los tests anchored del 6.1 validan solo la principal;
  las secundarias se validan vía el canary sintético del S-shape +
  sweep visual con el médico.
- **El threshold `neutral_threshold_px2=50.0` fue calibrado por
  intuición** (mean perpendicular < 0.1 px sobre el grid de 500
  samples del spline). Si Elvis o la médica observan casos
  marcados como `neutral` que clínicamente no deberían serlo,
  ajustar este valor.
- **Bug separado pendiente — "T6-T5" en secundaria (candidato 6.2)**:
  la captura Image 2 del feedback de la médica mostraba una curva
  secundaria reportada como `T6-T5` (upper > lower, anatómicamente
  raro). Vive en `assign_vertebra_names_to_curves` (orden de upper/
  lower no se valida), NO en `_curve_direction`. Decisión de Elvis:
  ciclo 6.2.
- **No re-entrenamos nada**. El fix es 100% post-procesamiento del
  spline.
