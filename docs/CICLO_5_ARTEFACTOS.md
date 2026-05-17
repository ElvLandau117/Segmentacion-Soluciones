# Ciclo 5 — UX clinica del Cobb · Artefacto de Salida

> **Fecha de cierre:** 2026-05-17 noche
> **Estado:** ✅ COMPLETO
> **URL pública:** https://huggingface.co/spaces/ElvLandau/spine-segmentation
> **Próximo ciclo (tentativo):** Ciclo 6 — Refinamiento del modelo, entrega académica final, sustentación.

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
