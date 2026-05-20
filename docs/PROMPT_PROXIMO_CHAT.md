# Prompt para el próximo chat — Ciclos 5 + 5.1..5.8 cerrados, pendiente brief de Ciclo 6

> **Estado al 2026-05-20:** Ciclos 5 + 5.1 + 5.2 + 5.3 + 5.4 + 5.5 + 5.6 + 5.7 + 5.8 ✅ COMPLETOS.
> - Assessment via binary (más robusto que multiclass).
> - Visualización Cobb tipo Fig 1 de Shi et al. 2025.
> - Detección multi-curva (5.2).
> - Cobertura del binary + UI informativa (5.3).
> - Robustez ante rotación + UX viz Cobb (5.4).
> - Control manual de rotación (5.5).
> - Live preview de la rotación (5.6).
> - Limpieza multiclass del frontend + toggle ES/EN (5.7).
> - **Polish del tab Explainability (5.8):** Grad-CAM y Confidence Map
>   ahora se enmascaran por la `binary_mask` predicha (solo pintan
>   dentro de la columna detectada), el Grad-CAM se renormaliza al
>   percentile 95 para mejor contraste, el panel side-by-side gana
>   títulos in-image y mini-colorbars, y el Markdown explicatorio se
>   tradujo a ES/EN con sección "Cómo leerlo".
> - Smoke remoto verde. Suite: **60 passed + 1 skipped**.
>
> **No hay handoff técnico pendiente.** El próximo chat debe definir el
> scope del Ciclo 6 y escribir el brief en `docs/CICLO_6_BRIEF.md`.

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (sección 5 = estado de los ciclos).
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclos 5 + 5.1..5.8.
   Lee especialmente la sección 18 (explainability polish: masking +
   colorbars + bilingüe).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (2026-05-20)

**Ciclos 1-5.8 ✅ COMPLETOS:**

- App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer.
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- **60 tests** pasando (1 gated por checkpoints locales).
- Mecanismo reproducible de updates: `scripts/upload_to_space.py` (con
  `--path-in-repo` explícito siempre).
- UX clínica Cobb completa.
- Rotación end-to-end (5.4 + 5.5 + 5.6): detección + control manual + live preview.
- Multiclass cleanup + toggle ES/EN (5.7).
- **Explainability polish (5.8):** Grad-CAM y Confidence enmascarados
  por la columna detectada (cero pintura fuera del spine), percentile-clip
  p95 en cam para contraste, títulos + colorbars in-image, markdown
  bilingüe con sección "Cómo leerlo".

### Lo que sigue (Ciclo 6)

El Ciclo 5 + sus 8 addenda están cerrados. El Ciclo 6 NO tiene brief
todavía. Candidatos identificados:

1. **Recalibrar threshold de tilt** (12° → 14-15°). S_100/S_150
   (escoliosis severas con tilt ~12.6°) disparan el ROTATION WARNING
   aunque la curva es real. Confuso post-rotación manual.
2. **Fallback multiclass cuando binary cubre poco** (fix E del Ciclo 5.3
   plan original). Útil cuando el binary no detecta zona lumbar.
3. **Reentrenamiento con augmentation lumbar agresivo** — CLAHE, crop
   variable, gamma, contrast jitter centrados en el tercio inferior.
   Considerar pre-training RadImageNet.
4. **Seg-Grad-CAM auténtico** en vez de Grad-CAM vanilla con
   `SemanticSegmentationTarget`. El 5.8 mejoró el render pero el cam
   subyacente sigue siendo el mismo algoritmo. Cambiar requeriría
   añadir `compute_cam_per_layer` con upsampling.
5. **Recalibrar el umbral `is_partial` del Coverage** (Ciclo 5.3). El
   default actual marca S_21 como parcial aunque cubre la columna
   completa (es padding lo que reduce el ratio).
6. **Flip horizontal/vertical en la UI** — para radiografías espejadas.
7. **i18n Nivel C** — traducir también tabs, slider y botones rápidos.
8. **Live preview de la SEGMENTACIÓN** (no solo de la rotación) —
   mostrar overlay binario en tiempo real conforme el slider cambia.
9. **Colorbar con valores numéricos** en explainability (actualmente
   solo "Alta/Baja" — añadir 0.25, 0.5, 0.75 si cabe).
10. **Quantización INT8 para tablet** (modelo ligero para edge).
11. **CI con GitHub Actions** — automatizar `pytest` + lint en cada PR.
12. **Sustentación oral + slides + demo en vivo + smoke test cross-device.**
13. **Artículo IEEE/ACM** — si los resultados del refinamiento lo soportan.

**Acción recomendada para el próximo chat:**

1. Leer los 5 archivos del onboarding.
2. Conversar con Elvis para priorizar 1-3 items del Ciclo 6.
3. Discutir honestamente qué es viable vs aspiracional (no decir sí a todo).
4. Escribir `docs/CICLO_6_BRIEF.md`.
5. Empezar la primera unidad.

### Restricciones operativas (recordatorio)

- Metodología: **Spec-Driven Work + Work Orchestration** (Leonardo Gonzalez).
- Cada cambio = 1 commit con convención de `WORKFLOW.md` sección 4.
- Commits **sin** co-autoría de IA (sólo `Elvis Hernandez`).
- **NO** force push a `main` ni al Space.
- **NO** `git push hf` para parchar el Space — usar `python scripts/upload_to_space.py`.
- **Siempre pasar `--path-in-repo` explícito** para evitar sobrescribir
  el shim raíz cuando los archivos comparten basename.
- Tests locales: `pytest tests/ -v` debe seguir verde (60/61).
- Si surgen ideas tomadas de papers: revisar críticamente qué es aplicable a
  nuestro dataset (174 imgs, single-rater, sin landmarks de endplate) vs qué
  requiere recursos que no tenemos. Honestidad ante todo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktree del Ciclo 5.8: cerrado y merged. Próximo ciclo creará su propio
  worktree desde main.

## ↑ HASTA AQUÍ
