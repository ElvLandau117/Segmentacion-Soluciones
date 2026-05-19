# Prompt para el próximo chat — Ciclo 5 + 5.1 + 5.2 + 5.3 cerrados, pendiente brief de Ciclo 6

> **Estado al 2026-05-19:** Ciclo 5 + 5.1 + 5.2 + 5.3 ✅ COMPLETOS.
> - Assessment via binary (más robusto).
> - Visualización Cobb tipo Fig 1 de Shi et al. 2025.
> - Detección multi-curva (5.2): TODAS las curvas (S-shape, triple-curve)
>   con dirección + nombres Tn/Lm.
> - **Cobertura del binary (5.3):** S_22 (pivote) pasaba de falso-negativo
>   a verdadero-positivo. Fixes A+B+C+D + multi-pass adaptativo de smoothing
>   en `cobb_from_binary` + UI con bloque `=== COVERAGE ===` y Assessment
>   "Inconclusive — insufficient coverage" cuando partial + 0°.
> - Smoke remoto verde sobre 9 casos: N_1, S_22, S_21, S_100, S_45, S_77,
>   S_120, S_130, S_150. S_100 preserva 2 curvas (no regresión vs 5.2).
> - Suite: 32 passed + 1 skipped.
>
> **No hay handoff técnico pendiente.** El próximo chat debe definir el scope
> del Ciclo 6 (refinamiento de modelo y/o entrega académica final) y escribir
> el brief en `docs/CICLO_6_BRIEF.md`.

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (sección 5 = estado de los ciclos).
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclo 5 + 5.1 + 5.2 + 5.3.
   Lee especialmente la sección 13 (cobertura del binary, smoke S_22).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (2026-05-19)

**Ciclos 1-5.3 ✅ COMPLETOS:**

- App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer.
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- **32 tests** pasando (1 gated por checkpoints locales).
- Mecanismo reproducible de updates: `scripts/upload_to_space.py`.
- **UX clínica Cobb (Ciclo 5 + 5.1 + 5.2):** Assessment basado en binary
  (más robusto), visualización tipo Fig 1 de Shi et al. 2025, multi-curve
  detection (S-shape, triple), label-transfer del multiclass para nombrar
  vértebras.
- **Cobertura del binary (Ciclo 5.3):** S_22 (caso pivote del Ciclo 5.3)
  pasa de "0° Normal" falso-negativo a "2 curvas Mild + WARNING de coverage
  parcial". Fix combinado: umbral binario 0.3 + cierre morfológico vertical
  + multi-pass adaptativo de smoothing + bloque `=== COVERAGE ===` en la
  UI + Assessment "Inconclusive" cuando partial + 0°. S_100 preserva sus 2
  curvas (no regresión vs Ciclo 5.2).

### Lo que sigue (Ciclo 6)

El Ciclo 5 + sus 3 addenda están cerrados. El Ciclo 6 NO tiene brief
todavía. Candidatos identificados:

1. **Fallback multiclass cuando binary cubre poco** (fix E del plan original
   del 5.3, que se decidió no hacer si los quick wins bastaban). Útil si
   en producción aparecen casos donde la probabilidad binaria del modelo
   está por debajo de 0.3 en la zona lumbar — ahí ningún post-proc ayuda.
2. **Reentrenamiento con augmentation lumbar agresivo** — CLAHE, crop
   variable, gamma, contrast jitter centrados en el tercio inferior.
   Considerar pre-training RadImageNet.
3. **Mejorar explicabilidad** — enmascarar el confidence map por la máscara
   predicha (mostrar solo dentro de la columna) + probar Seg-Grad-CAM en vez
   de Grad-CAM vanilla.
4. **Recalibrar el umbral de "is_partial"**. El default actual (`ratio <
   0.7` OR `n_vert < 15`) marca S_21 como parcial aunque cubre la columna
   completa (es padding lo que reduce el ratio). Posible: usar bounding box
   real de la columna como denominador, no `image_height`.
5. **Quantización INT8 para tablet** (modelo ligero para edge).
6. **CI con GitHub Actions** — automatizar `pytest` + lint en cada PR.
7. **Sustentación oral + slides + demo en vivo + smoke test cross-device.**
8. **Artículo IEEE/ACM** — si los resultados del refinamiento lo soportan.

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
- Tests locales: `pytest tests/ -v` debe seguir verde (32/33).
- Si surgen ideas tomadas de papers: revisar críticamente qué es aplicable a
  nuestro dataset (174 imgs, single-rater, sin landmarks de endplate) vs qué
  requiere recursos que no tenemos. Honestidad ante todo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktree del Ciclo 5.3: cerrado y merged. Próximo ciclo creará su propio
  worktree desde main.

## ↑ HASTA AQUÍ
