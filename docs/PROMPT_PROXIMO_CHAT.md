# Prompt para el próximo chat — Ciclo 5 + 5.1 + 5.2 + 5.3 + 5.4 cerrados, pendiente brief de Ciclo 6

> **Estado al 2026-05-19:** Ciclo 5 + 5.1 + 5.2 + 5.3 + 5.4 ✅ COMPLETOS.
> - Assessment via binary (más robusto).
> - Visualización Cobb tipo Fig 1 de Shi et al. 2025.
> - Detección multi-curva (5.2).
> - Cobertura del binary + UI informativa (5.3): S_22 pasó de
>   falso-negativo a verdadero-positivo, multi-pass adaptativo de smoothing.
> - **Robustez ante rotación + UX viz Cobb (5.4):** SVD del skeleton para
>   detectar tilt → bloque `=== ROTATION WARNING ===`. Filtros para curvas
>   degeneradas (y-distance + upper==lower). Dedup + anti-overlap de
>   rótulos en la viz. N_61 (rotada) baja de 4 curvas fantasma a 1 con
>   warning. S_22 pierde la degenerada T9-T9 y queda con 1 sola curva
>   limpia.
> - Smoke remoto verde sobre 10 casos del dataset (N_1, N_61, S_21, S_22,
>   S_45, S_77, S_100, S_120, S_130, S_150). Sin regresión.
> - Suite: 41 passed + 1 skipped.
>
> **No hay handoff técnico pendiente.** El próximo chat debe definir el
> scope del Ciclo 6 (refinamiento de modelo y/o entrega académica final)
> y escribir el brief en `docs/CICLO_6_BRIEF.md`.

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (sección 5 = estado de los ciclos).
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclo 5 + 5.1 + 5.2 +
   5.3 + 5.4. Lee especialmente la sección 14 (rotación, curvas
   degeneradas, dedup de rótulos).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (2026-05-19)

**Ciclos 1-5.4 ✅ COMPLETOS:**

- App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer.
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- **41 tests** pasando (1 gated por checkpoints locales).
- Mecanismo reproducible de updates: `scripts/upload_to_space.py`.
- UX clínica Cobb (Ciclo 5 + 5.1 + 5.2): Assessment binary, viz tipo
  Fig 1 Shi et al., multi-curve detection, label-transfer del multiclass.
- Cobertura del binary (Ciclo 5.3): bloque `=== COVERAGE ===`, Assessment
  "Inconclusive" cuando partial + 0°.
- **Robustez de rotación + UX viz (Ciclo 5.4):** bloque
  `=== ROTATION WARNING ===` cuando SVD del skeleton ve tilt > 12°.
  Filtros para curvas degeneradas (`MIN_IP_Y_DISTANCE_PX = 30` +
  drop si `upper == lower`). Dedup + anti-overlap de rótulos en la
  viz para que 2 curvas que comparten vértebra no produzcan texto
  ilegible. **N_61** (Normal rotada) baja de 4 curvas fantasma a 1.
  **S_22** ya no muestra la degenerada T9-T9.

### Lo que sigue (Ciclo 6)

El Ciclo 5 + sus 4 addenda están cerrados. El Ciclo 6 NO tiene brief
todavía. Candidatos identificados:

1. **Auto de-rotación previo al pipeline**. Cuando el SVD detecta tilt,
   en lugar de solo advertir, rotar la imagen ANTES de correr binary +
   multiclass. Reduce la curva fantasma de N_61 a 0 y elimina el warning
   borderline en escoliosis severas (S_100, S_150).
2. **Recalibración del threshold de tilt** (12° → 14-15°) tras observar
   que casos severos como S_100 disparan el warning marginalmente.
3. **Fallback multiclass cuando binary cubre poco** (fix E del plan
   original del 5.3). Útil si en producción aparecen casos donde la
   probabilidad binaria del modelo está por debajo de 0.3 en la zona
   lumbar — ahí ningún post-proc ayuda.
4. **Reentrenamiento con augmentation lumbar agresivo** — CLAHE, crop
   variable, gamma, contrast jitter centrados en el tercio inferior.
   Considerar pre-training RadImageNet.
5. **Mejorar explicabilidad** — enmascarar el confidence map por la
   máscara predicha + probar Seg-Grad-CAM en vez de Grad-CAM vanilla.
6. **Recalibrar el umbral `is_partial` del Coverage** (Ciclo 5.3). El
   default actual marca S_21 como parcial aunque cubre la columna
   completa (es padding lo que reduce el ratio). Usar bounding box real
   de la columna como denominador.
7. **Quantización INT8 para tablet** (modelo ligero para edge).
8. **CI con GitHub Actions** — automatizar `pytest` + lint en cada PR.
9. **Sustentación oral + slides + demo en vivo + smoke test cross-device.**
10. **Artículo IEEE/ACM** — si los resultados del refinamiento lo soportan.

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
- Tests locales: `pytest tests/ -v` debe seguir verde (41/42).
- Si surgen ideas tomadas de papers: revisar críticamente qué es aplicable a
  nuestro dataset (174 imgs, single-rater, sin landmarks de endplate) vs qué
  requiere recursos que no tenemos. Honestidad ante todo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktree del Ciclo 5.4: cerrado y merged. Próximo ciclo creará su propio
  worktree desde main.

## ↑ HASTA AQUÍ
