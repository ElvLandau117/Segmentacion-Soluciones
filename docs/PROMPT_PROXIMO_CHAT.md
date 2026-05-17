# Prompt para el próximo chat — Ciclo 5 + 5.1 cerrados, pendiente brief de Ciclo 6

> **Estado al 2026-05-17 noche:** Ciclo 5 + 5.1 ✅ COMPLETOS.
> - Assessment via binary (más robusto).
> - Visualización Cobb tipo Fig 1 de Shi et al. 2025: cajas verdes en end vertebrae
>   + perpendiculares rojas al endplate + arco del ángulo + speedometer para casos
>   leves + overlay del binary (spline + inflection points).
> - UI dual-Cobb con CONCORDANCIA.
> - Smoke remoto verde con colores correctos (rojo/verde/cyan/amarillo).
> - Caso S_21 (escoliosis leve) ahora se detecta como "Mild" correctamente.
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
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en el Ciclo 5 (Cobb UX).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (2026-05-17 noche)

**Ciclos 1-5 ✅ COMPLETOS:**

- App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer.
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- 21 tests pasando (1 gated por checkpoints locales).
- Mecanismo reproducible de updates: `scripts/upload_to_space.py`.
- **UX clínica Cobb (Ciclo 5 + 5.1):** Assessment basado en binary (más robusto),
  visualización tipo Fig 1 de Shi et al. 2025 con perpendiculares + arco +
  cajas + endplate markers + speedometer + binary overlay (spline + inflection
  points), UI con ambos métodos + indicador de concordancia, convención RGB
  consistente.

### Lo que sigue (Ciclo 6)

El Ciclo 5 está cerrado. El Ciclo 6 NO tiene brief todavía. Candidatos
identificados en el handoff del Ciclo 5:

1. **Mejorar Cobb multiclase** — SVD sobre centroides de vértebras detectadas
   + constraint biomecánico como post-proc + votación robusta para descartar
   outliers (vértebras fragmentadas).
2. **Mejorar explicabilidad** — enmascarar el confidence map por la máscara
   predicha (mostrar solo dentro de la columna) + probar Seg-Grad-CAM en vez
   de Grad-CAM vanilla.
3. **Refinamiento del modelo** — augmentation agresiva + balanceo de C3-C5
   + pre-training con RadImageNet + ensemble.
4. **Quantización INT8 para tablet** (modelo ligero para edge).
5. **CI con GitHub Actions** — automatizar `pytest` + lint en cada PR.
6. **Sustentación oral + slides + demo en vivo + smoke test cross-device.**
7. **Artículo IEEE/ACM** — si los resultados del refinamiento lo soportan.

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
- Tests locales: `pytest tests/ -v` debe seguir verde (17/18).
- Si surgen ideas tomadas de papers: revisar críticamente qué es aplicable a
  nuestro dataset (174 imgs, single-rater, sin landmarks de endplate) vs qué
  requiere recursos que no tenemos. Honestidad ante todo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktree del Ciclo 5: `.claude/worktrees/cycle5` (branch `cycle5-cobb-ui`).
  Mergear a main + borrar al cierre.

## ↑ HASTA AQUÍ
