# Prompt para el próximo chat — Ciclos 5 + 5.1 + 5.2 + 5.3 + 5.4 + 5.5 + 5.6 cerrados, pendiente brief de Ciclo 6

> **Estado al 2026-05-19:** Ciclos 5 + 5.1 + 5.2 + 5.3 + 5.4 + 5.5 + 5.6 ✅ COMPLETOS.
> - Assessment via binary (más robusto).
> - Visualización Cobb tipo Fig 1 de Shi et al. 2025.
> - Detección multi-curva (5.2).
> - Cobertura del binary + UI informativa (5.3).
> - Robustez ante rotación + UX viz Cobb (5.4): bloque ROTATION WARNING +
>   filtros para curvas degeneradas + dedup de rótulos.
> - Control manual de rotación (5.5): slider -180..180 + 5 botones
>   rápidos. N_61 con +13° → 0° Normal correctamente.
> - **Live preview de la rotación (5.6):** mover el slider rota la
>   imagen en vivo (~50-150ms) gracias a `gr.State` para la original +
>   handler en `slider.change`. El médico ve cómo queda ANTES de gastar
>   los 10s de Analyze.
> - Smoke remoto verde. Suite: **47 passed + 1 skipped**.
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
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclos 5 + 5.1 + 5.2
   + 5.3 + 5.4 + 5.5 + 5.6. Lee especialmente las secciones 15 (manual
   rotation) y 16 (live preview).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (2026-05-19)

**Ciclos 1-5.6 ✅ COMPLETOS:**

- App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer.
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- **47 tests** pasando (1 gated por checkpoints locales).
- Mecanismo reproducible de updates: `scripts/upload_to_space.py`
  (acuérdate de pasar `--path-in-repo` explícito para evitar overwrite
  del shim raíz, lección del 5.5).
- UX clínica Cobb completa.
- **Rotación end-to-end (Ciclos 5.4 + 5.5 + 5.6):** detección
  automática del tilt (warning informativo) + control manual via
  slider y 5 botones rápidos + **live preview** que rota la imagen en
  tiempo real al mover el slider. El médico ve la imagen rotada y
  decide ANTES de presionar Analyze.

### Lo que sigue (Ciclo 6)

El Ciclo 5 + sus 6 addenda están cerrados. El Ciclo 6 NO tiene brief
todavía. Candidatos identificados:

1. **Recalibrar threshold de tilt** (12° → 14-15°). S_100/S_150
   (escoliosis severas con tilt ~12.6°) disparan el ROTATION WARNING
   aunque la curva es real. Confuso post-rotación manual.
2. **Fallback multiclass cuando binary cubre poco** (fix E del Ciclo 5.3
   plan original). Útil cuando el binary no detecta zona lumbar.
3. **Reentrenamiento con augmentation lumbar agresivo** — CLAHE, crop
   variable, gamma, contrast jitter centrados en el tercio inferior.
   Considerar pre-training RadImageNet.
4. **Mejorar explicabilidad** — enmascarar el confidence map por la
   máscara predicha + probar Seg-Grad-CAM en vez de Grad-CAM vanilla.
5. **Recalibrar el umbral `is_partial` del Coverage** (Ciclo 5.3). El
   default actual marca S_21 como parcial aunque cubre la columna
   completa (es padding lo que reduce el ratio).
6. **Flip horizontal/vertical en la UI** — para radiografías espejadas
   (caso muy raro pero documentado).
7. **Live preview de la SEGMENTACIÓN** (no solo de la rotación):
   mostrar overlay binario en tiempo real conforme el slider cambia,
   sin disparar Analyze. Caro en compute pero posible con un mini-pase
   downscaled. Nice-to-have.
8. **Quantización INT8 para tablet** (modelo ligero para edge).
9. **CI con GitHub Actions** — automatizar `pytest` + lint en cada PR.
10. **Sustentación oral + slides + demo en vivo + smoke test cross-device.**
11. **Artículo IEEE/ACM** — si los resultados del refinamiento lo soportan.

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
- Tests locales: `pytest tests/ -v` debe seguir verde (47/48).
- Si surgen ideas tomadas de papers: revisar críticamente qué es aplicable a
  nuestro dataset (174 imgs, single-rater, sin landmarks de endplate) vs qué
  requiere recursos que no tenemos. Honestidad ante todo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktree del Ciclo 5.6: cerrado y merged. Próximo ciclo creará su propio
  worktree desde main.

## ↑ HASTA AQUÍ
