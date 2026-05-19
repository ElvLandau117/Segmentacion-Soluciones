# Prompt para el próximo chat — Ciclo 5 + 5.1 + 5.2 + 5.3 + 5.4 + 5.5 cerrados, pendiente brief de Ciclo 6

> **Estado al 2026-05-19:** Ciclo 5 + 5.1 + 5.2 + 5.3 + 5.4 + 5.5 ✅ COMPLETOS.
> - Assessment via binary (más robusto).
> - Visualización Cobb tipo Fig 1 de Shi et al. 2025.
> - Detección multi-curva (5.2).
> - Cobertura del binary + UI informativa (5.3): S_22 pasó de
>   falso-negativo a verdadero-positivo, multi-pass adaptativo de smoothing.
> - Robustez ante rotación + UX viz Cobb (5.4): SVD del skeleton →
>   bloque `=== ROTATION WARNING ===`. Filtros para curvas degeneradas
>   (y-distance + upper==lower). Dedup + anti-overlap de rótulos en viz.
> - **Control manual de rotación en la UI (5.5):** Slider -180..180 + 5
>   botones rápidos (-90/-5/Reset/+5/+90) bajo el componente Image. El
>   médico decide si rotar; el pipeline analiza la imagen tal como
>   queda. **N_61 con rotación +13° produce 0° Normal correctamente**,
>   eliminando el falso positivo del 5.4. Auto-rotación descartada por
>   ambigüedad de signo (`-tilt_deg` habría EMPEORADO N_61).
> - Smoke remoto verde. Suite: **44 passed + 1 skipped**.
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
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclo 5 + 5.1 + 5.2 +
   5.3 + 5.4 + 5.5. Lee especialmente la sección 15 (rotación manual).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (2026-05-19)

**Ciclos 1-5.5 ✅ COMPLETOS:**

- App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer.
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- **44 tests** pasando (1 gated por checkpoints locales).
- Mecanismo reproducible de updates: `scripts/upload_to_space.py`.
- UX clínica Cobb (Ciclo 5 + 5.1 + 5.2): Assessment binary, viz tipo
  Fig 1 Shi et al., multi-curve detection, label-transfer del multiclass.
- Cobertura del binary (Ciclo 5.3): bloque `=== COVERAGE ===`, Assessment
  "Inconclusive" cuando partial + 0°.
- Robustez de rotación + UX viz (Ciclo 5.4): bloque `=== ROTATION WARNING ===`
  cuando SVD del skeleton ve tilt > 12°. Filtros degeneradas + dedup
  rótulos.
- **Control manual rotación (Ciclo 5.5):** slider -180..180 + 5 botones
  rápidos bajo la imagen. El médico decide cuánto rotar; el pipeline
  analiza la imagen tal como queda. **N_61 con +13° → 0° Normal**
  correctamente. El warning del 5.4 sigue activo como guía post-rotación.

### Lo que sigue (Ciclo 6)

El Ciclo 5 + sus 5 addenda están cerrados. El Ciclo 6 NO tiene brief
todavía. Candidatos identificados:

1. **Live preview de la rotación**. Hoy el slider modifica el valor pero
   la imagen no se rota visualmente hasta presionar Analyze. Con `gr.State`
   guardando la imagen original + un callback en `rotation_slider.change`
   se puede mostrar el preview en tiempo real. ~30 LOC.
2. **Recalibrar threshold de tilt** (12° → 14-15°). S_100/S_150 (escoliosis
   severas con tilt ~12.6°) disparan el ROTATION WARNING aunque la
   curva es real. Confunde post-rotación manual.
3. **Reset slider al cargar nueva imagen**. Hoy si rotas la primera y
   subes otra, el slider sigue en el valor anterior. Pequeño UX bug.
4. **Fallback multiclass cuando binary cubre poco** (fix E del Ciclo 5.3).
   Útil para casos donde el binary no detecta zona lumbar.
5. **Reentrenamiento con augmentation lumbar agresivo** — CLAHE, crop
   variable, gamma, contrast jitter centrados en el tercio inferior.
   Considerar pre-training RadImageNet.
6. **Mejorar explicabilidad** — enmascarar el confidence map por la
   máscara predicha + probar Seg-Grad-CAM en vez de Grad-CAM vanilla.
7. **Recalibrar el umbral `is_partial` del Coverage** (Ciclo 5.3). El
   default actual marca S_21 como parcial aunque cubre la columna
   completa (es padding lo que reduce el ratio). Usar bounding box real
   de la columna como denominador.
8. **Flip horizontal/vertical en la UI** — para radiografías espejadas.
9. **Quantización INT8 para tablet** (modelo ligero para edge).
10. **CI con GitHub Actions** — automatizar `pytest` + lint en cada PR.
11. **Sustentación oral + slides + demo en vivo + smoke test cross-device.**
12. **Artículo IEEE/ACM** — si los resultados del refinamiento lo soportan.

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
- **Pasar `--path-in-repo` explícito** en `upload_to_space.py` cuando varios
  archivos tienen mismo basename (e.g. `app.py` raíz vs `spine_segmentation/deployment/app.py`).
- Tests locales: `pytest tests/ -v` debe seguir verde (44/45).
- Si surgen ideas tomadas de papers: revisar críticamente qué es aplicable a
  nuestro dataset (174 imgs, single-rater, sin landmarks de endplate) vs qué
  requiere recursos que no tenemos. Honestidad ante todo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktree del Ciclo 5.5: cerrado y merged. Próximo ciclo creará su propio
  worktree desde main.

## ↑ HASTA AQUÍ
