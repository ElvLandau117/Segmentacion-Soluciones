# Prompt para el próximo chat — Ciclo 4 cerrado, pendiente brief de Ciclo 5

> **Estado al 2026-05-17 noche:** Ciclo 4 ✅ COMPLETO. Bug `gradio-client
> TypeError api_info` resuelto via upgrade a Gradio 5.50.0 + bump de
> `huggingface_hub` a `>=0.33.5`. Space corriendo end-to-end, smoke test
> automatizado verde (gradio-client predict en ~13 s).
>
> **No hay handoff técnico pendiente.** El próximo chat debe definir el scope
> del Ciclo 5 (refinamiento del modelo y/o entrega académica) y escribir el
> brief correspondiente en `docs/CICLO_5_BRIEF.md`.

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (ver sección 5 para el estado de los ciclos).
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (incluyendo el fix
   del bug `api_info`).
4. `README.md` — visión general del proyecto + URL pública del Space.

### Estado actual (2026-05-17 noche)

**Ciclo 4 ✅ COMPLETO:**

- App pública en `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
- 4 tabs operativos (Binary, Vertebrae, Cobb, Explainability) con disclaimer médico.
- Pesos en `ElvLandau/spine-checkpoints` (DeepLabV3+ multiclase + UNet binario).
- Gradio 5.50.0 + huggingface_hub>=0.33.5 + Python 3.11.
- `scripts/upload_to_space.py` para parchar el Space vía `HfApi` sin tocar git remoto.
- 14 tests (13 passing + 1 gated por checkpoints locales).
- Smoke test automatizado verde (predict via gradio-client → 4 overlays + texto en ~13 s).

### Lo que sigue (Ciclo 5)

El Ciclo 4 está cerrado. El Ciclo 5 NO tiene brief todavía. Antes de programar,
**decidamos juntos el scope** considerando las limitaciones documentadas en
`docs/CICLO_4_ARTEFACTOS.md` sec 7. Candidatos:

1. **Refinamiento del modelo** — augmentation agresiva, pre-training con
   RadImageNet, ensemble; objetivo: subir el Dice promedio de ~0.30 y mejorar
   detección de C3.
2. **Mejora del Cobb multiclase** — método endplate actual devuelve 90°
   degenerado en pacientes normales (limitación arctan, `AGENTS.md` sec 4.5);
   alternativas: agregar guard, usar ensemble de los 2 métodos, fitting robusto.
3. **Quantización INT8 para tablet** — modelo ligero para edge.
4. **Sustentación oral + demo en vivo** — preparar slides, captura del Space
   funcionando, casos clínicos de demostración.
5. **Smoke test cross-device manual** — móvil, tablet, desktop con capturas.
6. **CI con GitHub Actions** — opcional, no requerido por rúbrica.

**Acción recomendada para el próximo chat:**

1. Leer los 4 archivos del onboarding.
2. Conversar con Elvis para priorizar 1-3 items del Ciclo 5.
3. Escribir `docs/CICLO_5_BRIEF.md` con la spec (siguiendo el patrón de
   `docs/CICLO_4_DESPLIEGUE_BRIEF.md`).
4. Empezar la primera unidad de trabajo del Ciclo 5.

### Restricciones operativas (recordatorio)

- Metodología: **Spec-Driven Work + Work Orchestration** (Leonardo Gonzalez).
- Cada cambio = 1 commit con convención de `WORKFLOW.md` sección 4.
- Commits **sin** co-autoría de IA (sólo `Elvis Hernandez`).
- **NO** force push a `main` ni al Space.
- **NO** `git push hf` para parchar el Space — usar `python scripts/upload_to_space.py`.
- Tests locales: `pytest tests/ -v` debe seguir verde (13/14).

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Branch principal: `main` (sincronizado con `origin/main`)

## ↑ HASTA AQUÍ
