# Prompt para el próximo chat — Ciclo 6 (refinamiento del modelo + entrega)

> **Estado al 2026-05-20:** Ciclos 1, 2, 3, 4, 5, 5.1..5.11 ✅ COMPLETOS.
> Ciclo 6 NO tiene brief todavía — el primer paso del próximo chat es
> definirlo y aprobarlo con Elvis antes de implementar.

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (sección 5 = estado de los ciclos).
   Al cierre del Ciclo 5.11, todos los ciclos 1–5.11 están ✅ completos.
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclos 5 + 5.1..5.11.
   La **sección 20** describe el último cambio cerrado (fix de convención
   de lateralidad clínica + sample S_200 en la reference image). La
   **sección 19** describe el cambio inmediato anterior (imagen fija de
   referencia clínica bilingüe).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (al cierre del Ciclo 5.11)

✅ App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
✅ Toggle ES/EN funcional (default Español) — header markdown +
   explainability markdown + diagnosis report + **reference image
   bilingüe (Ciclo 5.9, base S_200 desde 5.10, arrows derivados del
   spine bbox desde 5.11)** son todos sensibles al toggle.
✅ Pestaña Cobb Angle: detección multi-curva, viz tipo Fig 1 Shi et al.,
   slider de rotación con live preview + 5 botones rápidos. **Convexity
   en convención clínica (anatomía del paciente, no perspectiva del
   viewer) desde el Ciclo 5.10** — S_158 ahora reporta "convexidad
   derecha" correctamente.
✅ Pestaña Explainability:
   - Imagen FIJA arriba con 5 callouts numerados + colorbars +
     disclaimer (Ciclo 5.9, recreación programática del mockup; base
     `S_200.jpg` desde Ciclo 5.10; flechas sample-invariantes derivadas
     del spine bbox desde Ciclo 5.11).
   - Panel DINÁMICO abajo: Grad-CAM + Confidence enmascarados por la
     columna detectada, percentile-clip p95, títulos in-image +
     colorbars verticales (Ciclo 5.8).
   - Markdown explicativo "Cómo leerlo / How to read it" bilingüe.
✅ Suite pytest: **63 passed + 1 skipped**.
✅ main local + origin sincronizados.

### Tarea para este chat: Ciclo 6 — Pendiente de brief

Elvis aún no priorizó los candidatos del Ciclo 6. **Antes de implementar
nada, define el alcance con él.** Candidatos identificados a lo largo
del Ciclo 5 (en orden tentativo de valor clínico vs esfuerzo):

| # | Candidato | Esfuerzo | Valor |
|---|---|---|---|
| 0 | **Recalibrar threshold de tilt** (12° → 14-15°) tras observar S_100/S_150/S_158 borderline | Bajo (1 PR) | Reduce falsos positivos del ROTATION WARNING en escoliosis severa. S_158 con tilt 13.4° sigue gatillando warning. |
| 2 | **Fix E — fallback multiclass cuando binary cubre poco** | Medio | Recupera casos donde el coverage warning aparece pero el multiclass sí tiene cobertura completa |
| 3 | **Recalibrar umbral `is_partial` del Coverage** (sobre-sensible — S_21 marca WARNING cubriendo ~95%) | Bajo | UX cleanup |
| 4 | **Reentrenamiento con augmentation lumbar agresivo** (CLAHE, crop variable, gamma, contrast jitter) | Alto (8-16h GPU) | Sube el Dice del binary en zona lumbar → menos falsos negativos tipo S_22 pre-5.3 |
| 5 | **Seg-Grad-CAM auténtico** (no solo masking del 5.8) | Medio | Activaciones espaciales más precisas para segmentación densa |
| 6 | **Colorbar con valores numéricos** (0.25, 0.5, 0.75) en lugar de solo Alta/Baja | Bajo | UX cleanup del panel dinámico |
| 7 | **Anotaciones numéricas sobre hot-spots del CAM real** (e.g., "X% activación máx aquí") | Medio | Cuantifica lo que la imagen fija del 5.9/5.11 sólo describe cualitativamente |
| 8 | **Flip horizontal/vertical en la UI** para radiografías espejadas | Bajo | Cubre caso raro pero molesto |
| 9 | **i18n Nivel C — traducir tabs, slider y botones rápidos** | Medio (requiere recrear Blocks) | Completa la experiencia bilingüe |
| 10 | **Live preview de la SEGMENTACIÓN** (no solo rotación) | Alto | Feedback inmediato; costoso en CPU |
| 11 | **Quantización INT8** para tablet | Medio-Alto | Cumple la promesa "modelo ligero offline" del README |
| 12 | **CI con GitHub Actions** | Bajo | Pinning de los 62 tests en cada PR |
| 13 | **Sustentación oral + slides + demo en vivo + smoke test cross-device** | Alto | Entrega académica |
| 14 | **Artículo IEEE/ACM** si los resultados lo soportan | Muy alto | Aporte científico |

### Acción recomendada para arrancar

1. Leer los 5 archivos del onboarding (AGENTS.md, WORKFLOW.md,
   `docs/CICLO_5_ARTEFACTOS.md` sec 19, `docs/CICLO_4_ARTEFACTOS.md`,
   `README.md`).
2. Preguntar a Elvis cuáles candidatos quiere abordar y en qué orden.
3. Una vez priorizados, **crear el brief** del Ciclo 6 en
   `docs/CICLO_6_BRIEF.md` antes de tocar código.
4. Crear worktree `cycle6` con branch `cycle6-<scope>` desde `main`.
5. Implementar las unidades en orden, commit por unidad.
6. Cerrar: AGENTS.md sec 5 + sec 9 + `docs/CICLO_6_ARTEFACTOS.md` +
   refresh este `docs/PROMPT_PROXIMO_CHAT.md`.
7. Merge fast-forward a main + cleanup branch + worktree + push.

### Restricciones operativas (no negociables)

- Metodología: **Spec-Driven Work + Work Orchestration** (Leonardo Gonzalez).
- Cada cambio = 1 commit con convención de `WORKFLOW.md` sección 4.
- Commits **sin** co-autoría de IA (solo `Elvis Hernandez`).
- **NO** force push a `main` ni al Space.
- **NO** `git push hf` para parchar el Space — usar
  `python scripts/upload_to_space.py`.
- **Siempre pasar `--path-in-repo`** explícito en uploads.
- Tests locales: `pytest tests/ -v` debe seguir verde (65/66 actual,
  esperado +N al cerrar Ciclo 6).
- Honestidad ante todo: si una idea no funciona, decirlo y planear
  retrabajo en vez de marcar la tarea como completa.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`
- Branch principal: `main` (sincronizado con `origin/main`)

## ↑ HASTA AQUÍ
