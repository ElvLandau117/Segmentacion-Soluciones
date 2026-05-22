# Prompt para el próximo chat — Ciclo 6.2+ (post-sustentación)

> **Estado al 2026-05-22:** Ciclos 1, 2, 3, 4, 5, 5.1..5.12, 6.0, 6.1 ✅ COMPLETOS.
> Sustentación oral del 2026-05-23 ya pasó. El Ciclo 6.1 (lateralidad por
> chord signed-area) cerró post-sustentación tras feedback de la médica
> colaboradora. El próximo chat debe arrancar definiendo el alcance del
> Ciclo 6.2 con Elvis (candidato principal: bug del orden upper/lower en
> `assign_vertebra_names_to_curves` — "T6-T5" en curva secundaria).
>
> **Atajos rápidos**:
> - `docs/SUSTENTACION_GUIA.md` — guía operativa usada en la defensa (sec 8 Q&A #6 actualizada con Ciclo 6.1)
> - `docs/DECISIONS.md` — índice navegable de decisiones por ciclo
> - `docs/CICLO_5_ARTEFACTOS.md` sec 23 — addendum 6.1 (lateralidad)
> - `docs/CICLO_6_ARTEFACTOS.md` — qué entregó el Ciclo 6.0

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (sección 5 = estado de los ciclos).
   Al cierre del Ciclo 6.1, todos los ciclos 1–6.1 están ✅ completos.
   **Atajo para entender decisiones rápido**: `docs/DECISIONS.md` (índice
   navegable por ciclo + tema).
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_5_ARTEFACTOS.md` sec 23 — último addendum cerrado (Ciclo
   6.1, fix de lateralidad por chord signed-area, 2026-05-22). Antes de
   eso, sec 22 (Ciclo 5.12) y sec 20 (Ciclo 5.10, fix original de
   lateralidad que fue refinado por el 6.1).
4. `docs/CICLO_6_ARTEFACTOS.md` — qué se entregó en el Ciclo 6.0
   (pre-sustentación).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (al cierre del Ciclo 6.1 — 2026-05-22 post-sustentación)

✅ App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
✅ Toggle ES/EN funcional (default Español).
✅ Pestaña Cobb Angle: detección multi-curva, viz tipo Fig 1 Shi et al.,
   slider de rotación con live preview + 5 botones rápidos.
✅ **Lateralidad clínica (Ciclo 5.10 + refinamiento 6.1)** — convención
   anatomía del paciente vía algoritmo chord signed-area. S-shapes
   reportan convexidades opuestas correctamente. Sweep visual sobre 12
   casos: 6/7 S-shapes cumplen el principio del IP + 3/3 contra GT
   oficial. Tests anchored a `MaIA/RadiographMetrics/`.
✅ Pestaña Explainability: imagen fija + panel dinámico (5.8, 5.9, 5.11, 5.12).
✅ Suite pytest: **73 passed + 1 skipped**.
✅ main local + origin sincronizados.

### Tarea para este chat: Ciclo 6.2 — Pendiente de brief

Elvis aún no priorizó los candidatos del Ciclo 6.2. **Antes de implementar
nada, define el alcance con él.** Candidatos identificados (en orden
tentativo de valor clínico vs esfuerzo):

| # | Candidato | Esfuerzo | Valor |
|---|---|---|---|
| 0 | **Fix orden upper/lower en `assign_vertebra_names_to_curves`** (caso reportado: secundaria sale como "T6-T5", upper > lower) | Bajo | Bug visible al médico. Vive en otra función que `_curve_direction`. Quedó documentado en el addendum 6.1 como known issue. |
| 1 | **Recalibrar threshold de tilt** (12° → 14-15°) tras observar S_100/S_150/S_158 borderline | Bajo (1 PR) | Reduce falsos positivos del ROTATION WARNING en escoliosis severa. S_158 con tilt 13.4° sigue gatillando warning. |
| 2 | **Fix E — fallback multiclass cuando binary cubre poco** | Medio | Recupera casos donde el coverage warning aparece pero el multiclass sí tiene cobertura completa |
| 3 | **Recalibrar umbral `is_partial` del Coverage** (sobre-sensible — S_21 marca WARNING cubriendo ~95%) | Bajo | UX cleanup |
| 4 | **Toggle de convención de imagen** (AP estándar vs flipped) | Medio | Cubre el caso de radiografías que vienen flipped del equipo radiológico |
| 5 | **Reentrenamiento con augmentation lumbar agresivo** (CLAHE, crop variable, gamma, contrast jitter) | Alto (8-16h GPU) | Sube el Dice del binary en zona lumbar → menos falsos negativos tipo S_22 pre-5.3 |
| 6 | **Seg-Grad-CAM auténtico** (no solo masking del 5.8) | Medio | Activaciones espaciales más precisas para segmentación densa |
| 7 | **Colorbar con valores numéricos** (0.25, 0.5, 0.75) en lugar de solo Alta/Baja | Bajo | UX cleanup del panel dinámico |
| 8 | **Anotaciones numéricas sobre hot-spots del CAM real** | Medio | Cuantifica lo que la imagen fija del 5.9/5.11 sólo describe cualitativamente |
| 9 | **Flip horizontal/vertical en la UI** para radiografías espejadas | Bajo | Cubre caso raro pero molesto |
| 10 | **i18n Nivel C — traducir tabs, slider y botones rápidos** | Medio (requiere recrear Blocks) | Completa la experiencia bilingüe |
| 11 | **Quantización INT8** para tablet | Medio-Alto | Cumple la promesa "modelo ligero offline" del README |
| 12 | **CI con GitHub Actions** | Bajo | Pinning de los 73 tests en cada PR |
| 13 | **Artículo IEEE/ACM** si los resultados lo soportan | Muy alto | Aporte científico |

### Acción recomendada para arrancar

1. Leer los 5 archivos del onboarding.
2. Preguntar a Elvis cuáles candidatos quiere abordar y en qué orden.
3. Una vez priorizados, **crear el brief** del Ciclo 6.2 en
   `docs/CICLO_6_BRIEF.md` antes de tocar código.
4. Crear worktree `cycle6_2-<scope>` con branch desde `main`.
5. Implementar las unidades en orden, commit por unidad.
6. Cerrar: AGENTS.md sec 5 + sec 9 + `docs/CICLO_5_ARTEFACTOS.md`
   addendum 24 (o `docs/CICLO_6_ARTEFACTOS.md` si Elvis prefiere
   separar) + refresh este `docs/PROMPT_PROXIMO_CHAT.md`.
7. Merge fast-forward a main + cleanup branch + worktree + push.

### Restricciones operativas (no negociables)

- Metodología: **Spec-Driven Work + Work Orchestration** (Leonardo Gonzalez).
- Cada cambio = 1 commit con convención de `WORKFLOW.md` sección 4.
- Commits **sin** co-autoría de IA (solo `Elvis Hernandez`).
- **NO** force push a `main` ni al Space.
- **NO** `git push hf` para parchar el Space — usar
  `python scripts/upload_to_space.py` con `--path-in-repo` explícito.
- Tests locales: `pytest tests/ -v` debe seguir verde (73 + 1 actual,
  esperado +N al cerrar Ciclo 6.2).
- Si vas a hacer un cambio de lateralidad o algoritmo de Cobb, corre
  primero `scripts/sweep_laterality.py` como baseline + post-fix —
  patrón establecido en Ciclo 6.1.
- Honestidad ante todo: si una idea no funciona, decirlo y planear
  retrabajo en vez de marcar la tarea como completa.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`
- Branch principal: `main` (sincronizado con `origin/main`)
- Worktrees activos: ver `git worktree list`

## ↑ HASTA AQUÍ
