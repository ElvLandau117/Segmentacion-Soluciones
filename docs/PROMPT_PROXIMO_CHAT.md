# Prompt para el próximo chat — Ciclo 5.9 (imagen fija de referencia clínica)

> **Estado al 2026-05-20:** Ciclos 1, 2, 3, 4, 5, 5.1, 5.2, 5.3, 5.4,
> 5.5, 5.6, 5.7, 5.8 ✅ COMPLETOS. Ciclo 5.9 planeado y aprobado por
> Elvis — añadir una imagen fija de referencia (mockup educativo del
> compañero médico) en la pestaña Explainability, bilingüe ES + EN.
> El plan completo vive en
> `C:\Users\User\.claude\plans\estas-en-modo-planificacion-piped-crayon.md`.

---

## PEGAR DESDE AQUÍ ↓

Hola Claude. Soy **Elvis Hernández**. Continuamos el proyecto de
**Segmentación Vertebral para diagnóstico de escoliosis** (Maestría en IA,
U. Andes, Coursera).

### Onboarding obligatorio (lee primero en este orden)

1. `AGENTS.md` — memoria persistente (sección 5 = estado de los ciclos).
   Al cierre del Ciclo 5.8, todos los ciclos 1–5.8 están ✅ completos.
2. `WORKFLOW.md` — reglas no negociables del repo.
3. `docs/CICLO_5_ARTEFACTOS.md` — qué se entregó en Ciclos 5 + 5.1..5.8.
   Lee especialmente la **sección 18** (Explainability polish del 5.8:
   masking de Grad-CAM + Confidence, colorbars, markdown bilingüe).
4. `docs/CICLO_4_ARTEFACTOS.md` — qué se entregó en el Ciclo 4 (despliegue).
5. `README.md` — visión general + URL pública del Space.

### Estado actual (al cierre del Ciclo 5.8)

✅ App pública: `https://huggingface.co/spaces/ElvLandau/spine-segmentation`.
✅ Toggle ES/EN funcional (default Español) — header markdown +
   explainability markdown + diagnosis report bilingües.
✅ Pestaña Cobb Angle: detección multi-curva, viz tipo Fig 1 Shi et al.,
   slider de rotación con live preview + 5 botones rápidos.
✅ Pestaña Explainability (5.8): Grad-CAM y Confidence enmascarados por
   la columna detectada, percentile-clip p95 en cam, títulos in-image
   + colorbars verticales por subpanel.
✅ Suite pytest: 60 passed + 1 skipped.
✅ main local + origin sincronizados en `b2d293b`.

### Tarea para este chat: Ciclo 5.9 — Imagen fija de referencia clínica en Explainability

Mi compañero médico hizo un **mockup educativo** del panel
Explainability con 5 anotaciones numeradas (hot-spots, trayectoria
esperada, activaciones fuera de spine, zonas de alta confianza, bordes
de menor certeza) + colorbars con interpretación + disclaimer. Quiero
dejarlo **fijo** en la pestaña Explainability del Space — siempre
visible, encima del panel dinámico — para que cualquier médico que abra
la app sepa cómo interpretar lo que ve.

**Acción inmediata**: te adjunto la imagen del mockup en español. Tu
trabajo es:

1. **Guardar la imagen ES** en
   `spine_segmentation/deployment/assets/explainability_reference_es.png`
   (crea la carpeta `assets/` si no existe).

2. **Generar la versión en inglés** con las mismas strings traducidas
   (el plan en
   `C:\Users\User\.claude\plans\estas-en-modo-planificacion-piped-crayon.md`
   tiene la tabla ES → EN de todas las strings del mockup, en la sección
   "Strings del mockup a traducir"). Guárdala en
   `spine_segmentation/deployment/assets/explainability_reference_en.png`.

3. **Wire el componente UI** en
   `spine_segmentation/deployment/app.py`: añadir un `gr.Image` con la
   imagen de referencia **arriba** del `explain_output` dinámico
   actual. El componente debe ser `interactive=False`,
   `show_download_button=False`, `height=300`.

4. **Añadir helper en i18n**:
   `explain_reference_path(lang)` retorna el path al PNG correcto.

5. **Extender** `language_radio.change` para que también actualice el
   `reference_image.value` cuando el usuario togglea ES/EN.

6. **Tests**: 2 nuevos en `tests/test_app_smoke.py`:
   - Ambos PNGs existen y son no-vacíos.
   - `explain_reference_path("es")` y `("en")` retornan paths
     distintos.

7. **Deploy** via `scripts/upload_to_space.py` (con
   `--path-in-repo` explícito para cada archivo — lección del 5.5).

8. **Cierre**: actualizar AGENTS.md sec 5 + sec 9, añadir Addendum sec
   19 a `docs/CICLO_5_ARTEFACTOS.md`, refresh este
   `docs/PROMPT_PROXIMO_CHAT.md`, merge fast-forward a main, push a
   origin.

**Para la generación de la versión EN**: prefiero approach A (PIL
overlay sobre la ES) si las coordenadas de texto son claras. Si no,
approach B (recrear con matplotlib). La consistencia visual con el
mockup original es importante para que se mantenga el "look médico"
del diseño.

**Mimetizar la viz real (panel dinámico)**: NO en este ciclo. La
imagen fija es suficiente como leyenda; nuestro panel dinámico ya está
bien con el masking del 5.8. Si después de probar Elvis quiere
anotaciones numéricas sobre los hot-spots del cam, eso es Ciclo 6.

### Strings del mockup a traducir (referencia rápida)

| Español (original) | English (traducción) |
|---|---|
| 1. Mayor activación del modelo | 1. Highest model activation |
| Zonas cálidas (rojo/amarillo) indican las regiones que más influyeron en la predicción. | Warm zones (red/yellow) show the regions that most influenced the prediction. |
| 2. Trayectoria esperada de la columna | 2. Expected spine trajectory |
| El modelo centra su atención en la columna vertebral, siguiendo su curvatura anatómica. | The model focuses on the vertebral column, following its anatomical curvature. |
| 3. Activación fuera de la región anatómica | 3. Activation outside the anatomical region |
| Se observa activación en zonas externas a la columna (bordes, artefactos o estructuras no relevantes). | Activation appears outside the spine (borders, artifacts, or irrelevant structures). |
| 4. Alta confianza en la trayectoria segmentada | 4. High confidence in the segmented trajectory |
| Las zonas verdes indican alta certeza del modelo en la predicción píxel a píxel. | Green zones indicate high model certainty in the pixel-by-pixel prediction. |
| 5. Bordes de menor certeza | 5. Lower-certainty edges |
| Los bordes (amarillo/naranja) presentan menor confianza y son críticos para el cálculo del ángulo de Cobb. | Edges (yellow/orange) show lower confidence and are critical for the Cobb angle calculation. |
| Menor influencia ◀ Mayor influencia | Lower influence ◀ Higher influence |
| Baja confianza ◀ Alta confianza | Low confidence ◀ High confidence |
| Grad-CAM (izquierda): Regiones que influyeron en la decisión del modelo. Zonas cálidas (rojo) = alta influencia. | Grad-CAM (left): Regions that influenced the model's decision. Warm zones (red) = high influence. |
| Mapa de Confianza (derecha): Certeza del modelo por píxel. Verde = alta confianza | Rojo = baja confianza (el médico debe revisar). | Confidence Map (right): Per-pixel model certainty. Green = high confidence | Red = low confidence (clinician must review). |
| Este sistema es una herramienta de apoyo. No reemplaza el criterio del especialista. | This system is a support tool. It does not replace the specialist's judgment. |

### Restricciones operativas (no negociables)

- Metodología: **Spec-Driven Work + Work Orchestration** (Leonardo Gonzalez).
- Cada cambio = 1 commit con convención de `WORKFLOW.md` sección 4.
- Commits **sin** co-autoría de IA (solo `Elvis Hernandez`).
- **NO** force push a `main` ni al Space.
- **NO** `git push hf` para parchar el Space — usar
  `python scripts/upload_to_space.py`.
- **Siempre pasar `--path-in-repo`** explícito en uploads.
- Tests locales: `pytest tests/ -v` debe seguir verde (60/61 actual,
  esperado 62/63 al cerrar 5.9).
- Honestidad ante todo: si la generación de la versión EN no queda
  bien con approach A, probar B sin asumir éxito; si ninguno queda
  bien, decirlo y planear retrabajo.

### Contexto adicional

- Username HF: **ElvLandau**
- Username GitHub: **ElvLandau117**
- Token HF Write: en `~/.cache/huggingface/token`
- Hardware Space: CPU Basic free (2 vCPU, 16 GB RAM)
- Pesos en HF Hub: `ElvLandau/spine-checkpoints`
- Branch principal: `main` @ `b2d293b` (sincronizado con `origin/main`)

### Acción recomendada para arrancar

1. Leer los 5 archivos del onboarding (AGENTS.md, WORKFLOW.md,
   `docs/CICLO_5_ARTEFACTOS.md` sec 18, `docs/CICLO_4_ARTEFACTOS.md`,
   `README.md`).
2. Confirmar que recibiste la imagen ES del mockup (te la adjunto al
   inicio del chat).
3. Crear worktree `cycle5_9` con branch
   `cycle5_9-explainability-reference` desde `main`.
4. Implementar las 8 sub-tareas listadas arriba en orden.
5. Deploy + smoke remoto (verificar visualmente que ambas imágenes
   se ven bien y que el toggle ES/EN las cambia correctamente).
6. Cerrar: AGENTS.md sec 5 + addendum sec 19 en
   `docs/CICLO_5_ARTEFACTOS.md` + refresh este
   `docs/PROMPT_PROXIMO_CHAT.md`.
7. Merge fast-forward a main + cleanup branch + worktree.

### Candidatos diferidos para Ciclo 6 (después del 5.9)

- Recalibrar threshold de tilt (12° → 14-15°).
- Fix E: fallback multiclass cuando binary cubre poco.
- Reentrenamiento con augmentation lumbar agresivo (CLAHE, crop
  variable, gamma, contrast jitter).
- Seg-Grad-CAM auténtico (no solo el masking actual).
- Recalibrar el umbral `is_partial` del Coverage.
- Flip horizontal/vertical en la UI para radiografías espejadas.
- i18n Nivel C — traducir también tabs, slider y botones rápidos.
- Live preview de la SEGMENTACIÓN (no solo de la rotación).
- Colorbar con valores numéricos (0.25, 0.5, 0.75) en lugar de solo
  Alta/Baja.
- Quantización INT8 para tablet.
- CI con GitHub Actions.
- Sustentación oral + slides + demo en vivo + smoke test cross-device.
- Artículo IEEE/ACM si los resultados lo soportan.

## ↑ HASTA AQUÍ
