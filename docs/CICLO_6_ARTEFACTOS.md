# Ciclo 6.0 — Pre-sustentación · Artefacto de Salida

> **Fecha de cierre:** 2026-05-22 (sustentación al día siguiente 2026-05-23).
> **Estado:** ✅ COMPLETO.
> **URL pública:** https://huggingface.co/spaces/ElvLandau/spine-segmentation.
> **Próximo ciclo:** 6.1+ post-sustentación (refinamiento del modelo,
> reentrenamiento con augmentation, posible artículo IEEE/ACM).

---

## 1. Resumen ejecutivo

Ciclo dedicado a cerrar gaps de visibilidad / accesibilidad del proyecto
antes de la defensa oral del día siguiente. Sin cambios de código —
puro polish de documentación y estructura del repo para que el jurado
encuentre todo sin fricción y Elvis pueda explicar el proyecto end-to-end
sin titubear.

Hallazgos pre-cycle (del audit Read-only):

1. **Bug latente del README** (desde Ciclo 4): 8 ocurrencias del
   placeholder `<usuario>/spine-segmentation` con la nota "por completar
   tras crear el Space" — un jurado leyendo eso concluye que la app NO
   está desplegada (cuando lleva semanas RUNNING).
2. **Carpetas requeridas por rúbrica ausentes**: `modelos/` y `datos/`
   no existían como carpetas raíz visibles en GitHub.
3. **Notebook alternativo del equipo** (Julian, ~5 MB) no commiteado.
4. **Discrepancia de métricas paper vs repo** (Dice binario 0.88 vs
   multiclass 0.34) sin narrativa explícita para Q&A.

## 2. Cumplimiento de rúbrica Coursera/U. Andes (post-Ciclo 6.0)

| Criterio | Peso | Estado | Evidencia |
|---|:-:|---|---|
| Funcionamiento | 35% | ✅ | App RUNNING 24/7, smoke verde en 10+ casos |
| Documentación (README) | 15% | ✅ | URL real visible en primera tabla; descripción + dependencias + 12 env vars + 2 opciones de deploy |
| Parametrización | 15% | ✅ | `.env.example` + `config.py` con 12 variables |
| Usabilidad / UI | 35% | ✅ | 4 tabs, toggle ES/EN, reference image educativa, disclaimer |

Estructura de carpetas (post-Ciclo 6.0):

| Carpeta | Estado | Notas |
|---|---|---|
| `notebooks/` | ✅ | 4 notebooks (EDA, training principal, training alternativo Keras, informe final) + README |
| `modelos/` | ✅ nueva | README explica que pesos viven en HF Hub (226 MB) |
| `datos/` | ✅ nueva | README explica que dataset es propiedad U. Andes |

## 3. Unidades de trabajo y commits

| # | Unidad | Commit |
|---|---|---|
| 6.0.1 | Fix README placeholders | `c352db3` |
| 6.0.2 | Carpetas `modelos/` + `datos/` con READMEs | `1f75970` |
| 6.0.3 | Notebook alternativo Keras + `notebooks/README.md` | `4183533` |
| 6.0.4 | `docs/SUSTENTACION_GUIA.md` | `364733b` |
| 6.0.5 | Cierre docs (AGENTS + este artefacto + DECISIONS + PROMPT) | (este commit) |

## 4. Recursos producidos

### Documentación nueva
- `modelos/README.md` (~70 LOC): pesos en HF Hub, cómo descargar, cómo
  subir nuevos.
- `datos/README.md` (~60 LOC): dataset propiedad U. Andes, estructura
  esperada, contacto.
- `notebooks/README.md` (~50 LOC): tabla de los 4 notebooks, qué hace
  cada uno, cuál alimenta el deploy.
- `docs/SUSTENTACION_GUIA.md` (~476 LOC): **el más valioso del ciclo**.
  Guía operativa para la defensa oral.
- `docs/CICLO_6_ARTEFACTOS.md` (este archivo).

### Documentación modificada
- `README.md`: 8 reemplazos de `<usuario>` → `ElvLandau`.
- `docs/HF_SPACES_SETUP.md`: 12 reemplazos del mismo placeholder.
- `AGENTS.md`: sec 5 (entrada Ciclo 6.0) + sec 9 (5 decisiones nuevas)
  + header (link a SUSTENTACION_GUIA).
- `docs/DECISIONS.md`: 1 fila nueva para Ciclo 6.0.
- `docs/PROMPT_PROXIMO_CHAT.md`: refresh post-sustentación.

### Notebook nuevo (commiteado)
- `notebooks/02b_training_alternativo_unet_keras.ipynb` (4.8 MB): pipeline
  Keras/TensorFlow del equipo, U-Net + ResNet50 simple en Colab.

## 5. Métricas del ciclo

- **Duración**: ~3 h (planificación + exploración + implementación + close).
- **Commits**: 5 (sin Co-Authored-By IA, conforme política).
- **Líneas añadidas**: ~830 (excluyendo el .ipynb binario).
- **Líneas eliminadas**: ~20 (sed de placeholders).
- **Archivos modificados**: 5.
- **Archivos nuevos**: 6.
- **Tests passing**: 66/67 (sin regresión — este ciclo no toca código).
- **NO hubo deploy al Space** — ciclo solo de docs.

## 6. Decisiones registradas (en `AGENTS.md` sec 9)

| Decisión | Razón |
|---|---|
| Fix README `<usuario>` → `ElvLandau` | Cerrar bug latente del Ciclo 4 que confundía al jurado |
| Crear `modelos/`+`datos/` con READMEs explicativos en vez de subir artifacts | Cumple rúbrica letra sin duplicar bytes; READMEs dan contexto en 30 seg |
| Commitear notebook Julian como `02b_*alternativo*` | Trabajo del equipo, aporta comparativa Keras. Convención de nombre comunica "es variante, no el desplegado" |
| `SUSTENTACION_GUIA.md` operativo | Guía paso-a-paso para que Elvis no titubee en demo + Q&A |

## 7. Lo que NO entró en este ciclo (deferido a post-sustentación)

- **Reentrenamiento del modelo** con augmentation lumbar agresivo.
- **Recalibración de threshold de tilt** (12° → 14-15°).
- **Fix E**: fallback multiclass cuando binary cubre poco.
- **Seg-Grad-CAM auténtico** (no solo masking del 5.8).
- **Quantización INT8** para tablet.
- **CI con GitHub Actions** para pinning automático.
- **Cualquier cambio de código** que pudiera introducir regresión
  pre-sustentación.

## 8. Handoff al Ciclo 6.1+ (post-sustentación)

### Lo que el Ciclo 6.1+ puede asumir como dado
- App pública RUNNING en HF Spaces con la convención clínica del Ciclo
  5.10 + arrows del reference image fijados (Ciclo 5.12).
- Repo público con estructura completa post-rúbrica (3 carpetas raíz +
  notebooks + docs + tests).
- Suite de 66 tests pytest.
- `docs/SUSTENTACION_GUIA.md` como documento de referencia para
  consultar decisiones rápido (es como un FAQ del proyecto).
- `docs/DECISIONS.md` como índice navegable.
- Feedback de la sustentación (a recoger inmediatamente después).

### Candidatos para el Ciclo 6.1+
- Incorporar feedback específico del jurado (priorizar lo solicitado).
- Reentrenamiento si la limitación de Dice multiclass 0.34 fue
  comentada negativamente.
- Mejoras de UX adicionales si el jurado pidió algo específico.
- Artículo IEEE/ACM si los resultados + feedback lo soportan.

## 9. Referencias

- [`CICLO_5_ARTEFACTOS.md`](CICLO_5_ARTEFACTOS.md) — historial de los
  12 sub-ciclos previos.
- [`SUSTENTACION_GUIA.md`](SUSTENTACION_GUIA.md) — guía operativa para
  la defensa oral.
- [`DECISIONS.md`](DECISIONS.md) — índice navegable de decisiones.
- [`../README.md`](../README.md) — visión general (rúbrica-compliant).
- [`../AGENTS.md`](../AGENTS.md) — memoria persistente.
- [`../WORKFLOW.md`](../WORKFLOW.md) — reglas del repo.
- [`Informe_final_escoliosis_IEEE.pdf`](Informe_final_escoliosis_IEEE.pdf) —
  paper IEEE consolidado del equipo.
