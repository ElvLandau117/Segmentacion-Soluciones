"""i18n strings for the Gradio app (Ciclo 5.7).

Two languages: Spanish (default, matches the project audience at U. Andes
in Colombia) and English (for international evaluators / committee).

Keys are short snake_case identifiers. Each one maps to BOTH languages.
Missing keys fall back to the key itself — visible regression aid that
nudges the developer to fill them in. Use `t(key, lang)` from app.py.
"""

from __future__ import annotations

import os

DEFAULT_LANG = "es"
SUPPORTED_LANGS = ("es", "en")

# Map of UI display label -> internal language code. Used by the Gradio
# radio component which speaks human labels.
LABEL_TO_LANG = {
    "Español": "es",
    "English": "en",
}


# ---------------------------------------------------------------------------
# Strings used by build_results_text (the diagnosis panel)
# ---------------------------------------------------------------------------
DIAGNOSIS_STRINGS = {
    "es": {
        # COBB block
        "cobb_block_header_curves": "=== ANGULO COBB - Curvas detectadas ===",
        "cobb_block_header_simple": "=== ANGULO COBB ===",
        "principal_label":  "Curva principal:  ",
        "secondary_label":  "Curva secundaria: ",
        "curve_n_label":    "Curva {n}:        ",
        "convex_right":     "convexidad derecha",
        "convex_left":      "convexidad izquierda",
        "convex_neutral":   "sin convexidad clara",
        "convex_unknown":   "direccion desconocida",
        "binary_method":    "Metodo binary:",
        "binary_no_curves": "(sin curvas clinicamente significativas sobre el umbral de ruido)",
        "binary_error":     "Metodo binary: ERROR -",
        # Fallback (only when binary failed)
        "multi_fallback_prefix": "Metodo multiclass (fallback):",
        "multi_error_prefix":    "Metodo multiclass: ERROR -",
        # COVERAGE block
        "coverage_header":   "=== COBERTURA ===",
        "coverage_covers":   "Mascara binary cubre:",
        "coverage_of":       "de",
        "coverage_vertebrae_word": "vertebras",
        "warning_lower_spine_prefix": "ADVERTENCIA: La columna inferior",
        "warning_upper_spine_prefix": "ADVERTENCIA: La columna superior",
        "warning_not_segmented_suffix": "NO fue segmentada — el angulo Cobb puede ser engañoso.",
        "warning_partial_generic":      "ADVERTENCIA: Cobertura parcial de la columna — el angulo Cobb puede ser engañoso.",
        "coverage_no_names":            "Mascara binary cubre ~{pct}% de la altura de la imagen (sin vertebras multiclass para nombrar el rango)",
        # ROTATION WARNING
        "rotation_header":     "=== ADVERTENCIA DE ROTACION ===",
        "rotation_line_1":     "La imagen parece estar inclinada {tilt:.1f} grados respecto a la vertical (umbral {threshold:.0f} grados).",
        "rotation_line_2":     "El metodo binary ajusta x = f(y) y puede reportar la rotacion como escoliosis.",
        "rotation_line_3":     "Recaptura con el paciente derecho, o use el slider de rotacion para enderezar la imagen.",
        # ASSESSMENT
        "assessment_header_principal": "=== EVALUACION (basada en Binary principal) ===",
        "assessment_header_binary":    "=== EVALUACION (basada en Binary) ===",
        "assessment_header_fallback":  "=== EVALUACION (basada en Multiclass fallback) ===",
        "assessment_inconclusive":     "No concluyente - cobertura binaria insuficiente para calcular Cobb. Revise la segmentacion antes de interpretar.",
        "assessment_normal":           "Normal (< 10 grados)",
        "assessment_mild":             "Escoliosis leve (10-25 grados)",
        "assessment_moderate":         "Escoliosis moderada (25-40 grados)",
        "assessment_severe":           "Escoliosis severa (> 40 grados)",
        "curves_total_prefix":  "Numero total de curvas detectadas:",
        "shape_double":         "doble curva (S-shape)",
        "shape_triple":         "triple curva",
        "shape_n_curves":       "{n} curvas",
        # VERTEBRAE DETECTED
        "vertebrae_header_prefix": "=== VERTEBRAS DETECTADAS",
        # Misc
        "no_image":      "Por favor cargue una radiografia.",
        "no_model":      "No hay modelo cargado. Por favor proporcione las rutas de los checkpoints.",
        # Explainability tab (Ciclo 5.8)
        "explain_title_gradcam":     "Grad-CAM (atencion del modelo)",
        "explain_title_confidence":  "Confianza (certeza del modelo)",
        "explain_colorbar_high":     "Alta",
        "explain_colorbar_low":      "Baja",
        # Rotation controls (Ciclo 6.2 — i18n del slider + boton Reset)
        "rotation_slider_label":   "Rotar imagen (grados). Negativo = sentido horario.",
        "rotation_reset_button":   "Reiniciar",
    },
    "en": {
        # COBB block
        "cobb_block_header_curves": "=== COBB ANGLE - Detected curves ===",
        "cobb_block_header_simple": "=== COBB ANGLE ===",
        "principal_label":  "Principal curve: ",
        "secondary_label":  "Secondary curve:",
        "curve_n_label":    "Curve {n}:        ",
        "convex_right":     "convex right",
        "convex_left":      "convex left",
        "convex_neutral":   "no clear convexity",
        "convex_unknown":   "unknown direction",
        "binary_method":    "Binary method:",
        "binary_no_curves": "(no clinically meaningful curves above the noise floor)",
        "binary_error":     "Binary method: ERROR -",
        # Fallback (only when binary failed)
        "multi_fallback_prefix": "Multiclass method (fallback):",
        "multi_error_prefix":    "Multiclass method: ERROR -",
        # COVERAGE block
        "coverage_header":   "=== COVERAGE ===",
        "coverage_covers":   "Binary mask covers:",
        "coverage_of":       "of",
        "coverage_vertebrae_word": "vertebrae",
        "warning_lower_spine_prefix":  "WARNING: Lower spine",
        "warning_upper_spine_prefix":  "WARNING: Upper spine",
        "warning_not_segmented_suffix": "NOT segmented — Cobb angle may be misleading.",
        "warning_partial_generic":      "WARNING: Partial spine coverage — Cobb angle may be misleading.",
        "coverage_no_names":            "Binary mask covers ~{pct}% of image height (no multiclass vertebrae to name the range)",
        # ROTATION WARNING
        "rotation_header":     "=== ROTATION WARNING ===",
        "rotation_line_1":     "Image appears tilted {tilt:.1f} deg from vertical (threshold {threshold:.0f} deg).",
        "rotation_line_2":     "The binary Cobb method fits x = f(y) and may report rotation as scoliosis.",
        "rotation_line_3":     "Re-capture with the patient straight, or use the rotation slider to straighten the image.",
        # ASSESSMENT
        "assessment_header_principal": "=== ASSESSMENT (based on Binary principal) ===",
        "assessment_header_binary":    "=== ASSESSMENT (based on Binary) ===",
        "assessment_header_fallback":  "=== ASSESSMENT (based on Multiclass fallback) ===",
        "assessment_inconclusive":     "Inconclusive - insufficient binary coverage to compute Cobb. Review segmentation before interpreting.",
        "assessment_normal":           "Normal (< 10 degrees)",
        "assessment_mild":             "Mild scoliosis (10-25 degrees)",
        "assessment_moderate":         "Moderate scoliosis (25-40 degrees)",
        "assessment_severe":           "Severe scoliosis (> 40 degrees)",
        "curves_total_prefix":  "Total curves detected:",
        "shape_double":         "double curve (S-shape)",
        "shape_triple":         "triple curve",
        "shape_n_curves":       "{n} curves",
        # VERTEBRAE DETECTED
        "vertebrae_header_prefix": "=== VERTEBRAE DETECTED",
        # Misc
        "no_image":      "Please upload a radiograph.",
        "no_model":      "No model loaded. Please provide checkpoint paths.",
        # Explainability tab (Ciclo 5.8)
        "explain_title_gradcam":     "Grad-CAM (model attention)",
        "explain_title_confidence":  "Confidence (model certainty)",
        "explain_colorbar_high":     "High",
        "explain_colorbar_low":      "Low",
        # Rotation controls (Ciclo 6.2 — slider + Reset button i18n)
        "rotation_slider_label":   "Rotate image (degrees). Negative = clockwise.",
        "rotation_reset_button":   "Reset",
    },
}


# ---------------------------------------------------------------------------
# Markdown blocks for the top-of-page intro
# ---------------------------------------------------------------------------
EXPLAIN_MARKDOWN = {
    "es": """
**Grad-CAM** (izquierda): regiones DENTRO de la columna detectada que mas
influyeron en la prediccion del modelo. Zonas calidas (rojo/amarillo) =
mayor influencia. Fuera de la columna se muestra la radiografia original
en escala de grises — el modelo no evaluo esas zonas.

**Mapa de Confianza** (derecha): certeza por pixel DENTRO de la columna
detectada. Verde = alta confianza, rojo = baja confianza (el medico debe
revisar manualmente esa zona antes de tomar decisiones clinicas).

*Como leerlo:* idealmente, las zonas calientes del Grad-CAM coinciden con
las vertebras y curvaturas visibles, y la confianza es uniformemente
verde sobre toda la columna. Si el Grad-CAM se concentra fuera de las
vertebras, o si hay regiones rojas extensas en la confianza, el resultado
del Cobb debe interpretarse con cautela.

*Este sistema es una herramienta de apoyo. NO reemplaza el criterio del
especialista.*
""",
    "en": """
**Grad-CAM** (left): regions INSIDE the detected spine that most
influenced the model's prediction. Warm tones (red/yellow) = higher
influence. Outside the spine, the original grayscale radiograph is shown
— the model did not evaluate those areas.

**Confidence Map** (right): per-pixel certainty INSIDE the detected
spine. Green = high confidence, red = low confidence (the clinician
should review those areas manually before making clinical decisions).

*How to read it:* ideally, the Grad-CAM hot zones align with visible
vertebrae and curvatures, and the confidence is uniformly green over the
whole spine. If the Grad-CAM concentrates off the vertebrae, or if there
are extensive red regions in the confidence map, the Cobb result should
be interpreted with caution.

*This system is a support tool. It does NOT replace the specialist's
judgment.*
""",
}


MARKDOWN_HEADER = {
    "es": """
# Segmentacion Espinal para Diagnostico de Escoliosis
### Segmentacion vertebral automatica y medicion del angulo de Cobb desde radiografias

Cargue una radiografia espinal para obtener:
- **Segmentacion binaria** de la columna vertebral
- **Segmentacion multiclase** de las vertebras individuales
- **Medicion automatica del angulo de Cobb**
- **Explicabilidad** — mapas Grad-CAM y de confianza (por que decidio el modelo)
""",
    "en": """
# Spine Segmentation for Scoliosis Diagnosis
### Automatic vertebral segmentation and Cobb angle measurement from X-ray radiographs

Upload a spinal X-ray radiograph to get:
- **Binary segmentation** of the spinal column
- **Multiclass segmentation** of individual vertebrae
- **Automated Cobb angle** measurement
- **Explainability** — Grad-CAM and confidence maps (why the model decided)
""",
}


# Compact usage instructions shown between the input image and the
# rotation controls (Ciclo 6.2). Two pedagogical bullets: how to handle
# tilted radiographs + what each output tab contains.
INSTRUCTIONS_MARKDOWN = {
    "es": """
**¿La radiografia vino inclinada o de lado?** Usa el slider de abajo o
los botones rapidos para enderezarla antes de hacer click en *Analyze* —
el metodo del angulo de Cobb asume columna vertical y emitira un
*ROTATION WARNING* si detecta tilt > 12°.

**Que obtendras al analizar:** *Binary* = forma global de la columna ·
*Vertebrae* = vertebras individuales coloreadas · *Cobb Angle* = angulo
+ curvas detectadas con su lateralidad · *Explainability* = mapas
Grad-CAM y de confianza (que regiones miro el modelo).
""",
    "en": """
**Did the X-ray come in tilted or sideways?** Use the slider below or
the quick buttons to straighten it before clicking *Analyze* — the
Cobb-angle method assumes a vertical spine and will emit a *ROTATION
WARNING* if tilt > 12° is detected.

**What you'll get when you analyze:** *Binary* = overall spine shape ·
*Vertebrae* = individual color-coded vertebrae · *Cobb Angle* = angle +
detected curves with their laterality · *Explainability* = Grad-CAM and
confidence maps (which regions the model looked at).
""",
}


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    """Look up a UI string by key in the given language.

    Falls back to Spanish when `lang` is unknown, then to the literal key
    when the key is missing — a visible "key-as-text" placeholder makes
    missing translations obvious during development.
    """
    lang_dict = DIAGNOSIS_STRINGS.get(lang) or DIAGNOSIS_STRINGS[DEFAULT_LANG]
    return lang_dict.get(key, key)


def header_markdown(lang: str = DEFAULT_LANG) -> str:
    """Return the top-of-page markdown intro in the given language."""
    return MARKDOWN_HEADER.get(lang, MARKDOWN_HEADER[DEFAULT_LANG])


def explain_markdown(lang: str = DEFAULT_LANG) -> str:
    """Return the Explainability-tab markdown in the given language."""
    return EXPLAIN_MARKDOWN.get(lang, EXPLAIN_MARKDOWN[DEFAULT_LANG])


def instructions_markdown(lang: str = DEFAULT_LANG) -> str:
    """Return the compact usage-instructions markdown in the given
    language. Shown between the input image and the rotation controls."""
    return INSTRUCTIONS_MARKDOWN.get(lang, INSTRUCTIONS_MARKDOWN[DEFAULT_LANG])


def label_to_lang(label: str) -> str:
    """Map a UI radio label ('Español' / 'English') to the lang code."""
    return LABEL_TO_LANG.get(label, DEFAULT_LANG)


# ---------------------------------------------------------------------------
# Static reference assets — Ciclo 5.9
# ---------------------------------------------------------------------------
# A clinical-collaborator mockup is rendered once per language by
# `scripts/generate_explain_reference.py` and committed to disk under
# `assets/`. The Gradio app shows it ABOVE the dynamic Grad-CAM /
# Confidence panel so a clinician learns how to read the heatmaps before
# seeing their own case.
EXPLAIN_REFERENCE_FILES = {
    "es": "explainability_reference_es.png",
    "en": "explainability_reference_en.png",
}


def explain_reference_path(lang: str = DEFAULT_LANG) -> str:
    """Absolute path to the static Explainability-tab reference PNG for `lang`.

    Falls back to the default language when `lang` is unknown so the UI
    never crashes on a typo — it just shows the Spanish version.
    """
    lang_key = lang if lang in EXPLAIN_REFERENCE_FILES else DEFAULT_LANG
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", EXPLAIN_REFERENCE_FILES[lang_key])
