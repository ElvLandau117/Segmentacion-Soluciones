# SUSTENTACION_GUIA.md — Guía completa para defender el proyecto

> **Fecha de sustentación**: 2026-05-23.
> **Autor de esta guía**: documento operativo para que Elvis Hernández
> abra el browser durante la presentación y siga la narrativa punto por
> punto.
>
> **🚀 URL pública del Space**: https://huggingface.co/spaces/ElvLandau/spine-segmentation
> **📦 Repo GitHub**: https://github.com/ElvLandau117/Segmentacion-Soluciones
> **📄 Paper IEEE**: `docs/Informe_final_escoliosis_IEEE.pdf`
> **🗂 Índice de decisiones**: `docs/DECISIONS.md`

---

## 1. Resumen ejecutivo (para los primeros 60 segundos)

Desarrollamos un **prototipo de inteligencia artificial para asistir el
diagnóstico de escoliosis** desde radiografías AP (anteroposteriores).
El sistema **segmenta automáticamente la columna vertebral y las 22
vértebras individuales** (C3 a L5), **calcula el ángulo de Cobb por dos
métodos complementarios** y **explica visualmente sus predicciones** con
mapas Grad-CAM + mapas de confianza. Está **desplegado públicamente** en
Hugging Face Spaces (CPU gratuito), es **bilingüe español/inglés** y sigue
**convenciones clínicas radiológicas reales** (lateralidad anatómica del
paciente, detección multi-curva, advertencias de coverage y rotación).

**Es una herramienta de APOYO al especialista**, no un reemplazo. El
disclaimer médico es visible en todo momento.

---

## 2. Equipo y división del trabajo

| Persona | Rol en el paper IEEE | Rol en el repo/deploy |
|---|---|---|
| **Diana Lorena Giraldo Arboleda** | Co-autora, segmentación binaria y métricas Cobb central | — |
| **Elvis Raul Hernández Cáceres** | Co-autor, segmentación multiclase + arquitecturas | **Autor principal del repo + deploy + UX clínica + Ciclos 5.1–5.12** |
| **Julian David Florido González** | Co-autor, pipeline alternativo Keras/TensorFlow | Aporta `notebooks/02b_training_alternativo_unet_keras.ipynb` |
| **Juan Pablo Obando Álvarez** | Co-autor, postprocesamiento y Cobb vertebral | — |

**Universidad**: Maestría en IA, Universidad de los Andes, Bogotá, Colombia.

---

## 3. Lo que el jurado puede VERIFICAR sin instalar nada

### Tres entregables visibles desde el browser:

1. **App pública** → https://huggingface.co/spaces/ElvLandau/spine-segmentation
2. **Repo GitHub** → https://github.com/ElvLandau117/Segmentacion-Soluciones
3. **Pesos en HF Hub** → https://huggingface.co/ElvLandau/spine-checkpoints

### Cumplimiento de la rúbrica Coursera/U. Andes (100 pts):

| Criterio | Peso | Estado | Evidencia |
|---|:-:|---|---|
| Funcionamiento de la app | 35% | ✅ | App RUNNING 24/7, smoke tests verdes en 9+ casos del dataset |
| Documentación (README) | 15% | ✅ | `README.md` con descripción + dependencias + 12 env vars + 2 opciones de deploy |
| Parametrización (env vars / config) | 15% | ✅ | `.env.example` + `spine_segmentation/config.py` con 12 variables |
| Usabilidad / UI | 35% | ✅ | 4 tabs claras, toggle ES/EN, reference image educativa, disclaimer visible |

### Carpetas que la rúbrica exige:

| Carpeta | Estado | Comentario |
|---|---|---|
| `notebooks/` | ✅ | 4 notebooks: EDA, training, alternativo Keras, informe final |
| `modelos/` | ✅ | README explicativo (pesos viven en HF Hub por ser >100 MB) |
| `datos/` | ✅ | README explicativo (dataset propiedad U. Andes, no redistribuible) |

---

## 4. Mapeo paper IEEE ↔ app deployada

| Sección del paper IEEE | Qué hay en la app | Dónde verlo en la app |
|---|---|---|
| Sección II "Estado del arte" (U-Net, DeepLabV3+, Grad-CAM, SpineCheck) | Modelos usados en producción + explicabilidad | Backstage; mencionado en README sec 9 |
| Sección III.A "Datos y partición" (250 imgs, 70/15/15, seed=42) | `data_splits.json` reproducible | Repo raíz |
| Sección III.B "Preprocesamiento" (resize 512×512, padding) | Pipeline interno al cargar imágenes | Backstage en `spine_segmentation/data/transforms.py` |
| Sección III.C "Arquitecturas comparadas" (U-Net+ResNet50, U-Net+EfficientNet-B4, DeepLabV3+ResNet50) | + 2 transformers extra del repo (MiT-B3, MAnet+MiT-B5) | `notebooks/02_training_experiments.ipynb` + README sec 9 tabla |
| Sección III.D "Cobb angle" (método central binary + método vertebral v5) | Ambos métodos visibles en el reporte | Tab **Cobb Angle** + Diagnosis Results |
| Sección III.E "Postprocesamiento morfológico" | Cierre vertical, smoothing del spline, multi-pass adaptativo | Backstage `spine_segmentation/postprocessing/morphology.py` |
| Sección III.F "Explicabilidad" (Grad-CAM + confidence maps) | Panel dinámico con masking + percentile clip + colorbars | Tab **Explainability** (sección dinámica abajo) |
| Sección IV "Resultados" (Dice binario 0.88, Cobb MAE 20°) | Métricas reportadas en README sec 9 | README + tabla en este documento |
| Sección V "Discusión y limitaciones" | Warnings activos en producción (coverage, rotación, partial) | Diagnosis Results panel del lado derecho |

---

## 5. Métricas — qué dice cada cosa (CRÍTICO para Q&A)

> Si el jurado pregunta "¿por qué el paper reporta Dice 0.88 y el repo
> 0.34?", esta sección tiene la respuesta clara.

### Las métricas son de TAREAS DISTINTAS, no comparables directamente:

| Métrica | Valor reportado | Tarea | Modelo | Fuente |
|---|:-:|---|---|---|
| **Dice binario** | **0.8840** | Segmentar la columna completa (1 clase) | U-Net ResNet50 | Paper IEEE sec IV |
| **IoU binario** | 0.7960 | Misma | U-Net ResNet50 | Paper IEEE sec IV |
| **Dice multiclass** | **0.3378** | Segmentar las 22 vértebras individuales (24 clases con bg+other) | DeepLabV3+ ResNet50 (ganador entre 5) | Repo AGENTS.md sec "Métricas" |
| **Cobb MAE central** | **20.25°** | Calcular el ángulo de Cobb sobre el binary skeleton | U-Net ResNet50 | Paper IEEE sec IV |
| **Cobb MAE vertebral v5** | 18.06° | Calcular el ángulo de Cobb por endplate de las multiclass | varias | Paper IEEE sec IV |
| **Cobb Pearson r (binary)** | **0.7571** | Misma | U-Net ResNet50 | Paper IEEE sec IV |
| **Cobb Pearson r (vertebral)** | 0.1473 | Misma (limitada por ruido del multiclass) | varias | Paper IEEE sec IV |

### Por qué Dice multiclass es ~3x menor que Dice binario:

- **Binary** = 1 clase, columna entera. Las máscaras de entrenamiento
  cubren mucha más área → más fácil de aprender.
- **Multiclass** = 24 clases (bg + 22 vértebras + other). **Desbalance
  extremo**: C3 ocupa solo 0.0006% de píxeles, C4 0.11% Dice. El modelo
  literalmente no puede aprender clases tan raras con 174 imágenes de
  training.
- **No es un error del modelo, es una limitación del dataset y de la tarea.**

### Por qué el deploy elige DeepLabV3+ multiclass (no U-Net binary):

- El deploy sirve AMBOS: DeepLabV3+ para multiclass (información
  anatómica detallada, nombres de vértebras) + U-Net binary para Cobb
  (más robusto).
- El Cobb se calcula sobre el binary (Pearson 0.76) porque el multiclass
  tiene correlación 0.15 — ruido para la medición angular.
- El multiclass se usa para naming ("C5-T10"), no para el Cobb numérico.

**Pin de respuesta si jurado pregunta**: "El paper reporta dos tareas:
binario (Dice 0.88) y multiclass (no enfatizado). El repo añade un
comparativo de 5 arquitecturas en la tarea multiclass (Dice 0.34 del
ganador). NO son contradictorias — son métricas de tareas DIFERENTES.
La app usa el binary para Cobb y el multiclass para nombrar vértebras."

---

## 6. Cómo funciona la app — script paso a paso para la demo en vivo

> **Antes de la demo (idealmente 1 min antes)**: abrir el Space en una
> pestaña fresca para despertarlo (cold-start tarda 30-60 s la primera
> vez). Tener 3-4 radiografías listas para arrastrar:
> - `MaIA_Scoliosis_Dataset/Scoliosis/S_158.jpg` (escoliosis severa, ~68° en el binary method)
> - `MaIA_Scoliosis_Dataset/Scoliosis/S_100.jpg` (S-shape doble, 84°+65°)
> - `MaIA_Scoliosis_Dataset/Normal/N_1.jpg` (normal sano)
> - `MaIA_Scoliosis_Dataset/Normal/N_61.jpg` (normal pero rotado en captura)

### Acto 1 — Tour rápido de la UI (30 seg)

1. Abrir la URL del Space.
2. Señalar arriba el toggle **Idioma / Language** → "la app es bilingüe
   ES/EN, default español porque la audiencia objetivo es U. Andes /
   Colombia".
3. Señalar el panel **izquierdo**: zona de upload + slider de rotación +
   5 botones rápidos + botón Analyze.
4. Señalar el panel **derecho**: 4 tabs (Binary Segmentation, Vertebrae
   Segmentation, Cobb Angle, Explainability) + Diagnosis Results.
5. Apuntar al **disclaimer médico** al pie: "Aviso médico — esta
   herramienta es de APOYO al diagnóstico, no reemplaza al especialista".

### Acto 2 — Caso de escoliosis severa: S_158 (1.5 min)

1. **Subir** `S_158.jpg` (arrastrar o click upload).
2. **NO rotar** (es captura recta), click **Analyze** (~10 s en CPU).
3. Mientras espera, decir: "el modelo corre 2 redes neuronales en
   paralelo: una binaria que detecta toda la columna y una multiclass
   que detecta las 22 vértebras individuales".
4. Tab **Binary Segmentation**: "la columna queda marcada en verde
   semitransparente, este overlay es la salida del U-Net binary".
5. Tab **Vertebrae Segmentation**: "cada vértebra en un color distinto,
   de cervical (rojo) a lumbar (azul) — esta es la salida del
   DeepLabV3+ multiclass".
6. Tab **Cobb Angle** (el más visual): "cajas verdes en las vértebras
   superior e inferior del Cobb, líneas rojas perpendiculares al
   endplate, arco del ángulo formando la medida, mini Cobb-meter en la
   esquina cuando el ángulo es pequeño".
7. **Leer Diagnosis Results en voz alta**:
   ```
   === ANGULO COBB - Curvas detectadas ===
   Curva principal: 68.3 deg (T2 - T8, convexidad derecha)

   === COBERTURA ===
   Mascara binary cubre: C6 - T11 (13 de ~22 vertebras, ~49%)
   ADVERTENCIA: Cobertura parcial — el angulo Cobb puede ser engañoso.

   === ADVERTENCIA DE ROTACION ===
   La imagen parece estar inclinada 13.4 grados (umbral 12).
   El metodo binary ajusta x = f(y) y puede reportar la rotacion como escoliosis.

   === EVALUACION (basada en Binary principal) ===
   Escoliosis severa (> 40 grados)
   ```
8. **Punto clave clínico** (decir explícitamente): "**convexidad
   DERECHA** — convención clínica radiológica, es la derecha del
   paciente (no del viewer). Fix del Ciclo 5.10 después de feedback de
   una médica colaboradora".
9. **Punto clave de seguridad**: "la app no oculta sus advertencias:
   COVERAGE alerta de segmentación parcial, ROTATION alerta de captura
   inclinada. El médico ve todo el contexto antes de decidir".

### Acto 3 — Tab Explainability (1 min)

1. Click tab **Explainability**.
2. "Arriba hay una **imagen fija de referencia** con 5 callouts
   numerados explicando cómo leer Grad-CAM y Confidence Map. Es
   educativa, siempre visible para que un médico que abre la app por
   primera vez entienda los colores."
3. "Abajo está el **panel dinámico** del caso real: Grad-CAM a la
   izquierda (qué regiones influyeron en la predicción, escala
   azul-amarillo-rojo) + Confidence Map a la derecha (certeza pixel a
   pixel del modelo, escala verde-amarillo-rojo)."
4. "Notar que ambos paneles tienen el **fondo en grises de la
   radiografía** y solo se colorea DENTRO de la columna detectada. Esto
   es el masking del Ciclo 5.8 — antes el cmap pintaba todo y daba la
   impresión falsa de 'baja confianza en todo el fondo'."
5. **Toggle a English** (click en el radio arriba). "El header markdown
   + explain markdown + reference image se traducen al instante. El
   reporte del Diagnosis se traduce en el próximo Analyze."

### Acto 4 — Demo del slider de rotación: N_61 (45 seg — opcional si hay tiempo)

1. Volver a la zona de upload. Subir `N_61.jpg`.
2. "Esta es una radiografía Normal pero capturada ligeramente rotada".
3. **Sin tocar el slider**, click Analyze.
4. Diagnosis Results: "ROTATION WARNING + 1 curva fantasma 17° catalogada
   como Mild scoliosis — falso positivo causado por la rotación".
5. Volver atrás. **Mover el slider a +13°** lentamente — "la imagen rota
   en vivo en menos de 300 ms, gracias al gr.State del Ciclo 5.6".
6. Cuando la columna se ve vertical, click Analyze.
7. Diagnosis Results: "**0.0° Normal**, ROTATION WARNING desapareció".
8. **Punto clave de diseño**: "elegimos slider MANUAL en lugar de
   auto-rotate porque la convención de signo entre nuestra detección
   SVD y `cv2.getRotationMatrix2D` no es trivial — en N_61 confirmamos
   empíricamente que rotar `-tilt_deg` EMPEORA el caso. El control
   manual elimina ese riesgo: el médico ve y decide".

### Acto 5 — Demo multi-curva: S_100 (45 seg — opcional)

1. Subir `S_100.jpg` (S-shape severa).
2. Click Analyze.
3. Diagnosis Results: "**2 curvas detectadas**":
   ```
   Curva principal:  84.2 deg  (T5 - T12, convexidad derecha)
   Curva secundaria: 65.0 deg  (T12 - L4, convexidad izquierda)
   Numero total de curvas: 2 (doble curva, S-shape)
   ```
4. **Punto clave clínico**: "antes del Ciclo 5.2 esto se reportaba como
   UN solo ángulo engañoso (probablemente más bajo por cancelación
   entre las dos curvas opuestas). Ahora replica el estilo del informe
   radiológico clínico real — Julian compartió un ejemplo de informe
   con doble curva que motivó esta mejora".

### Acto 6 — Cierre (15 seg)

1. Volver al tab **Cobb Angle** del último caso (para tener algo visual
   en pantalla mientras se cierra).
2. "El código completo está en
   https://github.com/ElvLandau117/Segmentacion-Soluciones, los pesos
   en https://huggingface.co/ElvLandau/spine-checkpoints, el paper en
   `docs/Informe_final_escoliosis_IEEE.pdf`."
3. "Abrir preguntas."

---

## 7. Narrativa de 10 minutos para la exposición oral

> Estructura recomendada con timing. Cada bullet es 1-2 oraciones, no
> un discurso completo.

### 0:00–1:00 — Motivación clínica
- Escoliosis = deformidad 3D de la columna, evaluada con radiografías AP.
- Estándar: **ángulo de Cobb manual** = subjetivo, lento, depende del
  observador.
- Diferencias de pocos grados → cambios en el manejo clínico.
- **Pregunta**: ¿puede IA automatizar sin sacrificar interpretabilidad?

### 1:00–2:00 — Dataset + partición
- **MaIA Scoliosis Dataset**: 250 radiografías AP (71 Normal + 179
  Scoliosis).
- Máscaras binarias + multiclase (24 vértebras C3–L5).
- Partición 70/15/15 (174/38/38), seed=42, `data_splits.json` reproducible.
- Preprocesamiento: resize 512×512 preservando aspect ratio + padding.

### 2:00–3:30 — Arquitecturas + entrenamiento
- 5 modelos comparados: U-Net+ResNet50, U-Net+EfficientNet-B4,
  **DeepLabV3+ResNet50**, U-Net+MiT-B3, MAnet+MiT-B5.
- Justificación: CNN baseline → CNN eficiente → CNN multi-escala → 2
  transformers (atención global).
- Loss: Weighted Cross-Entropy + Generalized Dice (clases desbalanceadas).
- Hardware: RTX 4060 Ti 16 GB VRAM, AMP fp16, ~30 min por modelo.

### 3:30–5:00 — Resultados + decisiones de arquitectura del deploy
- **Dice multiclass**: DeepLabV3+ ganó con 0.3378 (contraintuitivo —
  superó transformers; el ASPP captura contexto multi-escala mejor para
  vértebras de tamaño variable).
- **Dice binary**: 0.88 — métrica del paper (segmentación columna
  completa).
- **Cobb MAE 20°** método central, **Pearson 0.76**.
- Decisión arquitectónica del deploy: servir **DOS modelos** (DeepLabV3+
  multiclass para naming + U-Net binary para Cobb), no el ensemble de
  los 5 (saving RAM + cold start).

### 5:00–7:00 — Demo en vivo (sigue Acto 1–6 del paso 6 de esta guía)

### 7:00–8:30 — Aspectos clínicos y de UX (Ciclos 5.1–5.12)
- **Multi-curva** (Ciclo 5.2): reportar todas las curvas, no solo una
  (S-shape común en escoliosis adolescente).
- **Coverage warning** (Ciclo 5.3): si el binary cubre <70%, advertir.
- **Rotación** (Ciclos 5.4-5.6): detección SVD + control manual con
  live preview.
- **Lateralidad** (Ciclo 5.10): convención clínica anatómica del paciente.
- **Explicabilidad** (Ciclos 5.8-5.12): Grad-CAM + Confidence
  enmascarados + reference image educativa fija.
- **i18n** (Ciclo 5.7): toggle ES/EN funcional.

### 8:30–10:00 — Limitaciones honestas + trabajo futuro + Q&A intro
- **Dataset pequeño** (250 imgs, single-rater, sin validación externa
  multi-céntrica).
- **Multiclass Dice bajo** (0.34) por desbalance extremo de clases
  cervicales (C3, C4).
- **No comparado vs radiólogos** formalmente.
- **Cobb 2D** ≠ naturaleza 3D real de la escoliosis.
- **Trabajo futuro**: reentrenamiento con augmentation lumbar agresivo,
  Seg-Grad-CAM auténtico, quantización INT8 para tablet, validación
  multi-céntrica.

---

## 8. Q&A anticipadas (preguntas que el jurado puede hacer)

### Q1. "El paper reporta Dice binario 0.88 pero el repo enfatiza Dice multiclass 0.34. ¿Cuál es real?"
**R**: Ambos son reales y miden tareas DIFERENTES. Dice binario 0.88 =
segmentar la columna completa como una sola región (más fácil, paper
sec IV). Dice multiclass 0.34 = segmentar las 22 vértebras individuales
(24 clases con desbalance extremo, repo `notebooks/02_training`).
La app deployada usa el binary para el Cobb (más robusto, Pearson 0.76)
y el multiclass para nombrar vértebras (Tn-Lm en el reporte).

### Q2. "¿Es la app para uso clínico real?"
**R**: NO. Es una **herramienta de APOYO visual al diagnóstico**, no
reemplaza al especialista. El disclaimer médico es visible en todo
momento. No está aprobada por ninguna autoridad regulatoria. Es un
prototipo académico.

### Q3. "¿Validación externa multi-céntrica?"
**R**: No. Limitación honesta documentada en el paper sec V y en el
repo `AGENTS.md` sec 5. Solo evaluación en el test set interno del
MaIA Dataset (38 imágenes). Trabajo futuro.

### Q4. "¿Cómo manejan la rotación de captura?"
**R**: Detección automática vía SVD del skeleton (Ciclo 5.4): si el
ángulo principal del spine se desvía >12° de la vertical, se emite
ROTATION WARNING. El médico decide manualmente cuánto rotar via slider
+ 5 botones rápidos con live preview (Ciclos 5.5–5.6). Rechazamos
auto-rotate por riesgo de ambigüedad de signo (confirmado empíricamente
en N_61: `-tilt_deg` empeora la rotación).

### Q5. "¿Por qué Hugging Face Spaces y no servidor propio?"
**R**: Gratis (CPU Basic 2 vCPU + 16 GB RAM), HTTPS gestionado, sin
mantenimiento de servidor, integrado con HF Hub (donde viven los
pesos). Alternativa Hetzner + Docker + Caddy está documentada en
`docs/DEPLOYMENT.md` y commiteada como "deploy alternativo" — el equipo
puede activarla si en algún momento quiere evitar el sleep de los
Spaces.

### Q6. "¿La convención de lateralidad?"
**R**: Estándar radiológico clínico = anatomía del paciente, no
perspectiva del viewer. En radiografía AP el lado derecho del paciente
aparece a la izquierda de la imagen. Implementado en
`_curve_direction` del Ciclo 5.10 y **refinado en el Ciclo 6.1**
(2026-05-22, post-sustentación) tras feedback de la médica
colaboradora sobre S-shapes. El algoritmo actual usa **chord
signed-area** (signo del área entre la curva y la chord que une los
dos inflection points) — invariante a la asimetría temporal de la
curva y garantiza convexidades opuestas para las dos curvas de un
S-shape. Caso S_158 sigue reportando "convexidad derecha"; S_22
ahora reporta "right" (era "left" pre-6.1, en disagreement con el
ground truth oficial). Detalle en
`docs/CICLO_5_ARTEFACTOS.md` sec 23.

### Q7. "¿Qué pasa si la radiografía es de calidad pobre?"
**R**: Graceful degradation:
- Si el binary cubre <70% → COVERAGE WARNING + Assessment "Inconclusive".
- Si tilt >12° → ROTATION WARNING + sugerencia de usar el slider.
- Si el binary falla por completo → fallback al multiclass + ERROR
  visible.
- El usuario nunca recibe un "Normal" silencioso cuando hay datos
  ausentes.

### Q8. "¿Cuántos casos del dataset han sido evaluados manualmente en el deploy?"
**R**: Smoke testing remoto se ha hecho sobre **10+ casos** del dataset
(N_1, N_61, S_21, S_22, S_45, S_77, S_100, S_120, S_130, S_150, S_158).
Cada Ciclo 5.x documenta los smoke tests en `docs/CICLO_5_ARTEFACTOS.md`.
S_158 es el caso pivote del Ciclo 5.10 (lateralidad).

### Q9. "¿Reproducibilidad? ¿Cómo verifico los resultados?"
**R**: Cuatro niveles:
1. `data_splits.json` reproducible con seed=42.
2. `requirements.txt` con versiones pinned.
3. `notebooks/03_informe_final.ipynb` carga pesos y reproduce métricas
   en CPU.
4. Suite de **66 tests pytest** (50+ unit + 16 contract tests) que
   pinea comportamiento crítico: convención de lateralidad, multi-curva,
   coverage, anchors derivados, etc.

### Q10. "¿Por qué este repo tiene MÁS modelos que el paper?"
**R**: El paper consolida el trabajo grupal del semestre (4 personas).
El repo es la extensión personal de Elvis para producción, con:
- 2 transformers extra (MiT-B3 + MAnet+MiT-B5) no enfatizados en el
  paper.
- 12 ciclos de UX clínica (5.1 a 5.12) post-deploy.
- Suite de tests.
- Sistema de DECISIONS.md por ciclo.
- Bilingüe ES/EN.

---

## 9. Cheat sheet de números clave

> Memorizar estos para responder sin titubear.

| Concepto | Número |
|---|:-:|
| Total radiografías dataset | 250 |
| Normal / Scoliosis | 71 / 179 |
| Split train/val/test | 174 / 38 / 38 |
| Tamaño imagen post-resize | 512×512 |
| Modelos comparados | 5 |
| Dice binario (paper) | 0.8840 |
| Dice multiclass (repo, ganador) | 0.3378 |
| Cobb MAE central | 20.25° |
| Cobb Pearson r central | 0.7571 |
| Vértebras detectadas | C3 a L5 (22) |
| Total clases multiclass | 24 (bg + 22 vert + other) |
| Tamaño pesos en HF Hub | 226 MB (2 .pth) |
| RAM en runtime | ~1.5-2 GB |
| Latencia inferencia (CPU) | 10-13 seg/imagen |
| Cold start del Space | 30-60 seg |
| Tests pytest | 66 passed + 1 skipped |
| Ciclos completados | 1, 2, 3, 4, 5, 5.1–5.12, 6.0 |
| Threshold tilt rotation | 12° |
| Threshold coverage parcial | 70% del alto / <15 vértebras |

---

## 10. Links rápidos para tener abiertos en otra pestaña

1. **Space**: https://huggingface.co/spaces/ElvLandau/spine-segmentation
2. **GitHub**: https://github.com/ElvLandau117/Segmentacion-Soluciones
3. **HF Hub pesos**: https://huggingface.co/ElvLandau/spine-checkpoints
4. **Paper IEEE PDF**: `docs/Informe_final_escoliosis_IEEE.pdf` (local)
5. **DECISIONS.md** (índice navegable): `docs/DECISIONS.md`
6. **Esta guía**: `docs/SUSTENTACION_GUIA.md` (la que estás leyendo)

---

## 11. Plan B — qué hacer si algo falla en vivo

| Falla | Plan B |
|---|---|
| Space no responde / cold start largo | Esperar 30-60 s. Si más de 2 min, refresh. Mientras tanto: mostrar README en GitHub. |
| FileNotFoundError /tmp/gradio (esporádico) | Refresh del browser + re-subir la imagen. Documentado en `docs/DECISIONS.md` "Known issues". |
| Una predicción tarda más de 30 s | Mencionar "CPU Basic free, tier gratuito de HF" y mostrar otro caso pre-cargado. |
| El jurado pide ver el código | Tab GitHub → `spine_segmentation/deployment/app.py` (es el entrypoint visible) o `_curve_direction` en `cobb_angle.py` (fix de lateralidad). |
| Pregunta técnica que no recuerdas | "Esa decisión está en `docs/DECISIONS.md` — déjame verificarla". Abrir el archivo y leer la fila correspondiente. |
| Pregunta clínica fuera de tu expertise | "Para esa pregunta clínica específica, nuestra colaboradora médica nos ha aconsejado [breve], pero la app es de apoyo, no de diagnóstico final." Honestidad + disclaimer. |
| Algo no funciona y NO entiendes por qué | "Eso es un comportamiento que no hemos visto antes — lo registramos como issue y lo investigamos." NUNCA inventar una respuesta técnica.

---

## 12. Recordatorios finales

- **Hablar pausado**. La demo tiene 10 segundos de espera mientras la
  app procesa — usar ese tiempo para narrar lo que está pasando.
- **Mostrar el disclaimer médico** explícitamente. Es una de las cosas
  que más valoran las rúbricas en IA aplicada a salud.
- **No inflar resultados**. Decir Dice 0.34 multiclass con la
  explicación clara (desbalance extremo) es mejor que decir Dice 0.88
  sin aclarar que es la tarea binaria.
- **Mostrar el campo de honestidad del repo**: `docs/DECISIONS.md`
  tiene una sección "Known issues" que demuestra autoconciencia
  técnica.
- **Lateralidad anatómica**: es un detalle pequeño pero MUY valorado
  por radiólogos. Mencionarlo siempre que se vea una convexidad.

---

**¡Éxito en la sustentación!** Esta guía + el browser + el código en
GitHub son todo lo que necesitas. Si algo no está claro acá, abre
`docs/DECISIONS.md` o el ciclo específico en `docs/CICLO_N_ARTEFACTOS.md`.
