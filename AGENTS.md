# AGENTS.md — Segmentacion Vertebral Multiclase para Diagnostico de Escoliosis

> **Spec-Driven Work (Pilar 6):** Artefacto persistente del proyecto.
> Cada ciclo lo actualiza. Todo nuevo chat/agente DEBE leerlo primero.
> Ultima actualizacion: 2026-05-22 | Ciclos: 1, 2, 3, 4, 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 6.0, 6.1 ✅ COMPLETOS. Ciclo 6.2+ : pendiente.
>
> **📋 Indice navegable de decisiones**: [`docs/DECISIONS.md`](docs/DECISIONS.md) (desde Ciclo 5.12).
> **🎤 Guia de sustentacion**: [`docs/SUSTENTACION_GUIA.md`](docs/SUSTENTACION_GUIA.md) (Ciclo 6.0).
>
> **🚀 URL publica de la app:** https://huggingface.co/spaces/ElvLandau/spine-segmentation
>
> **✅ Estado real:** App funcionando end-to-end con UX clinica mejorada (Ciclo 5 + 5.1 + 5.2):
> visualizacion del Cobb con cajas verdes en end vertebrae + perpendiculares rojas al endplate
> + arco del angulo + speedometer fallback + overlay del binary (spline + inflection points).
> **Detección multi-curva (5.2):** binary identifica TODAS las curvas (S-shape, triple-curve)
> via inflection points multiples del spline, reportadas como "Curva principal / secundaria /
> ..." con direccion y nombres de vertebrae (label transfer desde multiclass). UI dual-Cobb
> con indicador de concordancia, Assessment basado en Cobb binary principal. Smoke remoto
> verde: caso S_100 detecta 2 curvas (84° toracica T5-T12 + 65° lumbar T12-L4), antes
> habria salido como un solo angulo enganoso. El multiclass queda SOLO como ilustracion +
> fuente de nombres de vertebrae.

> **Si eres nuevo en el proyecto:** sigue la [`docs/RUTA_LECTURA.md`](docs/RUTA_LECTURA.md)
> antes de hacer cualquier cambio.

---

## 1. Proyecto

**Titulo:** Segmentacion automatica de columna vertebral y vertebras en radiografias de pacientes sanos y con escoliosis para el desarrollo de herramientas de apoyo a medidas radiologicas.

**Institucion:** Universidad de los Andes — Maestria en Inteligencia Artificial

**Objetivo:** Sistema de deep learning que:
1. Segmente vertebras individuales (multiclase, 24 clases) en radiografias AP
2. Calcule el angulo de Cobb automaticamente desde la segmentacion
3. Presente resultados con explicabilidad clinica (Grad-CAM + confianza)
4. Se despliegue en servidor (Hetzner) y sea usable desde tablet

**Dataset:** MaIA Scoliosis — 250 radiografias (71 Normal, 179 Escoliosis), mascaras multiclase PNG con IDs 0-35, remapeadas a 24 clases.

**Hardware:** Windows nativo, RTX 4060 Ti 16GB VRAM, 64GB RAM, 12 nucleos AMD.

---

## 2. Estructura del Repositorio

```
Trabajo final/
├── AGENTS.md                        <- ESTE ARCHIVO (leer primero siempre)
├── .gitignore
├── requirements.txt
├── Dockerfile                       # Deploy en Hetzner/cloud (PyTorch CPU)
├── data_splits.json                 # Split fijo 174/38/38 (seed=42)
│
├── spine_segmentation/              # Paquete Python principal
│   ├── config.py                    # Rutas, hiperparametros, 5 modelos
│   ├── data/
│   │   ├── dataset.py               # SpineMulticlassDataset (24 clases)
│   │   ├── transforms.py            # Albumentations 2.x (Affine, CLAHE, etc)
│   │   ├── splits.py                # Estratificado 70/15/15
│   │   └── class_mapping.py         # 3 esquemas: vertebrae_24, full_36, regional_5
│   ├── models/
│   │   ├── smp_models.py            # Factory SMP: create_model(name, num_classes)
│   │   └── losses.py                # WCE+GDice, Focal, class weights
│   ├── training/
│   │   └── trainer.py               # MLflow + AMP + early stopping + diff LR
│   ├── evaluation/
│   │   ├── metrics.py               # Dice, IoU, per-class, pixel acc
│   │   ├── visualize.py             # Overlays, per-class bar charts
│   │   ├── cobb_angle.py            # Metodo A (binario) + Metodo B (multiclase)
│   │   └── explainability.py        # Grad-CAM + confianza + panel clinico
│   ├── postprocessing/
│   │   ├── morphology.py            # Limpieza mascaras, esqueletizacion
│   │   └── vertebra_ordering.py     # Orden anatomico, endplates, end vertebrae
│   └── deployment/
│       ├── app.py                   # Gradio con tab explicabilidad
│       └── inference.py             # Pipeline end-to-end
│
├── scripts/
│   ├── train_multiclass.py          # python scripts/train_multiclass.py --model <nombre>
│   ├── evaluate.py                  # Evaluacion comparativa de todos los modelos
│   ├── compute_cobb_angles.py       # Batch Cobb + comparacion ground truth
│   └── explore_data.py              # EDA con visualizaciones
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Analisis exploratorio
│   └── 02_training_experiments.ipynb # Notebook maestro (cargar pesos = no GPU)
│
├── checkpoints/                     # Pesos .pth (NO en git, compartir aparte)
├── outputs/                         # Figuras generadas (se regeneran)
└── mlruns/                          # MLflow tracking (se regenera)
```

---

## 3. Los 5 Modelos y POR QUE

| # | Modelo | Paradigma | Justificacion |
|---|--------|-----------|---------------|
| 1 | U-Net + ResNet50 | CNN clasica | Baseline biomedico estandar. Skip connections en 5 niveles. Campo receptivo local (~427px en capa 4). |
| 2 | U-Net + EfficientNet-B4 | CNN eficiente | Compound scaling: mejor precision/computo. Para deploy en tablet/edge (77MB FP32, 19MB INT8). |
| 3 | DeepLabV3+ + ResNet50 | CNN multi-escala | ASPP con convoluciones dilatadas paralelas. Captura vertebras de tamano variable (cervicales pequenas vs lumbares grandes). |
| 4 | U-Net + MiT-B3 | Transformer | SegFormer encoder. Self-attention global: cada pixel ve TODA la imagen. Maneja oclusion vertebral. Efficient attention (lineal, no cuadratica). |
| 5 | MAnet + MiT-B5 | Transformer + atencion dual | Encoder MiT-B5 (attention global) + Decoder MAnet con Position Attention Module. DOBLE ATENCION. Maximo poder para vertebras rotadas/superpuestas. |

**Narrativa:** CNN baseline -> CNN eficiente -> CNN multi-escala -> Transformer -> Transformer+atencion dual. Progresion de capacidad para demostrar que attention mechanisms mejoran la segmentacion vertebral.

---

## 4. Problemas Conocidos del Dominio

### 4.1 Rotacion vertebral en escoliosis severa (CRITICO)
Las vertebras rotan axialmente en escoliosis severa. En radiografia AP (2D) esto causa superposicion parcial, distorsion del cuerpo vertebral, y ambiguedad en bordes.
- **CNNs fallan** porque solo ven ventanas locales — no pueden usar contexto distante
- **Transformers mitigan** con self-attention global: infieren la vertebra oculta desde las vecinas
- Esto es LA RAZON PRINCIPAL para incluir transformers en la comparacion

### 4.2 Caso de paciente con una sola vertebra
Existe al menos un caso atipico. Tratamiento: INCLUIR en el dataset, documentar como caso atipico, reportar metricas con/sin el caso.

### 4.3 Desbalance extremo de clases
- Background: 95.9% de pixeles (peso=0.1)
- C3: 0.0006% de pixeles (peso=10.0, maximo)
- Cervicales (C3-C5) son extremadamente raras y pequenas
- Mitigacion: Weighted Cross-Entropy + Generalized Dice Loss

### 4.4 Imagenes de tamano variable
Radiografias ~259x971 a ~381x1074. Solucion: resize preservando aspect ratio + padding a 512x512. Mascaras con interpolacion nearest-neighbor.

### 4.5 Angulos de Cobb limite
Algunos ground truth = ~90 grados exactos (limite matematico del arctan). Documentar como limitacion.

---

## 5. Estado Actual

### Ciclo 1 (completado) — Infraestructura
- [x] Paquete Python modular (`spine_segmentation/`)
- [x] Pipeline de datos: augmentation, splits, 3 esquemas de clases
- [x] 5 modelos configurados en config.py
- [x] Trainer con MLflow + AMP + early stopping
- [x] Evaluacion: Dice, IoU, per-class, visualizaciones
- [x] Cobb angle: 2 metodos (binario + multiclase)
- [x] Explicabilidad: Grad-CAM + confianza + panel clinico
- [x] Deploy local: Gradio app + Dockerfile
- [x] Notebook maestro para equipo (carga pesos, corre en CPU)

### Ciclo 2 (completado) — Entrenamiento Transformers + Git
- [x] Git inicializado con .gitignore correcto
- [x] AGENTS.md creado (memoria persistente del proyecto)
- [x] U-Net + MiT-B3 multiclase entrenado (Dice=0.3157)
- [x] MAnet + MiT-B5 multiclase entrenado (Dice=0.3271)
- [x] Hallazgo: transformers no superan CNN multi-escala

### Ciclo 3 (completado) — 5 Modelos + Paquete Equipo
- [x] DeepLabV3+ multiclase entrenado (Dice=0.3378 — MEJOR)
- [x] U-Net + ResNet50 multiclase entrenado (Dice=0.2691)
- [x] U-Net + EfficientNet-B4 multiclase entrenado (Dice=0.2189)
- [x] Evaluacion comparativa de los 5 modelos
- [x] Pesos exportados inference-only (838 MB, 66% reduccion)
- [x] Paquete OneDrive listo (`paquete_equipo_onedrive.zip`, 776 MB)
- [x] Notebook `03_informe_final.ipynb` estilo semestre anterior
- [x] Artefacto formal: [`docs/CICLO_3_ARTEFACTOS.md`](docs/CICLO_3_ARTEFACTOS.md)

### Ciclo 4 ✅ COMPLETO — Despliegue en HF Spaces
- [x] Reorganizar carpetas (requisitos_universidad/, docs/metodologia/, archive/)
- [x] Migrar pesos de OneDrive a Hugging Face Hub (script + guia)
- [x] Parametrizar config.py con 12 env vars (.env.example documentado)
- [x] app/ shim entrypoint (cumple convencion de la rubrica)
- [x] Modulo weights.py con autodescarga desde HF Hub + cache
- [x] Dockerfile v2 (multi-stage, usuario no-root, sin COPY checkpoints, healthcheck con curl)
- [x] docker-compose.yml + Caddyfile (SSL automatico via Let's Encrypt + nip.io) — deploy alternativo, no usado
- [x] Suite pytest (14 tests: 13 passing + 1 gated por requires_checkpoints)
- [x] README v2 alineado a la rubrica (35+35+15+15)
- [x] Runbook DEPLOYMENT.md + script reproducible deploy_hetzner.sh
- [x] Pivote a HF Spaces como hosting principal (no Hetzner)
- [x] Subida de pesos a `ElvLandau/spine-checkpoints` (226 MB en 2 .pth)
- [x] Space `ElvLandau/spine-segmentation` desplegado y RUNNING
- [x] **Fix bug `gradio-client TypeError api_info`** via upgrade a Gradio 5.50.0 + huggingface_hub>=0.33.5
- [x] **Script reusable `scripts/upload_to_space.py`** (HfApi.upload_file, sin git push)
- [x] **Smoke test end-to-end verde** (gradio-client predict devuelve 4 overlays + texto en ~13s)

Artefacto: [`docs/CICLO_4_ARTEFACTOS.md`](docs/CICLO_4_ARTEFACTOS.md)
Briefing original: [`docs/CICLO_4_DESPLIEGUE_BRIEF.md`](docs/CICLO_4_DESPLIEGUE_BRIEF.md)
Prompt para retomar: [`docs/PROMPT_PROXIMO_CHAT.md`](docs/PROMPT_PROXIMO_CHAT.md)

### Ciclo 5 ✅ COMPLETO — UX clinica del Cobb (sin reentrenamiento)
Inspirado parcialmente en Shi et al. 2025 (`archive/2509.24898v1.pdf`).
Sin replicar su modelo (no tenemos landmarks anotados), solo extrapolando
ideas de visualizacion + fusion de mediciones.

- [x] Assessment basado en Cobb binary (mas robusto; MAE 23° vs multi 26-45°)
- [x] **Visualizacion Cobb tipo Fig 1 del paper (Ciclo 5.1 polish)**:
      perpendiculares correctas al endplate (no a lo largo) + punto de
      interseccion + arco del angulo + marcadores cyan sobre el endplate +
      mini "Cobb-meter" en esquina para angulos pequeños + overlay del binary
      (spline blanco + inflection points amarillos). Modularizada en helpers
      en `evaluation/visualize.py`.
- [x] UI dual-Cobb: ambos metodos en paralelo + indicador de CONCORDANCIA
      (<=5° alta / <=15° revisar / >15° discrepancia)
- [x] Refactor: `build_results_text` extraido como helper testeable (puro)
- [x] **Deteccion multi-curva (Ciclo 5.2)**: `cobb_from_binary` ahora devuelve
      `curves: list[dict]` con TODAS las curvas detectables (pairs adyacentes
      de inflection points), no solo los 2 extremos. Cada curva con: angulo,
      direccion (right/left), ip_upper/ip_lower, rank. `assign_vertebra_names
      _to_curves` hace label-transfer desde multiclass para nombrar end
      vertebrae (T5-T12, etc.). UI texto reorganizada en bloques "Curva
      principal / secundaria / Curva N" + convexidad. Viz dibuja las 2
      mayores con colores distintos (red principal, magenta secundaria).
- [x] 12 tests nuevos en total (suite: 25 passed + 1 skipped)
- [x] Deploy via `scripts/upload_to_space.py` (4 archivos en Ciclo 5.2)
- [x] Smoke test remoto verde — S_100 detecta 2 curvas (84° toracica + 65°
      lumbar); el caso S_21 que antes daba "Normal"
      (multiclass=0.4°) ahora reporta "Mild scoliosis" correctamente
      (binary=15.1°, concordancia: "Review recommended")

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md)

### Ciclo 5.3 ✅ COMPLETO — Cobertura del binary + UX informativa (sin reentrenar)
Motivado por una prueba manual en `S_22` (S-shape clara): el binary solo cubrio
C6-T10 (~12 de 22 vertebrae), el spline se ajusto a la mitad superior casi recta
y la app reporto "0° — no clinically meaningful curves" cuando el ojo clinico ve
dos curvas. Cuello de botella = cobertura, no severidad ni algoritmo.

- [x] **Fix A** ([inference.py:117](spine_segmentation/deployment/inference.py)):
      umbral del binary 0.5 → 0.3 (acepta pixeles marginales en zona lumbar).
- [x] **Fix B** ([morphology.py:clean_binary_mask](spine_segmentation/postprocessing/morphology.py)):
      cierre morfologico vertical con kernel rect (3, 25) **antes** del "keep
      largest connected component". Puentea fragmentos lumbar↔toracico que
      antes se perdian al filtrar por componente mas grande.
- [x] **Fix C+D** ([cobb_angle.py](spine_segmentation/evaluation/cobb_angle.py)):
      defaults `smoothing_factor` 5000 → 1500 + `min_curve_deg` 3.0 → 2.0.
      Spline mas sensible captura inflexiones sutiles.
- [x] **Multi-pass adaptativo** (cobb_angle.py): si la primera pasada encuentra
      solo 1 curva, re-corre con `smoothing_factor * 3.3` y prefiere lo que de
      mas curvas. Reconcilia S_22 (necesita smoothing bajo) con S_100 (necesita
      smoothing alto para preservar S-shape).
- [x] **Fix F** (`evaluation/coverage.py` nuevo + integracion en inference.py +
      `build_results_text` en app.py): nuevo helper `compute_coverage_info`
      retorna y-range del binary + ratio + nombres de vertebrae cubiertas
      (mapeo nearest-y vs multiclass) + flag `is_partial`. UI emite bloque
      "=== COVERAGE === / Binary mask covers: C6-T10 (12 of ~22, ~88%) /
      WARNING: ... NOT segmented ..." cuando `is_partial=True`. Assessment
      cambia a "Inconclusive — insufficient coverage" cuando partial + 0°.
- [x] 7 tests nuevos (suite: 32 passed + 1 skipped, era 25+1).
- [x] Deploy via `scripts/upload_to_space.py` (5 archivos, commit atomico).
- [x] Smoke remoto verde sobre 9 casos: N_1, S_22, S_21, S_100, S_45, S_77,
      S_120, S_130, S_150. **S_22 ahora detecta 2 curvas (19.5° toracica +
      5.9° lumbar) + WARNING de coverage en vez de "0° Normal" engañoso.**
      S_100 mantiene 2 curvas (83°+61°) — sin regresion.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 13.

### Ciclo 5.4 ✅ COMPLETO — Robustez ante rotacion + UX de la viz Cobb
Smoke manual del Space (post 5.3) revelo 3 issues nuevos: (a) imagenes
rotadas como `N_61.jpg` generaban hasta 4 curvas fantasma porque el spline
`x = f(y)` interpreta la rotacion como inflexiones; (b) los rotulos
`[Principal/Secundaria] Superior/Inferior (Vn)` quedaban encimados cuando 2
curvas compartian vertebra; (c) curvas degeneradas con
`upper_vertebra == lower_vertebra` (e.g. "5.9° T9-T9" en S_22) pasaban el
filtro `min_curve_deg` y confundian al usuario.

- [x] **Fix G — Deteccion de rotacion via SVD** ([orientation.py nuevo](spine_segmentation/evaluation/orientation.py)):
      `compute_orientation_info(skeleton_points)` aplica SVD sobre el
      point-cloud del skeleton centrado. El primer singular vector es el
      eje principal; su angulo vs vertical = tilt_deg. `is_tilted` se
      activa cuando `abs(tilt) > 12°`. Sin nuevas dependencias
      (`numpy.linalg.svd`). Integrado en [inference.py](spine_segmentation/deployment/inference.py)
      via `extract_spine_skeleton` + `get_skeleton_points` sobre el binary
      mask ya limpio.
- [x] **Bloque `=== ROTATION WARNING ===`** en `build_results_text`
      ([app.py](spine_segmentation/deployment/app.py)). Decision UX: el
      Cobb binary se sigue mostrando (no se hide); el medico ve el numero
      y el aviso lado a lado y decide.
- [x] **Fix H — Filtrado de curvas degeneradas** ([cobb_angle.py](spine_segmentation/evaluation/cobb_angle.py)):
      (i) Nueva constante `MIN_IP_Y_DISTANCE_PX = 30` filtra pares de
      inflection points cuya y-distance en el grid del spline (500 pts) es
      sub-vertebral. (ii) `assign_vertebra_names_to_curves` elimina
      curvas con `upper == lower` despues del label transfer (no marca,
      elimina), y reindexa `rank` para preservar el orden principal/
      secundaria. (iii) [inference.py](spine_segmentation/deployment/inference.py)
      resyncroniza `cobb_angle_deg` / `inflection_points` con la nueva
      principal si el filtro corrio.
- [x] **Fix I — Dedup + anti-overlap de rotulos** ([visualize.py](spine_segmentation/evaluation/visualize.py)):
      Nuevo `_rects_overlap(a, b)` helper. `_draw_single_cobb_curve`
      acepta acumuladores `labeled_vertebrae: set` y
      `placed_label_rects: list`. Si una vertebra ya tiene rotulo de una
      curva previa (e.g. T9 = lower principal + upper secundaria), se
      omite el texto duplicado (la geometria — boxes, perpendiculares,
      arco — sigue dibujada). Si el rotulo colisionaria con uno previo,
      se desplaza verticalmente en pasos de 16 px hasta encajar.
      `draw_cobb_angle_visualization` instancia los acumuladores una vez
      y los threadea entre los `_draw_single_cobb_curve` calls.
- [x] 9 tests nuevos (suite: **41 passed + 1 skipped**, era 32+1).
- [x] Deploy via `scripts/upload_to_space.py` (5 archivos, commit atomico).
- [x] Smoke remoto verde sobre 10 casos (los 9 del Ciclo 5.3 + N_61):
      **N_61** baja de 4 curvas fantasma a 1 + ROTATION WARNING visible.
      **S_22** pierde la degenerada T9-T9 y queda con 1 sola curva
      principal limpia. S_100 mantiene 2 curvas con warning borderline
      (tilt 12.6°, justo sobre el umbral); diagnostico clinico no
      cambia (Severe scoliosis).

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 14.

### Ciclo 5.5 ✅ COMPLETO — Control manual de rotacion en la UI
Smoke remoto del 5.4 mostro que el ROTATION WARNING era informativo pero
el pipeline seguia corriendo sobre la imagen rotada: N_61 reportaba 1 curva
fantasma 17.1° "Mild scoliosis" + warning. Auto-rotacion tenia riesgos
(S_100 borderline + ambiguedad de signo de tilt_deg + falta de control
clinico). Solucion mejor (idea de Elvis): exponer un control manual de
rotacion en la UI Gradio.

- [x] **Helper module-level `rotate_image_for_analysis(image, deg)`**
      ([app.py](spine_segmentation/deployment/app.py)) — `cv2.warpAffine`
      con `BORDER_REPLICATE` y deadband de 0.5° para que slider en 0 no
      pague costo de warp.
- [x] **`predict()` closure acepta `rotation_deg`** y aplica el helper
      ANTES de delegar a `pipeline.predict()`. El pipeline interno ve la
      imagen ya enderezada por el medico. Cero cambios en
      `inference.py`, `cobb_angle.py`, `orientation.py`, `coverage.py`,
      `visualize.py`, `morphology.py`.
- [x] **UI nueva** debajo del componente de imagen:
      `gr.Slider(-180, 180, value=0, step=1, "Rotate image (deg). Negative = clockwise.")`
      + 5 botones rapidos: `↺ -90°`, `↺ -5°`, `Reset`, `↻ +5°`, `↻ +90°`.
      Cada boton hace clip a (-180, 180). `predict_btn.click` ahora pasa
      `[input_image, rotation_slider]` como inputs.
- [x] **El ROTATION WARNING del Ciclo 5.4 se queda**: si despues de la
      rotacion manual el SVD aun ve tilt > 12°, advierte ("you rotated,
      but it's still tilted"). Si despues de rotar queda < 12°, no
      warning. Si no se rotó y la imagen estaba vertical, no warning.
- [x] 3 tests nuevos (suite: **44 passed + 1 skipped**, era 41+1).
- [x] Deploy via `scripts/upload_to_space.py` (commit atomico con shim
      raiz + modulo en su ubicacion correcta — error de path-in-repo en
      el primer deploy se corrigio con un segundo commit que restauro
      el shim raiz).
- [x] Smoke remoto verde: **N_61 (rotacion +13°) reporta 0° Normal**,
      eliminando el falso positivo del 5.4. N_61 sin rotacion sigue con
      warning del 5.4. N_1, S_22 (rotacion=0) byte-identicos al 5.4.

**Hallazgo importante**: al probar N_61 con rotacion `-13°` (la
direccion que sugeria el plan inicial de auto-rotacion basado en
`-tilt_deg`), el tilt EMPEORO de 13.1° a 25.1°. La direccion correcta
era `+13°`. Esto confirma post-hoc que el approach auto-rotation era
peligroso: la convencion de signo entre `compute_orientation_info` y
`cv2.getRotationMatrix2D` no es trivial, y haber adivinado mal habria
sumado curvas fantasma en vez de quitarlas. El control manual le da
al medico el feedback visual para acertar la direccion en una sola
intervencion.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 15.

### Ciclo 5.6 ✅ COMPLETO — Live preview de la rotacion en la UI
El slider del 5.5 actualizaba un valor pero la imagen NO rotaba hasta
presionar Analyze (10s CPU). Elvis lo planteo desde la perspectiva del
usuario clinico: "que me muestre como queda al rotar para que pueda
decidir como dejarla y ahi si hacer el analisis". Sin feedback visual el
medico iteraba a ciegas, gastando 10s por intento.

- [x] **Helper module-level `preview_rotation_for_display(original, deg)`**
      ([app.py](spine_segmentation/deployment/app.py)) — delega a
      `rotate_image_for_analysis`, retorna None cuando no hay original.
- [x] **`gr.State(value=None)` para imagen original**. Guardado en
      `input_image.upload` para que el slider rote SIEMPRE la original,
      no la mostrada. Evita acumulacion de rotaciones cuando el usuario
      arrastra el slider por valores intermedios.
- [x] **`rotation_slider.change`**: rota original por el slider y
      escribe en `input_image` (la mostrada). `input_image.upload` no
      dispara cuando el preview escribe — sin event loop.
- [x] **5 botones rapidos** ahora retornan `(new_slider, rotated_image)`
      en una sola llamada (atomic widget+display update). Mas confiable
      que esperar a que `slider.change` dispare tras update programatico.
- [x] **`predict()` pierde `rotation_deg`**. Por que cuando Analyze
      dispara, `input_image` ya esta rotada por el preview pipeline.
      Predict solo delega al pipeline. Contrato mas limpio, cero riesgo
      de doble-rotacion.
- [x] **Reset button** ahora retorna `(0.0, original_unrotated)` — slider
      vuelve a 0 Y display vuelve a la original sin rotar.
- [x] 3 tests nuevos (suite: **47 passed + 1 skipped**, era 44+1).
- [x] Deploy via `scripts/upload_to_space.py` con `--path-in-repo` explicito
      (leccion del 5.5: sin esto el modulo va a basename, sobreescribiendo
      el shim raiz).
- [x] Smoke remoto verde: N_61, N_1, S_22 con slider=0 byte-identicos al
      5.5. Live preview validable solo en browser (UI-only).

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 16.

### Ciclo 5.7 ✅ COMPLETO — Limpieza multiclass del frontend + toggle ES/EN
Feedback de Elvis tras probar el live preview del 5.6: la info de
multiclass (bloque `=== CROSS-CHECK ===` con "Multiclass: 90.0 deg
(Upper=..., Lower=...; illustration only)" + "CONCORDANCIA: ...") en el
panel Diagnosis Results confundia al usuario — sin contexto algoritmico,
los numeros parecian contradictorios (binary 4 deg vs multi 90 deg). Y la
UI era inconsistente: el reporte en español pero el header markdown y
tabs en ingles.

- [x] **Fix M — Multiclass cleanup del frontend.** Eliminado el bloque
      `=== CROSS-CHECK binary vs multiclass ===` entero de
      `build_results_text` ([app.py](spine_segmentation/deployment/app.py)).
      Eliminada tambien la 3ra linea cyan "Multiclass (illustration only):
      X.X deg" del header en `draw_cobb_angle_visualization`
      ([visualize.py](spine_segmentation/evaluation/visualize.py)). El
      multiclass sigue usandose internamente para label transfer (nombres
      Tn-Lm en las cajas verdes) y para el tab "Vertebrae Segmentation" —
      no se ve mas su numero crudo en la UI principal. El multi-fallback
      cuando binary FALLA totalmente se conserva (alli si es el unico
      signal valido).
- [x] **Fix N — i18n module + toggle ES/EN** (Nivel B confirmado por
      Elvis: Diagnosis Results + Header markdown, default Español).
      Nuevo modulo [`i18n.py`](spine_segmentation/deployment/i18n.py)
      con `DIAGNOSIS_STRINGS` (ES + EN dicts) + `MARKDOWN_HEADER` +
      helpers `t(key, lang)` / `header_markdown(lang)` /
      `label_to_lang(label)`. `build_results_text(..., language='es')`
      lee strings via `t()`. `predict()` acepta `language_label`. UI
      gana `gr.Radio(['Español', 'English'], value='Español')` arriba de
      todo. `language_radio.change` actualiza el markdown header en
      vivo; el diagnosis text se re-traduce en el siguiente Analyze (10s
      de inferencia justo para retraducir strings fijos seria
      wasteful).
- [x] Texto del ROTATION WARNING actualizado: ahora apunta al "rotation
      slider to straighten the image" (la solucion real del 5.5/5.6) en
      vez del "trust the multiclass measurement" obsoleto.
- [x] 8 tests nuevos (suite: **55 passed + 1 skipped**, era 47+1). Tests
      antiguos que asercionaban strings en ingles ahora pasan
      `language='en'` explicito.
- [x] Deploy via `scripts/upload_to_space.py` (3 archivos, commit
      atomico, con `--path-in-repo` explicito — leccion del 5.5).
- [x] Smoke remoto verde: S_22 en Español ("Curva principal", "COBERTURA",
      "Escoliosis leve", "VERTEBRAS DETECTADAS") y en English ("Principal
      curve", "COVERAGE", "Mild scoliosis", "VERTEBRAE DETECTED").
      Confirmado en ambos idiomas que el bloque CROSS-CHECK NO aparece y
      el viz NO tiene la linea Multiclass cyan.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 17.

### Ciclo 5.8 ✅ COMPLETO — Polish del tab Explainability (Grad-CAM + Confidence)
Feedback de Elvis tras probar el 5.7: "las zonas que marca [el Grad-CAM]
no son... o no son claras". El Grad-CAM pintaba activaciones en TODA la
imagen (incluso fuera de la columna), confundiendo. El confidence map
saturaba el fondo en rojo (cmap RdYlGn pinta el 0 como rojo intenso) y
no tenia titulo ni escala visible. Tampoco habia explicacion clinica
clara de como interpretar cada panel.

- [x] **Fix O — Enmascarar Grad-CAM y Confidence por la mascara binary
      predicha**. `generate_gradcam` y `generate_confidence_map`
      ([explainability.py](spine_segmentation/evaluation/explainability.py))
      ahora aceptan `prediction_mask` opcional; cuando se pasa,
      multiplican el output por la mascara → activaciones/incertidumbre
      solo DENTRO de la columna detectada. En `app.py`, el render mezcla
      las zonas-de-fuera con la imagen original (cv2.where) para que el
      usuario conserve el contexto anatomico sin el ruido de color.
- [x] **Fix R — Percentile clip (p95) en Grad-CAM**. Tras el masking,
      el heatmap se renormaliza al p95 para que los hot-spots reales
      destaquen. Sin esto, un par de outliers comprimian la escala y
      el panel se veia "lavado".
- [x] **Fix P — Anotaciones in-image en el panel side-by-side**. Nueva
      helper `annotate_explainability_panel(cam, conf, language_label)`
      en [app.py](spine_segmentation/deployment/app.py): añade strip
      oscuro de 32px arriba de cada subpanel con titulo
      ("Grad-CAM (atencion del modelo)" / "(model attention)" y
      "Confianza (certeza del modelo)" / "(model certainty)") +
      mini-colorbar vertical de 18px de ancho a la derecha de cada
      subpanel con etiquetas "Alta/Baja" o "High/Low". Bilingue via
      i18n. Imagen final: 1100x544 (vs 1024x512 antes).
- [x] **Fix Q — Markdown clinico bilingue del tab Explainability**.
      Nuevo `EXPLAIN_MARKDOWN` dict + `explain_markdown(lang)` helper
      en [i18n.py](spine_segmentation/deployment/i18n.py). Wording
      clinico mejorado con seccion "Como leerlo" (que es un resultado
      bueno vs malo). El handler `language_radio.change` ahora
      actualiza el header markdown Y el explain markdown en una sola
      llamada.
- [x] 5 tests nuevos (suite: **60 passed + 1 skipped**, era 55+1).
- [x] Deploy via `scripts/upload_to_space.py` con `--path-in-repo`
      explicito.
- [x] Smoke remoto verde: panel explainability es 1100x544 con titulos
      visibles, colorbars renderizadas, funciona en ES y EN.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 18.

### Ciclo 5.9 ✅ COMPLETO — Imagen fija de referencia clinica en Explainability
Mockup educativo aportado por el medico companero (5 callouts numerados +
colorbars + disclaimer) para dejar fijo arriba del panel dinamico de
Explainability. Da al medico una leyenda visual de como interpretar
Grad-CAM y Confidence Map ANTES de ver su propio caso, reduciendo la
curva de lectura del 5.8.

- [x] **Generador one-shot** [`scripts/generate_explain_reference.py`](scripts/generate_explain_reference.py):
      compone el panel side-by-side desde un radiograph sample (S_22) +
      mascara binary del dataset, simula Grad-CAM (jet) y Confidence
      (RdYlGn) sobre la silueta de la columna, dibuja 5 callouts
      numerados con flechas leader + dos colorbars centradas + caption +
      disclaimer footer. Toma `--lang es|en|both`. No corre en runtime —
      solo para regenerar los assets.
- [x] **Assets bilingues** committeados:
      `spine_segmentation/deployment/assets/explainability_reference_es.png`
      y `_en.png` (~165 KB cada uno, 1126×716).
- [x] **`explain_reference_path(lang)`** nuevo en
      [`i18n.py`](spine_segmentation/deployment/i18n.py): retorna el path
      absoluto al PNG correcto. Fallback a Español si el lang es
      desconocido.
- [x] **UI** [`app.py`](spine_segmentation/deployment/app.py): nuevo
      `gr.Image(interactive=False, height=300)` ARRIBA del
      `explain_output` dinamico del 5.8. `language_radio.change` actualiza
      en vivo header_md + explain_md + reference_image en una sola
      llamada (extension natural del handler del 5.7/5.8).
- [x] 2 tests nuevos (suite: **62 passed + 1 skipped**, era 60+1).
- [x] Deploy via `scripts/upload_to_space.py` con `--path-in-repo`
      explicito para los 4 archivos (2 PNGs + i18n.py + app.py).
- [x] Smoke remoto verde: Space en RUNNING con la imagen ES visible al
      cargar y la EN visible al togglear el radio (verificado via
      gradio_client + HEAD HTTP 200).

**Decision honesta**: como la imagen PNG original del medico companero
no estaba disponible como path de archivo (solo como adjunto visual al
chat), los dos PNGs son **recreacion programatica del mockup**, no
copia binaria. Garantiza consistencia ES↔EN al 100%, pero puede no
coincidir pixel a pixel con el diseño original. Si Elvis quiere usar el
PNG exacto del medico, basta dejar el archivo en `assets/...es.png` y
re-correr `python scripts/generate_explain_reference.py --lang en` para
regenerar solo el EN con la misma plantilla.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 19.

### Ciclo 5.10 ✅ COMPLETO — Fix de lateralidad clinica + sample mas severo
Feedback de la compañera medica (especialista) tras probar el Space del
Ciclo 5.9: dos puntos. (1) la app reporta la convexidad en perspectiva
del viewer en vez de en anatomia del paciente — caso S_158 de la MaIA
(anatomicamente right-convex) reportado como "convexidad izquierda".
(2) la radiografia base del reference image (S_22, Cobb 24.9°) no se
reconoce como un caso real del dataset porque es demasiado modesta;
elegir un caso mas demostrativo (S_200) sube el valor pedagogico.

- [x] **Fix W — Convencion clinica de lateralidad**
      ([cobb_angle.py:_curve_direction](spine_segmentation/evaluation/cobb_angle.py)).
      Swap del ternario en linea 73 (right ↔ left) + docstring
      reescrita documentando explicitamente la convencion radiologica
      del espejo (paciente derecho = viewer izquierdo). El cambio es
      contained: i18n / app / visualize no necesitan tocarse, los
      strings "derecha" / "izquierda" siguen funcionando — solo cambia
      su SIGNIFICADO (ahora anatomia del paciente, antes perspectiva
      del viewer).
- [x] **Test de regresion** `test_curve_direction_uses_patient_anatomy
      _convention` que pinea el mapping post-fix (negative slope → "left",
      positive slope → "right") + documenta el flip vs el comportamiento
      pre-Ciclo 5.10. Asegura que un refactor futuro no re-invierta la
      convencion silenciosamente.
- [x] **Sample base S_22 → S_200**
      ([scripts/generate_explain_reference.py](scripts/generate_explain_reference.py)).
      S_200 muestra una S-curve clinicamente clara; el rendering
      educativo gana fidelidad pedagogica. Defaults del script bumped a
      S_200; las 2 PNGs en `assets/` regeneradas (~187 KB c/u).
- [x] 1 test nuevo (suite: **63 passed + 1 skipped**, era 62+1).
- [x] Deploy via `scripts/upload_to_space.py` con `--path-in-repo`
      explicito (3 archivos: cobb_angle.py + 2 PNGs).
- [x] Smoke remoto verde: predict(S_158) ahora reporta "convexidad
      derecha" para la curva principal (era "izquierda"). El reference
      image en el tab Explainability ya muestra la base S_200.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 20.

### Ciclo 5.11 ✅ COMPLETO — Fix de arrows del reference image (sample-invariant)
Feedback de Elvis al ver el deployed del Ciclo 5.10: las 5 flechas del
reference image apuntan a espacios vacios — "ese ejemplo no señala nada,
apunta a cosas que no tienen sentido". Causa raiz: los 4 hotspots
simulados (`gaussian_blob` en `_simulate_gradcam`) y los 5 `arrow_xy` de
los callouts estaban hardcoded para el layout de S_22 (spine centrado,
parcial). S_200 tiene el spine ligeramente a la derecha y full-height,
asi que los blobs colisionaban con el spine real y las flechas aterrizaban
fuera del spine y de los blobs.

- [x] **Refactor del generador** ([scripts/generate_explain_reference.py](scripts/generate_explain_reference.py)):
      2 helpers nuevos a nivel modulo:
      `_derive_visual_anchors(spine_mask) -> dict` retorna spine bbox +
      centroide + 4 blob positions (top, pelvis, left, right) DERIVADAS
      del bbox del spine, no hardcoded. `_pixel_to_figure_coords(px, py,
      ax_rect)` convierte pixel del imshow 512x512 a coords figure
      fraction. Constantes `AX_CAM_RECT` y `AX_CONF_RECT` mantienen el
      converter y `_build_figure` en sync.
- [x] **`_simulate_gradcam` acepta `anchors`** y coloca los 4
      gaussian_blob calls en posiciones derivadas — siempre fuera del
      spine real.
- [x] **`_build_figure` acepta `anchors`** y deriva los 5 arrow_xy via
      `_pixel_to_figure_coords` apuntando al centroide del spine
      (callouts 2, 4), al blob_top (callout 1), al blob_pelvis (callout
      3), y al borde lateral inferior del bbox (callout 5).
- [x] **`generate()` thread-ea anchors** desde
      `_derive_visual_anchors(mask)` hacia las 2 funciones consumidoras.
      Cero cambios cosmeticos en las posiciones de los rectangulos de
      callouts (siguen fijos en los margenes laterales — eso si es
      estetico, no sample-dependent).
- [x] 2 tests nuevos (suite: **65 passed + 1 skipped**, era 63+1):
      `test_derive_visual_anchors_places_blobs_outside_spine` valida
      bbox + blobs fuera del spine + fallback de empty mask;
      `test_pixel_to_figure_coords_handles_corners` valida los 4
      corners + midpoint del converter (figure y flipped).
- [x] Deploy via `scripts/upload_to_space.py` con `--path-in-repo`
      explicito (solo los 2 PNGs; el generator vive en el repo, no en
      el Space).
- [x] Smoke remoto: PNGs regenerados (~189 KB ES, ~185 KB EN) servidos
      via `/_on_language_change`. Validacion visual final: Elvis abre
      Explainability tab y ve que las 5 flechas aterrizan en
      blobs/spine, ya no en vacio.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 21.

### Ciclo 5.12 ✅ COMPLETO — Fix coord centering (aspect='equal') + DECISIONS.md
Tras desplegar el Ciclo 5.11, Elvis verifico visualmente el reference
image en el Space y reporto: las flechas SEGUIAN apuntando mal —
aterrizaban ~50-60 px ARRIBA de los blobs. Diagnostico riguroso: bug
en mi propio fix del Ciclo 5.11. `_pixel_to_figure_coords` asumia que
el imshow llenaba toda la ax_rect, pero matplotlib renderiza una
imagen 512x512 con `aspect='equal'` como un cuadrado centrado en el
ax_rect — para AX_CAM_RECT (2.928" wide x 4.153" tall), la imagen se
renderiza como 2.928 x 2.928 centrada verticalmente, con 0.61" (61 px)
de margen arriba y abajo. Para pixel (262, 20) la formula vieja daba
fy=0.8573 cuando el correcto era fy=0.7785 → desfase de 56 px hacia
arriba, exactamente lo que se veia en la captura.

Tambien aprovechamos el ciclo para 2 mejoras de housekeeping:
DECISIONS.md como indice navegable, y documentar el `FileNotFoundError
/tmp/gradio/*.jpg` esporadico como known upstream issue.

- [x] **Fix coord centering** ([scripts/generate_explain_reference.py](scripts/generate_explain_reference.py)):
      nuevo `_imshow_bbox_in_figure(ax_rect, fig_size_in, img_aspect)`
      retorna el rect REAL que ocupa el imshow dentro del ax_rect,
      compensando el centering de matplotlib. `_pixel_to_figure_coords`
      delega la geometria a ese helper y hace un mapping lineal dentro
      del bbox real. Nueva constante `FIG_SIZE_IN = (11.26, 7.16)`
      como source-of-truth (usada por _imshow_bbox_in_figure Y por
      `plt.figure(figsize=...)`).
- [x] **2 tests actualizados** (suite: **66 passed + 1 skipped**, era
      65+1): `test_pixel_to_figure_coords_accounts_for_aspect_equal_centering`
      (rename + asserts corregidos a los valores post-fix) +
      `test_imshow_bbox_centers_square_in_tall_rect` (nuevo, cubre
      ambas ramas width-limited y height-limited del bbox helper).
- [x] **`docs/DECISIONS.md` creado**: indice navegable por ciclo + por
      tema clinico + por tema arquitectonico + known issues. AGENTS.md
      sec 9 sigue como source-of-truth completa.
- [x] **Known issue documentado**: `FileNotFoundError /tmp/gradio/*.jpg`
      esporadico catalogado como upstream Gradio issue. Sin patch en
      este ciclo porque el crash ocurre en preprocessing de Gradio
      antes de nuestro callback (no atrapable desde Python).
- [x] Deploy via `scripts/upload_to_space.py` con `--path-in-repo`
      explicito (solo 2 PNGs).
- [x] Smoke remoto: PNGs regenerados servidos correctamente. Validacion
      visual final de Elvis: "las 5 flechas aterrizan EN los blobs,
      no encima."

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 22.

### Ciclo 6.0 ✅ COMPLETO — Pre-sustentacion (rubrica + README + guia oral)
Elvis tiene sustentacion del proyecto MAÑANA (2026-05-23). Pidio
verificar (a) que la rubrica Coursera/U. Andes esta expuesta claramente,
(b) el estado de GitHub (data, pesos, notebooks), (c) tener una guia
clara para explicar y sustentar el proyecto uniendo informe IEEE +
plataforma + despliegue.

Hallazgos criticos de la exploracion:
- README tenia 8 ocurrencias del placeholder `<usuario>/spine-segmentation`
  (jurado leeria "por completar tras crear el Space" cuando la app YA
  llevaba semanas RUNNING — bug latente desde Ciclo 4).
- `docs/HF_SPACES_SETUP.md` tenia 12 ocurrencias del mismo placeholder.
- Rubrica exige 3 carpetas raiz: `notebooks/` ✅, `modelos/` ❌, `datos/`
  ❌. Habia que crear las 2 faltantes con READMEs explicativos.
- Notebook del companero (Julian) `escoliosos_colab_Jul.ipynb` (4.8 MB)
  existia local pero NO commiteado al repo.
- Paper IEEE reporta Dice binario 0.88; repo enfatiza Dice multiclass
  0.34 — son tareas distintas, no contradictorias, pero hay que armar
  la narrativa explicita para que Elvis no se trabe en Q&A.

Cambios:

- [x] **Fix README placeholders** (`README.md` + `docs/HF_SPACES_SETUP.md`):
      20 reemplazos via sed de `<usuario>` → `ElvLandau`. Audit
      post-fix: grep retorna 0 ocurrencias del placeholder; 22
      ocurrencias nuevas de `ElvLandau/spine-segmentation`. URL real
      visible en la primera tabla del README ahora.
- [x] **Notebook del equipo committeado** como
      [`notebooks/02b_training_alternativo_unet_keras.ipynb`](notebooks/02b_training_alternativo_unet_keras.ipynb)
      (4.8 MB). Convencion de nombre `0Nb_*.ipynb` marca "variante de
      0N" — el principal `02_training_experiments.ipynb` (Elvis,
      PyTorch+SMP, 5 modelos) sigue siendo el que alimenta el deploy.
      [`notebooks/README.md`](notebooks/README.md) (tabla de los 4
      notebooks) aclara cual es el desplegado.
- [x] **Carpetas `modelos/` y `datos/`** creadas con READMEs
      explicativos (~150 LOC total), satisfaciendo la rubrica letra
      sin duplicar bytes:
      [`modelos/README.md`](modelos/README.md) explica que los pesos
      viven en HF Hub `ElvLandau/spine-checkpoints` (226 MB, decision
      Ciclo 4).
      [`datos/README.md`](datos/README.md) explica que el dataset es
      propiedad U. Andes (no redistribuible) + estructura esperada +
      contacto para acceso academico.
- [x] **[`docs/SUSTENTACION_GUIA.md`](docs/SUSTENTACION_GUIA.md)** —
      documento operativo (~476 lineas) en 12 secciones para que Elvis
      lo abra en el browser durante la presentacion:
      1) resumen ejecutivo,
      2) equipo + division del trabajo,
      3) entregables verificables + mapping rubrica,
      4) mapeo paper IEEE ↔ app deployada (tabla),
      5) **metricas explicadas** (CRITICO para Q&A: Dice binario 0.88
         vs multiclass 0.34 NO son contradictorias),
      6) demo paso-a-paso (6 actos, ~5 min: tour UI → S_158 severo →
         Explainability → slider rotacion N_61 → multi-curva S_100 →
         cierre),
      7) narrativa 10 min con timing,
      8) **10 Q&A anticipadas** con respuestas concretas,
      9) cheat sheet de numeros clave,
      10) links rapidos para tabs abiertos,
      11) plan B si algo falla en vivo,
      12) recordatorios finales (honestidad, disclaimer, pausa).
- [x] **Sin deploy al Space** — este ciclo es solo docs. Tests siguen
      verdes (66 passed + 1 skipped). Space sigue RUNNING.

Artefacto: [`docs/CICLO_6_ARTEFACTOS.md`](docs/CICLO_6_ARTEFACTOS.md).

### Ciclo 6.1 ✅ COMPLETO — Fix de lateralidad por chord signed-area
Post-sustentacion (2026-05-22): la medica colaboradora reporto con 5
capturas que la lateralidad seguia saliendo invertida en MUCHOS casos
pese al fix del Ciclo 5.10. Evidencia principal: una S-shape con
principal T11-L2 88.2° + secondary T4-T11 65.0° AMBAS reportadas
"izquierda" — anatomicamente imposible porque las dos curvas
separadas por un inflection point tienen convexidades OPUESTAS por
definicion.

Diagnostico raiz: el helper del Ciclo 5.10 usaba `dx_dy[mid_idx]`
(slope del spline en el midpoint geometrico entre los dos IPs) como
proxy para la convexidad. Eso falla porque el slope pasa por cero EN
EL APEX, no en el midpoint — el signo en el mid depende de la
asimetria temporal de la curva, no de la convexidad anatomica.

Cambios:

- [x] **Algoritmo nuevo `_curve_direction` (chord signed-area)** en
      [`spine_segmentation/evaluation/cobb_angle.py`](spine_segmentation/evaluation/cobb_angle.py).
      Firma nueva `(x_eval, y_eval, ip_a, ip_b, neutral_threshold_px2=50.0)`.
      Convexidad = signo del signed area entre la curva y la chord
      que une los 2 IPs (cross product 2D normalizado). Invariante a
      asimetria temporal, garantiza opposing en S-shapes.
- [x] **Suite sintetica nueva** (6 tests) en `tests/test_app_smoke.py`:
      parabolas right/left, S-shape canary (falla bajo Ciclo 5.10),
      chord casi-vertical, neutral threshold parametrizable, edge cases.
- [x] **Tests anchored al ground truth oficial** en
      [`tests/test_cobb_laterality_real.py`](tests/test_cobb_laterality_real.py)
      (nuevo): S_158 y S_22 validan contra `apex_x` vs `csvl.x_px` del
      `metrics_json/` oficial. Gated por skipif del dataset.
- [x] **Script de sweep visual** en
      [`scripts/sweep_laterality.py`](scripts/sweep_laterality.py) (nuevo):
      CLI que procesa N casos via `SpineSegmentationPipeline.predict()`
      y emite tabla MD/CSV. Reutilizable para futuros ciclos.
- [x] **Sweep baseline vs post-fix sobre 12 casos**: 5/7 S-shapes
      violaban el principio del IP pre-fix; 6/7 lo cumplen post-fix.
      3/3 contra GT oficial post-fix (era 2/3). Outputs en
      `outputs/sweep_laterality_baseline_cycle6_0.md` y
      `outputs/sweep_laterality_cycle6_1.md`.
- [x] **Deploy via upload_to_space.py** + smoke remoto sobre 4 casos
      (S_158, S_22, S_100, S_200) — todos reportan la lateralidad
      esperada. S_22 cambio de `left` a `right`, S_100 ahora reporta
      `principal derecha + secundaria izquierda` (era `right + right`).
- [x] **Suite pytest**: 73 passed + 1 skipped (era 66 + 1; -1 test
      del Ciclo 5.10 reemplazado, +6 sinteticos +2 anchored).

Conocido pendiente (no parte del 6.1, decision de Elvis):

- [ ] **Ciclo 6.2 candidato**: bug separado en
      `assign_vertebra_names_to_curves` que reporta `T6-T5` como
      secundaria (upper > lower, anatomicamente raro). Vive en otra
      funcion. Documentado como known issue.

Artefacto: [`docs/CICLO_5_ARTEFACTOS.md`](docs/CICLO_5_ARTEFACTOS.md) sec 23.

### Ciclo 6.2+ (post-sustentacion) — Refinamiento del modelo + entrega final
- [ ] Mejorar Cobb multiclase (SVD sobre centroides, constraint biomecanico
      post-proc, votacion robusta)
- [ ] Enmascarar confidence map por la prediccion (idea identificada Ciclo 5)
- [ ] Probar Seg-Grad-CAM en vez de Grad-CAM vanilla
- [ ] Quantizacion INT8 para edge (tablet)
- [ ] Refinamiento modelo (augmentation, ensemble, pre-training RadImageNet)
- [ ] CI con GitHub Actions (opcional)
- [ ] Articulo IEEE/ACM (si los resultados lo soportan)
- [ ] Slides de sustentacion + demo en vivo + smoke test cross-device

### Metricas (se actualiza al completar entrenamientos)
### TABLA FINAL — 5 MODELOS MULTICLASE ENTRENADOS

| Ranking | Modelo | Paradigma | Test Dice | Test IoU | PixAcc |
|---------|--------|-----------|-----------|----------|--------|
| 🥇 | **deeplabv3plus_resnet50** | CNN multi-escala (ASPP) | **0.3378** | **0.2556** | **0.9596** |
| 🥈 | manet_mit_b5 | Transformer + atencion dual | 0.3271 | 0.2383 | 0.9594 |
| 🥉 | unet_mit_b3 | Transformer (SegFormer) | 0.3157 | 0.2323 | 0.9578 |
| 4° | unet_resnet50 | CNN baseline | 0.2691 | 0.1883 | 0.9541 |
| 5° | unet_efficientnet_b4 | CNN eficiente (tablet) | 0.2189 | 0.1542 | 0.9548 |

### COBB ANGLE MAE (grados) — eval en casos de escoliosis del test set

| Modelo | Metodo | MAE | Correlacion |
|--------|--------|-----|-------------|
| unet_efficientnet_b4 (binary) | Skeleton | 23.0 | 0.66 |
| unet_resnet50 (binary) | Skeleton | 25.5 | 0.56 |
| unet_mit_b3 (multi) | Endplate | 28.2 | 0.27 |
| unet_efficientnet_b4 (multi) | Endplate | 26.8 | 0.20 |
| unet_resnet50 (multi) | Endplate | 39.4 | -0.12 |
| manet_mit_b5 (multi) | Endplate | 42.0 | -0.20 |
| deeplabv3plus (multi) | Endplate | 45.4 | -0.20 |

### HALLAZGOS IMPORTANTES

1. **DeepLabV3+ supero a los transformers** (contrario a hipotesis inicial)
   - ASPP con convoluciones atrous captura contexto multi-escala
   - Mejor para vertebras que varian mucho en tamano (cervicales vs lumbares)
   - No necesito self-attention global

2. **Transformers quedaron 2° y 3°** — buen desempeno pero no ganaron
   - Dataset pequeno (174 imgs) limita el aprendizaje de atencion
   - Aun asi supera a U-Net+ResNet50 clasico

3. **Cobb angle: metodo binario (skeleton) gana** al metodo multiclase
   - Errores en segmentacion multiclase se acumulan al calcular Cobb
   - Metodo binario es mas robusto
   - Para deploy clinico, usar binary para Cobb + multiclass para visualizacion

4. **Per-class Dice** (del mejor modelo, DeepLabV3+):
   - C7: 0.69, T1: 0.66, L5: 0.41 (buenas)
   - C4: 0.11, C3: 0.00 (no detectadas - vertebras muy raras en dataset)
   - Desbalance extremo limita el aprendizaje de clases raras

---

## 5.bis Metodología: Spec-Driven Work + Work Orchestration

El proyecto sigue la metodología de Leonardo Gonzalez:
- **Six Pillars of Spec-Driven Work** (2025)
- **From Spec-Driven Work to Work Orchestration** (2026)
- Transcripción del curso en `orchestantion agentic.txt`

### Los 6 Pilares aplicados aquí
1. **Colaboración sobre delegación** — humano + agente trabajan juntos, no solo "ejecuta esto"
2. **Procesos trazables** — toda decisión registrada en este AGENTS.md con su razón
3. **Orquestación entre herramientas** — Git + OneDrive + GPU local + Hetzner (futuro)
4. **Buenas specs** — `docs/CICLO_N_*_BRIEF.md` antes de ejecutar cada ciclo
5. **Spec + test + rule-driven** — `WORKFLOW.md` define reglas, cada cambio se verifica
6. **Artefacto final = input del siguiente ciclo** — `docs/CICLO_N_ARTEFACTOS.md`

### Los 5 Principios de Autonomía aplicados aquí
1. **Externalizar contexto útil** — AGENTS.md, docs/, no en chat
2. **Descomponer el trabajo** — unidades <8h por commit (ver WORKFLOW.md sec 3)
3. **Especialización** — diferentes scripts para train/eval/cobb/explainability
4. **Verificación = generación** — métricas en cada ciclo, no solo "el código corre"
5. **Legibilidad del repositorio** — `docs/RUTA_LECTURA.md` para nuevos colaboradores

### Estructura de ciclos
```
Ciclo N:
  Input:    docs/CICLO_(N-1)_ARTEFACTOS.md + AGENTS.md
  Trabajo:  unidades <8h, cada una con commit
  Output:   docs/CICLO_N_ARTEFACTOS.md + AGENTS.md actualizado
```

Reglas operativas en [`WORKFLOW.md`](WORKFLOW.md).

---

## 6. Configuracion de Entrenamiento

```python
# config.py - parametros actuales (Ciclo 3)
batch_size = 12         # Usa ~12 GB de 16 GB VRAM
image_size = 512        # Preserva detalle vertebral
encoder_lr = 1e-5       # Encoder pretrained: LR bajo
decoder_lr = 1e-4       # Decoder random: LR alto
epochs = 150            # Con early stopping patience=20
loss = WCE + GDice      # Weighted Cross-Entropy + Generalized Dice
optimizer = AdamW        # Con weight_decay=1e-4
scheduler = CosineAnnealing  # T_0=10, T_mult=2
AMP = True              # Mixed precision FP16
```

Comando de entrenamiento:
```bash
python scripts/train_multiclass.py --model <nombre> --scheme vertebrae_24
```

---

## 7. Deploy (dos niveles)

```
SERVIDOR (Hetzner) ─── Modelo completo (MAnet+MiT-B5, 352MB)
  │                    + Grad-CAM + Confianza + Reporte clinico
  │                    Docker: pytorch CPU, ~2-3GB imagen
  │ API REST
TABLET (consultorio) ─ Modelo ligero (EfficientNet-B4, 19MB INT8)
                       Opcion offline o via API al servidor
```

---

## 8. Para Nuevo Chat/Agente

**Antes de hacer cualquier cosa, sigue la ruta de lectura formal:**
→ [`docs/RUTA_LECTURA.md`](docs/RUTA_LECTURA.md)

Orden corto:
1. Leer este `AGENTS.md` completo
2. Leer [`WORKFLOW.md`](WORKFLOW.md) (reglas no negociables del repo)
3. Leer el ultimo `docs/CICLO_N_ARTEFACTOS.md` (estado actual real)
4. Si vas a continuar trabajo: leer el `docs/CICLO_(N+1)_*_BRIEF.md`
5. Revisar `config.py` para hiperparametros y rutas
6. Revisar `checkpoints/` para ver que modelos hay (si no estan, descargar de OneDrive)

**Al cerrar tu ciclo de trabajo:**
- ACTUALIZA este archivo (estado, metricas, decisiones)
- CREA `docs/CICLO_N_ARTEFACTOS.md` con los outputs del ciclo
- CREA `docs/CICLO_(N+1)_*_BRIEF.md` con la spec del siguiente
- ACTUALIZA `docs/PROMPT_PROXIMO_CHAT.md` para retomar el siguiente chat
- Commit con mensaje siguiendo `WORKFLOW.md` sec 4

---

## 9. Historial de Decisiones

| Fecha | Decision | Razon |
|-------|----------|-------|
| 2026-04-13 | PyTorch + SMP en vez de TensorFlow | SMP da acceso a multiples arquitecturas con 1 linea. TF requiere construir decoders manualmente. |
| 2026-04-13 | batch_size=8 en vez de 4 | Con bs=4 solo se usaban 4.8GB de 16GB VRAM. bs=8 usa ~8-10GB. |
| 2026-04-13 | Esquema 24 clases (no 36) | Las 13 "entidades" (costillas, pelvis) no son clinicamente relevantes. 22 vertebras + bg + other = 24. |
| 2026-04-13 | Agregar MiT-B3 y MAnet+MiT-B5 | Transformers con self-attention global resuelven oclusion vertebral por rotacion en escoliosis severa. CNNs solo ven ventanas locales. |
| 2026-04-13 | Solo multiclase, NO binario | El binario era del semestre anterior como guia. La solucion real es segmentar vertebras individuales. |
| 2026-04-13 | Explicabilidad obligatoria | AI medica requiere interpretabilidad. El medico necesita ver POR QUE el modelo decidio (Grad-CAM + confianza). No cajas negras. |
| 2026-04-13 | Deploy en 2 niveles (servidor + tablet) | Modelo completo en Hetzner via API, modelo ligero INT8 para tablet offline. |
| 2026-04-13 | batch_size 8 → 12 (Ciclo 3) | Solo se usaban 8 GB de 16 GB de VRAM. bs=12 usa ~12 GB y acelera. |
| 2026-04-13 | Compartir pesos via OneDrive, no GitHub | Pesos pesados (>100MB) no encajan en versionado de codigo. ZIP de 776 MB para el equipo. |
| 2026-04-13 | Exportar checkpoints inference-only | Quitar optimizer_state_dict reduce 66% el tamano. Suficiente para inferencia. |
| 2026-04-13 | Notebook 03_informe_final estilo lab semestre anterior | Familiar para el equipo + funciona en CPU. Cumple objetivo de reproducibilidad. |
| 2026-04-20 | Adoptar metodologia Spec-Driven + Work Orchestration | Cada ciclo cierra con artefacto en docs/, AGENTS.md como memoria. Patron OpenSymphony. |
| 2026-04-20 | Crear WORKFLOW.md (policy del repo) | Reglas no negociables visibles para humanos y agentes. Pilar 5 (rule-driven). |
| 2026-04-20 | Crear docs/ con artefactos por ciclo | Cada ciclo cierra con CICLO_N_ARTEFACTOS.md (input del siguiente). Pilar 6. |
| 2026-04-20 | Crear docs/PROMPT_PROXIMO_CHAT.md | Onboarding instantaneo del proximo chat con Claude. Aplica externalizar contexto. |
| 2026-05-17 | Migrar pesos de OneDrive a Hugging Face Hub | Estandar industria ML, gratis, versionado tipo git, soporta archivos grandes, intercambio sin re-deploy (cambias el .pth en HF y reinicias el container). |
| 2026-05-17 | Servir solo DeepLabV3+ en produccion (no los 5 modelos) | Ganador del Ciclo 3 con margen. Ahorra ~700 MB de RAM y 30-60s de arranque sin perder valor para el usuario final. Los otros 4 viven en HF + notebook para analisis comparativo. |
| 2026-05-17 | Convencion `app/main.py` como shim, no renombrar paquete | Cumple letra de la rubrica ("carpeta app/ o src/") sin churn en notebooks, scripts e imports del equipo. |
| 2026-05-17 | Caddy en vez de Nginx para reverse proxy + SSL | Let's Encrypt automatico en 1 linea de config vs cert-bot + cron + dhparams. Para un deploy academico, la complejidad del nginx es un impuesto sin upside. |
| 2026-05-17 | Dominio gratis con nip.io en vez de comprar uno | El equipo no tiene dominio asignado; nip.io da hostname resoluble + Let's Encrypt lo firma. Cambiar a dominio propio es 1 env var despues. |
| 2026-05-17 | .dockerignore agresivo (sin docs, notebooks, dataset, mlruns) | Reduce build context de ~5 GB a < 100 MB. Build mas rapido, menos data al daemon. |
| 2026-05-17 | Tests con pytest desde Ciclo 4 (no antes) | Hasta el Ciclo 3 la prioridad fue entrenar modelos. Con deploy llega el momento de pinear contratos (config, weights, app boot, disclaimer). |
| 2026-05-17 | Usuario no-root (uid 1000) en el container | Buena practica de seguridad estandar. Tiene su home y permisos chown sobre /data/checkpoints. |
| 2026-05-17 | Crear CLAUDE.md en raiz como pointer a AGENTS.md | AGENTS.md es la fuente publica auditable (estandar agents-md.io). CLAUDE.md existe solo para que Claude Code lo cargue automaticamente. Una sola fuente de verdad. |
| 2026-05-17 | Reorganizar raiz en carpetas tematicas (requisitos_universidad/, docs/metodologia/, archive/) | PDFs sueltos en raiz era ruido. Las nuevas carpetas tienen README explicativo y politicas claras. Los PDFs de la rubrica oficial SI se commitean (excepcion en .gitignore) por ser el contrato academico. |
| 2026-05-17 PM | **Pivote: HF Spaces como hosting principal (no Hetzner)** | Gratis, sin servidor propio, integrado con HF Hub de pesos. URL `huggingface.co/spaces/ElvLandau/spine-segmentation`. La infra Hetzner queda commiteada como "deploy alternativo" pero NO es el camino oficial. |
| 2026-05-17 PM | Pin `python_version: "3.11"` en YAML del Space | Python 3.13 (default HF) elimino `audioop` del stdlib; `pydub` (dep de gradio) lo necesita. |
| 2026-05-17 PM | Upgrade `gradio` a 5.x con `sdk_version: 5.0.1` | gradio 4.44 venia con `gradio-client 1.3` que tiene bug `TypeError 'bool not iterable'` en `api_info()`. La 5.x trae client 1.5+ con el fix. |
| 2026-05-17 PM | `create_app()` usa sentinel `None` y resuelve via config | Los defaults hardcoded `"unet_resnet50"` ignoraban `DEFAULT_MULTICLASS_MODEL`, causando mismatch al cargar pesos DeepLabV3+ en un Unet. La fuente de verdad es ahora `config.DEFAULT_*_MODEL` (env-driven). |
| 2026-05-17 PM | `app.py` raiz hace `launch(server_name="0.0.0.0", server_port=7860)` | HF Spaces a veces re-invoca como script tras crash; sin `server_name="0.0.0.0"` el container no puede bind a localhost. |
| 2026-05-17 PM | Crear repo de pesos `ElvLandau/spine-checkpoints` publico en HF Hub | 2 .pth subidos via `scripts/upload_weights.py`: DeepLabV3+ multiclase (102 MB) + UNet binario (124 MB) = 226 MB total. |
| 2026-05-17 PM | Force push al Space autorizado (1 vez) | El Space tenia un README placeholder auto-creado por HF; sobrescribir era necesario y seguro porque el Space era brand-new sin colaboradores. NO se uso force push para GitHub. |
| 2026-05-17 PM | LFS para PDFs oficiales | HF Spaces rechaza binarios no-LFS via su hook xet. Los 2 PDFs de Coursera quedaron en LFS en el branch `space-deploy` (solo para el push al Space; main de GitHub mantiene los blobs originales). |
| 2026-05-17 noche | Upgrade Gradio a 5.50.0 + bump huggingface_hub a >=0.33.5 | Resolvio el `TypeError: argument of type 'bool' is not iterable` en `gradio_client/utils.py:get_type()` que bloqueaba `/api_info` y por tanto el boton Analyze. La 5.0.1 tiene el bug; cualquier 5.30+ trae el fix del schema bool (issues GH #11116 y #11722); 5.50.0 es la ultima 5.x estable. El pin viejo de `huggingface_hub<0.28` era legacy de gradio 4.44 (`HfFolder`), y entraba en conflicto con el requisito de gradio 5.50 (`>=0.33.5`). |
| 2026-05-17 noche | Script `scripts/upload_to_space.py` para subir cambios via HfApi (no git push) | El Space ya tiene historia LFS limpia que no queremos sobrescribir. `HfApi.upload_file()` y `create_commit()` permiten patchear archivos individuales en un commit atomico sin tocar git remoto. Patron hermano de `upload_weights.py`. |
| 2026-05-17 noche | Assessment severity calculado sobre Cobb binary (no multiclass) | El binary es mas robusto en nuestros datos (MAE 23° + Pearson 0.66) vs multiclass (MAE 26-45°, correlacion negativa en DeepLabV3+: r=-0.20). El multiclass se queda visible como referencia anatomica (Upper-Lower vertebrae). Resuelve los falsos negativos donde escoliosis salia "Normal" por usar el multiclass ruidoso. |
| 2026-05-17 noche | Visualizacion Cobb tipo Fig 1 del paper Shi et al. 2025 | Cajas verdes en upper/lower end vertebrae + lineas tangentes rojas a los endplates + header numerico. Mejora dramatica de UX clinica sin reanotar el dataset. Implementado en `evaluation/visualize.py:draw_cobb_angle_visualization`. |
| 2026-05-17 noche | UI dual-Cobb con indicador de concordancia | Muestra binary y multiclass en paralelo + bloque CONCORDANCIA (<=5° alta / <=15° revisar / >15° discrepancia). Da al medico contexto para decidir, en vez de un solo numero. Pure helper `build_results_text` extraido a module-level para tests. |
| 2026-05-17 noche | NO replicar paper Shi et al. 2025 completo | Su modelo HRNet+Swin+dual-task requiere landmarks anotados de upper/lower endplate por vertebra. Nuestro MaIA tiene mascaras de segmentacion (no landmarks). Reanotar es trabajo clinico de semanas, no viable en este semestre. Si extrapolamos sus ideas A (visualizacion) y B (fusion de mediciones) — Ciclo 5. VWI y SVD sobre matriz de angulos quedan fuera (requieren los mismos landmarks). |
| 2026-05-17 noche (Ciclo 5.1) | Cobb viz: perpendiculares al endplate (no a lo largo) + arco + speedometer + overlay del binary | La viz del Ciclo 5 dibujaba lineas A LO LARGO del endplate. Convencion clinica + Fig 1 del paper usa perpendiculares (las que cruzan formando el angulo visible). Para angulos pequeños (<8°) las perpendiculares quedan casi paralelas y la interseccion sale fuera del frame: usar mini "Cobb-meter" en esquina como fallback con la aguja escalada 4x para que se mueva visiblemente. Se aprovechan los datos del binary (`spline_x/y` + `inflection_points`) que ya devolvia `cobb_from_binary` y que la viz no usaba. |
| 2026-05-17 noche (Ciclo 5.1) | Convencion de color RGB (no BGR) en visualizaciones que pasan por Gradio | La imagen llega a `draw_cobb_angle_visualization` como RGB (Gradio convencion). cv2 no transforma color spaces — solo escribe los 3 valores en orden. Pasar `(0, 0, 255)` (rojo BGR canonical) en un array RGB pinta AZUL puro. Todas las llamadas cv2 que esperan rojo usan ahora `(255, 0, 0)`; similar para amarillo `(255, 255, 0)`. Detectado por smoke remoto que encontro 0 red pixels. |
| 2026-05-17 noche (Ciclo 5.2) | `cobb_from_binary` devuelve `curves: list[dict]`, no un solo angulo | El algoritmo original solo usaba los 2 inflection points mas extremos del spline y reportaba UN angulo. Para escoliosis con doble curva (S-shape, comun en pacientes con curva toracica principal + curva lumbar compensatoria, como describe Julian Florido en su ejemplo de informe radiologico real), la segunda curva nunca aparecia en el reporte. Ahora se calcula un Cobb por cada par adyacente de IPs, se filtra debajo de 3° (ruido), y se ordenan por magnitud. Confirmado en smoke remoto con S_100: detecta 84° toracica + 65° lumbar. |
| 2026-05-17 noche (Ciclo 5.2) | Multiclass solo para naming + ilustracion, no para Assessment | Su Dice 0.34 (y el de Julian 0.09) son insuficientes para calcular Cobb por endplate. El multiclass se mantiene en el pipeline porque sus mascaras dan info anatomica (que vertebrae componen cada curva del binary) — `assign_vertebra_names_to_curves` hace label-transfer por nearest-y. Assessment severidad usa la curva mayor del binary, no el multiclass. |
| 2026-05-19 (Ciclo 5.3) | Cierre morfologico vertical en `clean_binary_mask`, antes del filtro por componente mayor | Cases tipo S_22 (S-shape rotoescoliosis con señal lumbar debil) producian un binary mask en dos fragmentos: uno toracico fuerte y otro lumbar tenue. El "keep largest connected component" descartaba el lumbar antes de poder pegarlo, y el spline se ajustaba solo a la mitad superior. Un kernel rect (3, 25) — alto y angosto — alcanza ~12 px arriba y ~12 px abajo: suficiente para puentear gaps lumbar tipicos (10-20 px en 512x512), demasiado angosto para arrastrar costillas o pelvis. El orden (cerrar antes de filtrar) es lo critico. |
| 2026-05-19 (Ciclo 5.3) | Multi-pass adaptativo en `cobb_from_binary` (smoothing usuario + retry con smoothing*3.3 si solo 1 curva) | El sweep contra el dataset mostro que NO existe un solo `smoothing_factor` que funcione para todos los casos: S_22 (S-shape sutil + coverage parcial) necesita smoothing bajo (~1500) para que el spline conserve las inflexiones; S_100 (S-shape severa, principal 84° + lumbar 65°) necesita smoothing alto (~5000) para que el spline no fusione las dos curvas. Solucion: refactor a `_cobb_from_binary_single_pass` + envoltorio que corre la pasada del usuario primero y reintenta con higher smoothing cuando encuentra exactamente 1 curva. Costo computacional cero en casos triviales (0 curvas) o ya-multi-curva (≥2 curvas). Preserva el comportamiento de Ciclo 5.2 en S_100 sin perder la sensibilidad nueva en S_22. |
| 2026-05-19 (Ciclo 5.3) | `compute_coverage_info` como modulo aparte (`evaluation/coverage.py`) y no dentro de `morphology.py` | La cobertura combina datos del binary (y-range, ratio) con la info anatomica del multiclass (nombres de vertebrae). No es estrictamente una operacion morfologica — es analisis del resultado de la segmentacion + label-transfer. Vive junto a `cobb_angle.py` porque son del mismo nivel conceptual (evaluation/) y comparten consumidor (build_results_text). El helper retorna `vertebrae_in_range`, `vertebrae_below_range`, `vertebrae_above_range` para que la UI pueda escribir "Lower spine (T11-L5) NOT segmented" en vez de un warning generico. |
| 2026-05-19 (Ciclo 5.3) | Assessment "Inconclusive - insufficient coverage" cuando partial + 0° | El bug original era que con `coverage_ratio < 0.7` y `cobb_angle_deg = 0`, el Assessment decia "Normal (< 10 degrees)" — falso negativo clinico. La nueva rama lo flag-ea como inconcluyente para que el medico revise la segmentacion antes de confiar en el numero. La rama solo se dispara con AMBAS condiciones (partial + ~0°) para no contaminar casos sanos verdaderos (N_1: coverage full + 0° -> sigue siendo "Normal"). |
| 2026-05-19 (Ciclo 5.4) | Deteccion de rotacion via SVD del skeleton + ROTATION WARNING en UI (Cobb binary tal cual) | `cobb_from_binary` ajusta un spline `x = f(y)` que asume columna vertical. N_61 (radiografia Normal rotada) producia 4 curvas fantasma 31.8°+20°+17.1°+14.1° cuando el multiclass — rotation-invariant por construir endplate angles per-vertebra — reportaba 0.6° correcto. SVD sobre los skeleton points (centrados) da el eje principal; su angulo vs vertical es la tilt magnitude. Threshold 12° (clinico: filtra jitter de captura, dispara con rotacion clara). Decision UX: el Cobb binary se sigue mostrando, NO se hide ni se cambia el Assessment. El medico ve el dato y el warning lado a lado y decide. |
| 2026-05-19 (Ciclo 5.4) | `MIN_IP_Y_DISTANCE_PX = 30` para filtrar pares de inflection points sub-vertebrales | El sweep del Ciclo 5.3 dejaba pasar curvas con IPs separadas <30 pts en el grid del spline (500-pt span). Esos pares producian Cobb numericamente valido (>min_curve_deg) pero anatomicamente espurios — la wiggle del spline sub-vertebra. En S_22 esto generaba "5.9° T9-T9" como secundaria espuria. 30 pts ~= 6% del alto, ~= 1 vertebra a 512x512. Curvas mas cortas se filtran como spline noise. |
| 2026-05-19 (Ciclo 5.4) | Eliminar (no marcar) curvas con `upper_vertebra == lower_vertebra` despues del label transfer | El filtro y-distance pasa la mayoria de los espurios, pero el label transfer nearest-y puede aun colapsar 2 IPs distintos al mismo centroide multiclass si la deteccion multiclass es poco densa en esa zona. Resultado: "T9-T9" como nombre cuando T9 es la unica vertebra cercana en y. Eliminar in-place + reindexar `rank` (en `assign_vertebra_names_to_curves`) + resync `cobb_angle_deg`/`inflection_points` en inference.py si la principal fue eliminada. Curvas con names == None (sin multiclass) NO se filtran — todavia llevan info geometrica util. |
| 2026-05-19 (Ciclo 5.4) | Dedup de rotulos por nombre de vertebra + anti-overlap por desplazamiento vertical | Cuando 2 curvas comparten vertebra (T9 = lower principal + upper secundaria), el codigo previo dibujaba 4 textos `[Principal] Inferior (T9)`, `[Secundaria] Superior (T9)`, etc. en el mismo `text_y`, produciendo una pared ilegible. Solucion en 2 capas: (1) dedup por `v["name"]` — si ya hay rotulo de una curva anterior, omitir; la geometria (boxes/perps/arco) sigue dibujada. (2) Si el rotulo nuevo colisionaria con un rect placed, desplazar `text_y` 16px abajo iterativamente hasta encajar o salir del frame. Acumuladores `labeled_vertebrae: set` y `placed_label_rects: list` se instancian en `draw_cobb_angle_visualization` y se pasan al helper por cada curva. |
| 2026-05-19 (Ciclo 5.5) | Control manual de rotacion en la UI (slider + 5 botones rapidos) en lugar de auto-rotacion | Auto-rotacion habria sido riesgosa: (a) S_100/S_150 borderline (tilt 12.6/12.8°) habrian sido rotados, posiblemente cambiando magnitudes anatomicas; (b) la convencion de signo entre `compute_orientation_info` (que usa SVD + arctan2 + wrap a (-90, 90]) y `cv2.getRotationMatrix2D` (positive=CCW) no es trivial — empiricamente confirme que `-tilt_deg` EMPEORA la rotacion en N_61, no la corrige. El control manual (slider -180..180 + 5 botones) elimina ambos problemas: el medico ve la imagen, decide si rotar, ajusta y obtiene feedback visual antes de Analyze. Generaliza ademas a rotaciones grandes (90° = radiografia atravesada) que ningun threshold automatico habria capturado. El warning del 5.4 sigue activo como guia post-rotacion ("you rotated, still tilted X°"). |
| 2026-05-19 (Ciclo 5.5) | Helper `rotate_image_for_analysis` con `BORDER_REPLICATE` y deadband 0.5° | `BORDER_REPLICATE` rellena bordes con el pixel mas cercano del original — para una radiografia donde el fondo ya es oscuro uniforme, produce un fade limpio en vez de bandas negras que el binary segmenter podria confundir con la cavidad toracica. Deadband 0.5° evita que el slider sentado en 0 pague el costo de `cv2.warpAffine` (que introduce ruido sub-pixel de interpolacion bilinear en cada llamada). Cero cambios en el pipeline interno — toda la cirugia es en el callback de Gradio. |
| 2026-05-19 (Ciclo 5.6) | `gr.State` para imagen original + live preview en `rotation_slider.change` | Sin un state separado, el preview rotaria la imagen MOSTRADA (que a su vez fue producto del cambio anterior del slider), acumulando rotaciones. Con `gr.State` la slider rota siempre la ORIGINAL upload. `input_image.upload` (que solo fira para uploads de usuario, no para escrituras programaticas del preview) guarda la original + resetea el slider a 0. Resultado: dragging el slider de 0 → 10 → 20 rota la original por 20°, no 30°. Apreciable visualmente en <300ms. |
| 2026-05-19 (Ciclo 5.6) | Botones de rotacion atomicos: retornan `(new_slider, rotated_image)` en un solo handler | Confiar solamente en `rotation_slider.change` para que dispare tras un update programatico no es robusto entre versiones de Gradio. Cada boton ahora calcula el nuevo slider value, rota la original por ese valor, y retorna ambos a sus respectivos outputs en un solo round-trip. Atomic widget+display update, sin race condition entre los dos. |
| 2026-05-19 (Ciclo 5.6) | `predict()` pierde el parametro `rotation_deg` que tenia en 5.5 | El live preview hace que `input_image` al momento de Analyze ya este rotada visualmente — predict solo necesita delegar al pipeline. Mantener `rotation_deg` ademas seria riesgo de doble-rotacion (si el closure rota nuevamente lo que ya esta rotado). El helper `rotate_image_for_analysis` se queda como funcion pura, usado por `preview_rotation_for_display`. Contrato mas limpio + el regresion-pin `test_predict_callback_no_longer_takes_rotation_deg` evita volver al estado anterior. |
| 2026-05-19 (Ciclo 5.7) | Quitar el bloque `=== CROSS-CHECK binary vs multiclass ===` del frontend | Elvis cito: "no le aporta al usuario y lo termina confundiendo, ya por detras si el multiclass es con lo que marca o detalla la vertebra no hay problema". Los numeros del multiclass (a veces 90° degenerado por la naturaleza ruidosa del Dice 0.34 multiclass + clamp del arctan) parecian contradecir el binary, confundiendo al usuario. La info de naming (Tn-Lm) y las cajas verdes (que SI son utiles) se calculan internamente con el multiclass pero el usuario no ve sus numeros crudos. Solo cuando el binary FALLA totalmente, mostramos el multi-fallback (alli si es el unico signal). |
| 2026-05-19 (Ciclo 5.7) | i18n via dict en `i18n.py`, default Español, Nivel B (Diagnosis Results + Header markdown) | Elvis pidio toggle ES/EN para que un evaluador anglofono pueda leer la app sin Google Translate. Default Español porque el target audience es U. Andes / Colombia. Nivel B (no Nivel C/full UI) porque Gradio no permite cambiar labels de componentes facilmente sin recrear el Blocks — los labels de tabs / slider / botones quedan en ingles (mayormente simbolos: Reset, Analyze, ↺ -90°). Patron: `t(key, lang)` con fallbacks a Español y a la key como placeholder visible. El header markdown se re-renderiza en `language_radio.change`; el diagnosis text se re-traduce en el siguiente Analyze (no se re-corre el modelo solo para retraducir strings fijos). |
| 2026-05-20 (Ciclo 5.8) | Enmascarar Grad-CAM y Confidence Map por la `binary_mask` predicha + percentile clip p95 + mezcla con imagen original fuera del spine | Elvis cito: "las zonas que marca no son claras". El Grad-CAM pintaba toda la imagen (incluso fuera de la columna), confundiendo. Y el cmap RdYlGn aplicado al confidence map sin masking pintaba el fondo en rojo intenso (porque 0=rojo en RdYlGn), simulando "baja confianza en todo el fondo" cuando en realidad esas zonas NO se evaluaron. Solucion en 3 capas: (1) masking en `generate_gradcam`/`generate_confidence_map` con `prediction_mask` opcional → pixels fuera del spine = 0. (2) Percentile clip p95 en el cam → contraste mejorado. (3) Render layer en `app.py` mezcla la imagen original con el cam/conf usando `cv2.where(outside, img, cam)` → el medico ve la radiografia en grises fuera del spine y el color solo donde el modelo realmente analizo. |
| 2026-05-20 (Ciclo 5.8) | Anotaciones in-image (titulo + colorbar) en el panel side-by-side del Explainability | Sin titulos ni escalas visibles, el usuario tenia que adivinar que panel era cual y que significaban los colores. Nuevo helper `annotate_explainability_panel(cam, conf, language_label)` añade strip oscuro de 32px arriba de cada subpanel con el titulo, y un colorbar vertical de 18px a la derecha con etiquetas "Alta/Baja" o "High/Low". Las strings van por i18n para que el toggle ES/EN los traduzca. El Markdown debajo del panel ("Como leerlo / How to read it") tambien se traduce y explica que es un resultado bueno vs malo clinicamente. |
| 2026-05-20 (Ciclo 5.9) | Imagen fija de referencia clinica ARRIBA del panel dinamico de Explainability, bilingue ES/EN | Mockup educativo del medico companero con 5 callouts numerados (hot-spots, trayectoria esperada, activacion fuera de spine, alta confianza, bordes de menor certeza) + colorbars + disclaimer. Sirve como leyenda fija para que el medico aprenda a leer el panel dinamico del 5.8 ANTES de ver su caso. Wire en gr.Image(interactive=False, height=300) arriba del explain_output. El handler language_radio.change ahora actualiza header_md + explain_md + reference_image en un solo round-trip (extension natural del 5.7/5.8). |
| 2026-05-20 (Ciclo 5.9) | Recreacion programatica del mockup con scripts/generate_explain_reference.py en lugar de copiar el PNG original | El PNG original del medico companero no estaba disponible como path de archivo (solo como adjunto visual al chat). Recrear el panel con matplotlib + cv2 (sample radiograph S_22 + binary mask + simulated jet/RdYlGn overlays + callouts + colorbars) garantiza consistencia ES↔EN al 100%. Trade-off: la version recreada puede no coincidir pixel a pixel con el diseño original. Mitigacion: el script se commitea con `--lang es|en|both` para regenerar al cambiar strings o reemplazar la base. Si Elvis quiere el PNG exacto del medico, basta colocarlo manualmente en assets/ y re-correr el script solo para la traduccion EN. |
| 2026-05-20 (Ciclo 5.10) | Convencion de lateralidad en `_curve_direction` = anatomia del paciente (NO perspectiva del viewer) | Las radiografias AP siguen la regla del espejo: el lado derecho del paciente aparece en el lado izquierdo de la imagen para quien la mira. Los informes radiologicos SIEMPRE describen lateralidad en anatomia del paciente, no en perspectiva del viewer. Nuestra app reportaba la convexidad en perspectiva del viewer — caso S_158 de la MaIA (anatomicamente right-convex) salia como "convexidad izquierda", confuso para el medico. Fix: swap del ternario en `cobb_angle.py:73` (right ↔ left). Cambio contained: i18n/app/viz no requieren tocarse, los strings "derecha"/"izquierda" siguen funcionando, solo cambia su significado. Documentado in-code con docstring extenso + regresion test que pinea el mapping post-fix. |
| 2026-05-20 (Ciclo 5.10) | Sample base de la reference image: S_22 → S_200 | S_22 (Cobb 24.9°, mild) era una eleccion conservadora del 5.9 pero la compañera medica no lo reconocio como caso del dataset porque visualmente la columna se ve casi recta. S_200 muestra una S-curve clinicamente clara, suficiente para ilustrar Grad-CAM y Confidence Map en un caso pedagogicamente representativo sin ser tan extremo que el rendering se vea caricaturesco. Defaults del script bumped a S_200 para que cualquier re-generacion futura use el mismo caso "oficial". |
| 2026-05-20 (Ciclo 5.11) | Posiciones de blobs y arrows del reference image DERIVADAS del spine bbox, no hardcoded | El Ciclo 5.10 expuso un bug latente: tras cambiar de S_22 a S_200, las 5 flechas del reference image apuntaban a espacios vacios y los blobs sinteticos colisionaban con el spine real. Causa: `_simulate_gradcam` tenia los 4 `gaussian_blob` con cx/cy hardcoded para S_22 (cx=240, cy=70 para top, etc.), y `_build_figure` tenia los 5 `arrow_xy` hardcoded en figure-fraction calibrados para S_22 (spine centrado). Fix de raiz: 2 helpers nuevos `_derive_visual_anchors(mask)` y `_pixel_to_figure_coords(px, py, ax_rect)` hacen que TODO se derive del bounding box del spine. Asi el script es ahora invariante al sample base — cualquier radiografia que se cargue tendra blobs OFF-spine y arrows que aterrizan en algo visible. Constantes `AX_CAM_RECT` / `AX_CONF_RECT` a tope del modulo evitan drift entre las 3 funciones que necesitan saber donde estan los paneles. |
| 2026-05-22 (Ciclo 5.12) | `_pixel_to_figure_coords` debe compensar el centering de `aspect='equal'` cuando ax_rect no es cuadrada en inches | Mi propio fix del Ciclo 5.11 tenia un bug latente: asumi que el imshow llenaba toda la ax_rect, pero matplotlib renderiza un imagen 512x512 con `aspect='equal'` como un cuadrado centrado dentro del rect (anchor='C' default). Para AX_CAM_RECT (2.928 in wide x 4.153 in tall), la imagen sale 2.928 x 2.928 con 0.61 in (61 px) de margen arriba y abajo. Resultado: las flechas aterrizaban ~56 px ARRIBA de los blobs. Fix: nuevo `_imshow_bbox_in_figure(ax_rect, fig_size_in, img_aspect)` computa el rect real del imshow; `_pixel_to_figure_coords` delega ahi y hace un mapping lineal dentro del bbox real. Constante `FIG_SIZE_IN = (11.26, 7.16)` a tope del modulo es source-of-truth (usada tanto por el bbox helper como por `plt.figure(figsize=...)`) — evita drift entre los dos. |
| 2026-05-22 (Ciclo 5.12) | `docs/DECISIONS.md` como indice navegable por ciclo + tema | AGENTS.md sec 9 ya tiene >820 lineas; un jurado/medico/futuro agente no deberia tener que leerlo todo para encontrar el "por que" de una decision. DECISIONS.md es una vista curada en 3 ejes (por ciclo, por tema clinico, por tema arquitectonico) + seccion de known issues. AGENTS.md sec 9 sigue como source-of-truth completa; DECISIONS.md es vista resumida con links. Cada cierre de ciclo añade entradas a ambos. |
| 2026-05-22 (Ciclo 5.12) | `FileNotFoundError /tmp/gradio/*.jpg` esporadico catalogado como known upstream issue (sin patch) | El error en logs ocurre en `gradio.Image.preprocess()` (preprocessing de Gradio) ANTES de invocar nuestro callback `predict()`. No se puede atrapar con try/except en Python porque el stack trace nunca llega a nuestro codigo. Causas conocidas en HF Spaces: cold start purga /tmp, Gradio cleanup periodico, doble-click rapido en Analyze. Es esporadico, no afecta a la mayoria de sesiones; mitigacion natural es refresh del browser + re-upload. Si la frecuencia sube en produccion, opciones futuras: upgrade de Gradio (riesgo de regresion gradio-client bug del Ciclo 4.10) o pre-copiar uploads en `input_image.upload` handler. Documentado en DECISIONS.md sec "Known issues". |
| 2026-05-22 (Ciclo 6.0) | Fix README placeholders `<usuario>` → `ElvLandau` (20 reemplazos en README + HF_SPACES_SETUP) | Pre-sustentacion audit detectó que el README tenia 8 ocurrencias del placeholder con la nota "por completar tras crear el Space" — un jurado leyendolo concluiria que la app NO esta desplegada (cuando lleva semanas RUNNING). Bug latente desde Ciclo 4 que NUNCA se cerro. Sed global + audit post-fix con grep vacio + 22 menciones nuevas de `ElvLandau/spine-segmentation`. URL pública visible en la primera tabla del README. |
| 2026-05-22 (Ciclo 6.0) | Crear `modelos/` y `datos/` con README explicativo en lugar de subir pesos/data al repo | La rubrica Coursera/U. Andes exige 3 carpetas raiz: `notebooks/`, `modelos/`, `datos/`. Las dos ultimas estaban ausentes. Decision: NO subir los pesos (226 MB de .pth, viven en HF Hub desde Ciclo 4) ni el dataset (propiedad U. Andes, no redistribuible). En su lugar, READMEs de ~70 lineas cada uno que explican DONDE viven los artifacts reales + como obtenerlos. Satisface la letra de la rubrica + da contexto al evaluador en 30 seg de lectura. |
| 2026-05-22 (Ciclo 6.0) | Commitear notebook alternativo del equipo como `02b_training_alternativo_unet_keras.ipynb` (4.8 MB) | Julian (companero) desarrollo un pipeline en Keras/TensorFlow + Colab paralelo al pipeline PyTorch + SMP de Elvis. Es trabajo del equipo y aporta valor comparativo (sus metricas Dice 0.88 binario coinciden con el paper IEEE). Convencion de nombre: `0Nb_*.ipynb` marca "variante de 0N" — el principal `02_training_experiments.ipynb` (PyTorch, 5 modelos) sigue siendo el que alimenta el deploy. Cero etiquetado por persona (decision de Elvis: presentar como trabajo equipo). Tamano 4.8 MB aceptado as-is; si el repo se siente lento en post-mortem, nbstripout reduce ~80%. |
| 2026-05-22 (Ciclo 6.0) | `docs/SUSTENTACION_GUIA.md` como artefacto operativo para la defensa oral | Mas valioso del ciclo. Documento de ~476 lineas en 12 secciones que une informe IEEE + repo + deploy: resumen ejecutivo, equipo, mapping rubrica/paper/app, **metricas explicadas (CRITICO para Q&A sobre el gap Dice binario 0.88 vs multiclass 0.34)**, demo paso-a-paso para 5 min de pantalla compartida, narrativa 10 min con timing, 10 Q&A anticipadas con respuestas concretas, cheat sheet de numeros, plan B si algo falla en vivo. Elvis lo abre en el browser mientras presenta y sigue punto por punto. Markdown (no PDF) para editar last-minute si surge feedback en el dry-run. |
| 2026-05-22 (Ciclo 6.1) | Reemplazar `_curve_direction` midpoint-slope (Ciclo 5.10) por algoritmo chord signed-area | El fix del Ciclo 5.10 fue minimum-viable-fix: swap de un ternario basado en evidencia de UN solo caso (S_158, validado por la medica colaboradora). Tras la sustentacion, la misma medica reporto con 5 capturas que la lateralidad seguia invertida en MUCHOS casos — en particular, S-shapes que reportaban ambas curvas con la misma convexidad (anatomicamente imposible por definicion del IP). Diagnostico de raiz: `dx_dy[mid_idx]` no es invariante a la asimetria temporal de la curva — el slope pasa por cero EN EL APEX, no en el midpoint geometrico. Algoritmo nuevo: signo del signed area entre la curva y la chord IPa-IPb (cross product 2D normalizado). Geometricamente correcto, garantiza opposing convexity en S-shapes. Sweep baseline: 5/7 S-shapes violaban el principio; post-fix: 6/7 cumplen + 3/3 contra GT oficial. |
| 2026-05-22 (Ciclo 6.1) | Cambio de firma de `_curve_direction` (`dx_dy` -> `x_eval, y_eval`) sin rompimiento externo | El algoritmo nuevo necesita las coordenadas del spline, no solo la primera derivada. La firma cambia de `(dx_dy, ip_a, ip_b)` a `(x_eval, y_eval, ip_a, ip_b, neutral_threshold_px2=50.0)`. Verifique con grep que el unico caller esta en `_cobb_from_binary_single_pass` linea 274 del mismo modulo, donde `x_eval` y `y_eval` ya estan en scope (lineas 221-222). Cambio completamente contained — `i18n.py`, `app.py`, `inference.py`, `visualize.py` no se tocan. Los strings `convex_right`/`convex_left` se reutilizan: solo cambia su significado algoritmico, no su mapping ni su texto. |
| 2026-05-22 (Ciclo 6.1) | Tests anchored al ground truth oficial de MaIA (`tests/test_cobb_laterality_real.py`) | El Ciclo 5.10 cerro sobre evidencia de 1 caso y no agrego test pinneado contra dataset real — eso fue parte de por que el bug residual paso desapercibido hasta post-sustentacion. Decision: agregar tests que cargan `RadiographMetrics/curves_csv/curve_*.csv` + `metrics_json/metrics_*.json` directamente, calculan `expected_direction = "right" if apex_x < csvl_x else "left"` (regla del espejo en AP), y validan contra `_curve_direction`. Gated por `skipif(not dataset_present)` para no romper CI si el dataset no esta montado. Cobre S_158 (pivot 5.10) y S_22 (caso flagged por el sweep baseline). Solo principal porque el dataset oficial no anota secundarias — secundarias se cubren por el canary sintetico de S-shape. |
| 2026-05-22 (Ciclo 6.1) | `scripts/sweep_laterality.py` como herramienta reutilizable para futuros ciclos | El sweep visual con tabla MD es la unica forma de captar regresiones cualitativas que pytest no puede ver (e.g., "las dos curvas de una S-shape reportan opposing laterality"). Diseñe el script para que sea reusable: lista de casos parametrizable, paths del dataset y checkpoints overrideable via flags, output a markdown + CSV. Funciona desde cualquier worktree (sys.path insert del repo root) y el resolver del MAIA dataset busca arriba en los ancestors. Se commitea en el worktree del 6.1 pero queda disponible para 6.2+. Outputs van a `outputs/` (gitignored) por tipo de ciclo. |
