# AGENTS.md — Segmentacion Vertebral Multiclase para Diagnostico de Escoliosis

> **Spec-Driven Work (Pilar 6):** Artefacto persistente del proyecto.
> Cada ciclo lo actualiza. Todo nuevo chat/agente DEBE leerlo primero.
> Ultima actualizacion: 2026-05-19 | Ciclos: 1, 2, 3, 4, 5, 5.1, 5.2, 5.3, 5.4 ✅ COMPLETOS. Ciclo 6: pendiente brief.
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

### Ciclo 6 (proximo) — Refinamiento del modelo + entrega final
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
