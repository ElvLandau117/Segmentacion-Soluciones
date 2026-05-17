# AGENTS.md — Segmentacion Vertebral Multiclase para Diagnostico de Escoliosis

> **Spec-Driven Work (Pilar 6):** Artefacto persistente del proyecto.
> Cada ciclo lo actualiza. Todo nuevo chat/agente DEBE leerlo primero.
> Ultima actualizacion: 2026-04-20 | Ciclos completados: 1, 2, 3 | Proximo ciclo: 4 (Despliegue)

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

### Ciclo 4 (proximo) — Despliegue
- [ ] Validar Dockerfile actual
- [ ] Optimizar tamano de imagen
- [ ] Preparar deploy en Hetzner
- [ ] Desplegar app web publicamente
- [ ] Configurar Nginx + SSL (opcional)
- [ ] Smoke test end-to-end
- [ ] Documentar deployment

Briefing detallado: [`docs/CICLO_4_DESPLIEGUE_BRIEF.md`](docs/CICLO_4_DESPLIEGUE_BRIEF.md)
Prompt para retomar: [`docs/PROMPT_PROXIMO_CHAT.md`](docs/PROMPT_PROXIMO_CHAT.md)

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
