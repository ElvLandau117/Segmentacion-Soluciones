# AGENTS.md — Segmentacion Vertebral Multiclase para Diagnostico de Escoliosis

> **Spec-Driven Work (Pilar 6):** Artefacto persistente del proyecto.
> Cada ciclo lo actualiza. Todo nuevo chat/agente DEBE leerlo primero.
> Ultima actualizacion: 2026-04-13 | Ciclo: 2

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

### Ciclo 1 (completado)
- [x] Infraestructura completa: paquete Python modular
- [x] Pipeline de datos: augmentation, splits, 3 esquemas de clases
- [x] 5 modelos configurados en config.py
- [x] Trainer con MLflow + AMP + early stopping
- [x] Evaluacion: Dice, IoU, per-class, visualizaciones
- [x] Cobb angle: 2 metodos (binario + multiclase)
- [x] Explicabilidad: Grad-CAM + confianza + panel clinico
- [x] Deploy: Gradio app con explicabilidad + Dockerfile
- [x] Notebook maestro para equipo (carga pesos, corre en CPU)

### Ciclo 2 (en progreso) — ENFOQUE: multiclase 24 vertebras
- [x] AGENTS.md creado
- [x] Git inicializado con .gitignore
- [ ] Modelo 1: unet_mit_b3 multiclase -> ENTRENANDO
- [ ] Modelo 2: manet_mit_b5 multiclase -> pendiente
- [ ] Modelo 3: unet_resnet50 multiclase -> pendiente
- [ ] Modelo 4: unet_efficientnet_b4 multiclase -> pendiente
- [ ] Modelo 5: deeplabv3plus_resnet50 multiclase -> pendiente
- [ ] Evaluacion comparativa final
- [ ] Calculo de Cobb con ground truth
- [ ] Paneles de explicabilidad

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

### CICLO 3 COMPLETADO

- [x] AGENTS.md creado (persistencia spec-driven)
- [x] Git inicializado con .gitignore
- [x] Los 5 modelos multiclase entrenados
- [x] Evaluacion comparativa completada
- [x] Pesos exportados inference-only (838 MB total, 66% reduccion)
- [x] Paquete para equipo (OneDrive): paquete_equipo_onedrive.zip (776 MB)
- [x] Documentacion completa (INSTRUCCIONES_EQUIPO.md, RESULTADOS.md)
- [x] Notebook final (03_informe_final.ipynb) estilo semestre pasado
- [x] README.md actualizado con resultados finales

---

## 6. Configuracion de Entrenamiento

```python
# config.py - parametros actuales
batch_size = 8          # Usa ~8-10 GB de 16 GB VRAM
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

1. **Lee AGENTS.md completo** antes de hacer cualquier cosa
2. El proyecto esta IMPLEMENTADO — no crear archivos desde cero
3. Revisa `config.py` para modelos y rutas
4. Revisa `checkpoints/` para modelos ya entrenados
5. Seccion 5 = donde continuar
6. Seccion 4 = problemas del dominio (no repetir errores)
7. **Al terminar tu ciclo: ACTUALIZA este archivo** (estado, metricas, decisiones nuevas)

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
