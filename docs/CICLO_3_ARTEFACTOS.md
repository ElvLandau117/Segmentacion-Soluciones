# Ciclo 3 — Artefacto de Salida

> **Fecha de cierre:** 2026-04-13
> **Estado:** ✅ COMPLETADO
> **Próximo ciclo:** [`CICLO_4_DESPLIEGUE_BRIEF.md`](CICLO_4_DESPLIEGUE_BRIEF.md)

---

## Resumen ejecutivo

Ciclo dedicado a **completar los 5 modelos multiclase**, **evaluarlos comparativamente**, y **preparar el paquete de pesos** para que el equipo pueda trabajar sin GPU.

**Resultado más importante:** **DeepLabV3+ ganó con Test Dice=0.3378**, superando a los transformers (resultado contrario a la hipótesis inicial).

---

## 1. Modelos entrenados

| Ranking | Modelo | Paradigma | Test Dice | Test IoU | PixAcc | Parámetros |
|---------|--------|-----------|-----------|----------|--------|------------|
| 🥇 | **DeepLabV3+ + ResNet50** | CNN multi-escala (ASPP) | **0.3378** | **0.2556** | **0.9596** | 26.7M |
| 🥈 | MAnet + MiT-B5 | Transformer + atención dual | 0.3271 | 0.2383 | 0.9594 | 92.2M |
| 🥉 | U-Net + MiT-B3 | Transformer (SegFormer) | 0.3157 | 0.2323 | 0.9578 | 47.4M |
| 4° | U-Net + ResNet50 | CNN clásica | 0.2691 | 0.1883 | 0.9541 | 32.5M |
| 5° | U-Net + EfficientNet-B4 | CNN eficiente | 0.2189 | 0.1542 | 0.9548 | 20.2M |

### Configuración común
- Image size: 512×512
- Batch size: 12 (usa ~12 GB VRAM)
- Optimizer: AdamW (encoder_lr=1e-5, decoder_lr=1e-4)
- Loss: Weighted Cross-Entropy + Generalized Dice
- Max epochs: 150 (early stopping patience=20)
- Mixed precision (AMP) habilitado
- GPU: NVIDIA RTX 4060 Ti 16GB

---

## 2. Ángulo de Cobb (evaluación en casos de escoliosis)

| Modelo | Método | MAE (°) | Correlación Pearson |
|--------|--------|---------|---------------------|
| U-Net + EfficientNet-B4 | Binario (skeleton) | **23.0** | **0.66** |
| U-Net + ResNet50 | Binario (skeleton) | 25.5 | 0.56 |
| U-Net + EfficientNet-B4 | Multiclase (endplate) | 26.8 | 0.20 |
| U-Net + MiT-B3 | Multiclase (endplate) | 28.2 | 0.27 |
| U-Net + ResNet50 | Multiclase (endplate) | 39.4 | -0.12 |
| MAnet + MiT-B5 | Multiclase (endplate) | 42.0 | -0.20 |
| DeepLabV3+ + ResNet50 | Multiclase (endplate) | 45.4 | -0.20 |

**Observación clave:** El método binario (basado en esqueletización) da **mejor MAE de Cobb** que el método multiclase. Razón: errores de identificación de vértebras individuales se acumulan al calcular el ángulo desde placas vertebrales.

**Implicación para el deploy:** Usar **binario para Cobb** + **multiclase para visualización** = mejor de ambos mundos.

---

## 3. Per-class Dice del mejor modelo (DeepLabV3+)

### Cervicales (más difíciles, ~0.001% píxeles cada una)
| Vértebra | Dice |
|----------|------|
| C7 | 0.6880 ✓ |
| C6 | 0.6098 ✓ |
| C5 | 0.5601 ✓ |
| C4 | 0.1074 ⚠️ |
| C3 | 0.0000 ❌ NO DETECTADA |

### Torácicas
| Vértebra | Dice |
|----------|------|
| T1 | 0.6595 ✓ |
| T2 | 0.5108 ✓ |
| T3 | 0.3545 |
| T4-T11 | 0.18-0.33 |
| T12 | 0.2498 |

### Lumbares (más fáciles, ~0.13% píxeles cada una)
| Vértebra | Dice |
|----------|------|
| L1 | 0.2686 |
| L2 | 0.3311 |
| L3 | 0.3724 |
| L4 | 0.3907 |
| L5 | 0.4140 ✓ |

---

## 4. Hallazgos clave

### ✅ DeepLabV3+ superó a los transformers
- Hipótesis inicial: transformers ganarían por self-attention global
- Realidad: ASPP (convoluciones atrous multi-escala) fue más efectivo
- Razón: vértebras varían mucho en tamaño (cervicales pequeñas vs lumbares grandes), y ASPP captura esto explícitamente con dilatación a diferentes rates
- Implicación: **CNN no está obsoleta** para segmentación médica con datasets pequeños

### ⚠️ Dataset muy pequeño para transformers puros
- 174 imágenes de train + 24 clases = pocos ejemplos por clase
- Self-attention necesita ~10x más datos para aprender bien
- Los transformers aun así rindieron 2° y 3° — no fue catastrófico

### ⚠️ Vértebra C3 imposible de detectar
- Frecuencia: 0.0006% del dataset (literalmente 269 píxeles entre 174 imágenes)
- Peso máximo (10.0) no fue suficiente
- Mitigación posible: dataset augmentation específico para cervicales

### 🔄 Trade-off precisión vs deployment
- MAnet+MiT-B5 (mejor Dice): 352 MB → servidor
- EfficientNet-B4 (Dice más bajo): 78 MB → tablet, edge
- Estrategia: deploy en 2 niveles según contexto

---

## 5. Recursos disponibles (para el Ciclo 4)

### Código
- Paquete Python modular: `spine_segmentation/`
- Scripts de entrenamiento/evaluación: `scripts/`
- App web Gradio: `spine_segmentation/deployment/app.py`
- Dockerfile (CPU inference): `Dockerfile`

### Modelos (no en git, OneDrive)
- Checkpoints originales (con optimizer): `checkpoints/` (~2.5 GB)
- Checkpoints inference-only: `checkpoints_inference/` (~838 MB)
- Paquete listo para OneDrive: `paquete_equipo_onedrive.zip` (~776 MB)

### Documentación
- `AGENTS.md` — memoria persistente del proyecto
- `WORKFLOW.md` — policy del repositorio
- `README.md` — overview general
- `notebooks/03_informe_final.ipynb` — notebook ejecutable end-to-end
- `paquete_equipo/INSTRUCCIONES_EQUIPO.md` — guía para compañeros
- `paquete_equipo/RESULTADOS.md` — análisis completo

### Outputs generados
- `outputs/model_comparison.csv` — tabla comparativa
- `outputs/per_class_dice_*.png` — gráficas por vértebra
- `outputs/cobb_*.png` — Bland-Altman plots

### Git
- Repo: https://github.com/ElvLandau117/Segmentacion-Soluciones
- Commits del ciclo 3:
  - `d4e7d56` — feat: initial project structure
  - `90af7a3` — feat: complete 5 multiclass models + team package

---

## 6. Limitaciones a abordar en Ciclo 4 (o futuros)

1. **Dice promedio bajo (~0.30)** — académicamente válido, insuficiente para uso clínico real
2. **C3 no detectada** — clase con desbalance extremo
3. **Cobb angle aún impreciso** (MAE ~25°) — necesita refinamiento post-procesado
4. **Test set pequeño (38 imágenes)** — alta variabilidad en métricas reportadas
5. **No hay validación inter-observador** con radiólogos reales

---

## 7. Métricas de proceso del ciclo

- **Duración del ciclo:** ~1 día (compresión inusual, normalmente 1-2 semanas)
- **Commits realizados:** 2
- **Modelos entrenados:** 5 multiclase (+ 2 binarios legacy del ciclo 2)
- **VRAM utilizada:** ~12 GB/16 GB con batch_size=12
- **Tokens consumidos en chat con Claude:** ~200K (estimación)

---

## 8. Decisiones tomadas en este ciclo

| Decisión | Razón |
|----------|-------|
| Subir batch_size 8 → 12 | Solo se usaban 8 GB de 16 GB de VRAM |
| Re-entrenar EfficientNet-B4 multi | El primer intento quedó atascado en Dice=0.10 |
| Solo multiclase, NO binarios | El semestre anterior ya cubrió binario; enfoque en vértebras individuales |
| Exportar checkpoints inference-only | Reducción 66% de tamaño (de 2.5 GB a 838 MB) |
| Compartir vía OneDrive, no Git | Pesos pesados no encajan en versionado de código |
| Adoptar WORKFLOW.md + docs/ | Aplicar metodología spec-driven (Pilar 6) |

---

## 9. Lo que NO se hizo en este ciclo (deferred)

- Despliegue real en servidor (Hetzner) — **se difiere al Ciclo 4**
- Quantización INT8 para tablet
- Ensemble de los 5 modelos
- Refinamiento del cálculo de Cobb angle
- Validación clínica con radiólogos
- Artículo IEEE/ACM — depende del despliegue

---

## 10. Handoff al Ciclo 4

**Lo que el Ciclo 4 puede asumir como dado:**
- 5 modelos entrenados con métricas registradas
- Pesos inference-only disponibles en `checkpoints_inference/`
- Paquete `paquete_equipo_onedrive.zip` listo para compartir
- Notebook ejecutable end-to-end (sin GPU)
- App Gradio funcional en local (`python -m spine_segmentation.deployment.app`)
- Dockerfile básico (CPU inference)

**Lo que el Ciclo 4 debe producir:**
- App web desplegada en servidor accesible públicamente
- Acceptance criteria de despliegue definidos (con rúbrica de evaluación)
- Documentación de despliegue (`docs/DEPLOYMENT.md`)
- `docs/CICLO_4_ARTEFACTOS.md` al cerrar

Ver [`CICLO_4_DESPLIEGUE_BRIEF.md`](CICLO_4_DESPLIEGUE_BRIEF.md) para el spec inicial.
