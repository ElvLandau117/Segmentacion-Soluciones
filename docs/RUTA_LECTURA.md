# Ruta de Lectura Sugerida

> **Para nuevos colaboradores (humanos o agentes).**
> Inspirado en https://opensymphony.dev/es/resources/

Esta es la ruta más eficiente para entender el proyecto sin perder tiempo:

---

## 📖 01. Lee primero el README

**Archivo:** [`README.md`](../README.md)

Aquí obtienes:
- Descripción del problema (escoliosis, ángulo de Cobb)
- Cómo instalar
- Cómo usar (notebook + app web)
- Tabla de resultados finales

⏱ Tiempo estimado: **10 minutos**

---

## 🧠 02. Lee AGENTS.md para el contexto completo

**Archivo:** [`AGENTS.md`](../AGENTS.md)

Aquí obtienes:
- **Por qué** se tomó cada decisión arquitectónica
- **Problemas conocidos** del dominio (rotación vertebral, etc.)
- **Estado actual** de cada ciclo
- **Métricas detalladas** de los 5 modelos
- **Historial de decisiones** con fechas y razones

Este es el **artefacto de memoria persistente** del proyecto. Si vas a colaborar, este es el archivo más importante.

⏱ Tiempo estimado: **20 minutos**

---

## 📋 03. Lee WORKFLOW.md para las reglas del repo

**Archivo:** [`WORKFLOW.md`](../WORKFLOW.md)

Aquí obtienes:
- **Definición de "Done"** para cada tipo de cambio
- **Convenciones de commits**
- **Qué va en Git vs OneDrive** (importante para no commitear pesos)
- **Estructura de ciclos** (spec-driven workflow)
- **Reglas no negociables**

⏱ Tiempo estimado: **10 minutos**

---

## 🎯 04. Lee el artefacto del último ciclo cerrado

**Archivo:** [`docs/CICLO_5_ARTEFACTOS.md`](CICLO_5_ARTEFACTOS.md) (con addenda hasta 6.1)

Lee la última sección activa (al cierre del Ciclo 6.1, es la **sección 23**:
"Addendum 6.1 — Fix de lateralidad por chord signed-area"). Las secciones
anteriores (1-22) son la historia operativa de los ciclos 5.x y 6.0.

Aquí obtienes:
- **Cambio más reciente al algoritmo** y por qué (post-sustentación)
- **Tabla baseline vs post-fix** del sweep de 12 casos
- **Tests añadidos / actualizados** + commits del ciclo
- **Aprendizajes y limitaciones honestas**

Para resultados de entrenamiento, consulta [`docs/CICLO_3_ARTEFACTOS.md`](CICLO_3_ARTEFACTOS.md)
(modelos entrenados, métricas finales, DeepLabV3+ como ganador).

⏱ Tiempo estimado: **15 minutos**

---

## 🚀 05. Si vas a continuar el trabajo: lee el brief del próximo ciclo

**Archivo:** [`docs/CICLO_4_DESPLIEGUE_BRIEF.md`](CICLO_4_DESPLIEGUE_BRIEF.md)

Aquí obtienes:
- **Objetivo** del ciclo de despliegue
- **Spec inicial** del sistema a desplegar
- **Acceptance criteria**
- **Plan tentativo de tareas** (a refinar con rúbrica)

⏱ Tiempo estimado: **15 minutos**

---

## 💻 06. Para ejecutar el sistema

**Notebook:** [`notebooks/03_informe_final.ipynb`](../notebooks/03_informe_final.ipynb)

Notebook maestro que:
- Carga los 5 modelos pre-entrenados (necesita los `.pth` en `checkpoints/`)
- Evalúa en el test set
- Visualiza predicciones
- Calcula ángulos de Cobb
- Genera paneles de explicabilidad (Grad-CAM + confianza)
- **Funciona en CPU** (no requiere GPU)

**App web:**
```bash
python -m spine_segmentation.deployment.app
# Abrir http://localhost:7860
```

⏱ Tiempo estimado: **20 minutos** (instalación + ejecución)

---

## 🔄 07. Si vas a iniciar un nuevo chat con Claude

**Archivo:** [`docs/PROMPT_PROXIMO_CHAT.md`](PROMPT_PROXIMO_CHAT.md)

Contiene el prompt exacto a pegar al inicio del chat para que Claude:
1. Lea automáticamente los archivos clave
2. Cargue todo el contexto del proyecto
3. Esté listo para la siguiente unidad de trabajo

---

## Resumen visual de la ruta

```
  README ──► AGENTS ──► WORKFLOW ──► CICLO_3_ARTEFACTOS
   10m       20m         10m            10m
                                          │
                                          ▼
                                 CICLO_4_BRIEF (si vas a continuar)
                                       15m
                                          │
                                          ▼
                                  notebook 03 (ejecutar)
                                       20m
```

**Tiempo total de onboarding: ~85 minutos.**

Después de esto, conoces el proyecto al nivel de cualquier miembro del equipo.

---

## ¿Y los documentos de metodología?

Si quieres entender la **metodología** que sigue el proyecto (spec-driven work + work orchestration), lee:

- `The Six Pillars of Spec-Driven Work` (Gonzalez 2025)
- `From Spec-Driven Work to Work Orchestration` (Gonzalez 2026)
- `orchestantion agentic.txt` (transcripción del video curso)

Están en la raíz del proyecto (excluidos de git por tamaño/derechos). Solicítaselos al líder.
