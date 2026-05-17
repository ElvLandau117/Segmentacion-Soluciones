# WORKFLOW.md — Policy del Repositorio

> **Inspirado en OpenSymphony / Spec-Driven Work (Leonardo Gonzalez 2025-2026)**
> Este archivo define las reglas del repositorio. Toda contribución (humana o agente) debe seguirlas.

---

## 1. Definición de "Done"

Una unidad de trabajo se considera **completada** cuando cumple TODAS estas condiciones:

### Para cambios de código (`spine_segmentation/`, `scripts/`)
- [ ] El código corre sin errores (`python scripts/<nombre>.py` o `python -m spine_segmentation.<modulo>`)
- [ ] Pasa los warnings críticos de Python (no `DeprecationWarning` ni `RuntimeError`)
- [ ] Funciona en CPU si es para inferencia (compatibilidad con compañeros sin GPU)
- [ ] Tiene un docstring en cada función pública
- [ ] Está commiteado con mensaje siguiendo las convenciones (sección 4)

### Para cambios de modelo (`checkpoints/`)
- [ ] Existe el checkpoint `.pth` en `checkpoints/`
- [ ] El checkpoint tiene `model_state_dict`, `model_name`, `task`, `num_classes`, `best_val_dice`, `epoch`
- [ ] Las métricas del modelo están registradas en `AGENTS.md` sección "Métricas"
- [ ] El experimento está en MLflow (`mlruns/`)

### Para cambios de documentación (`*.md`, `docs/`)
- [ ] La información está actualizada (no hay datos viejos contradictorios)
- [ ] Los enlaces funcionan (rutas relativas correctas)
- [ ] Si se cierra un ciclo: existe el artefacto correspondiente en `docs/CICLO_N_ARTEFACTOS.md`

### Para cambios de configuración (`config.py`, `requirements.txt`)
- [ ] Probado con el cambio en al menos un script de entrenamiento o inferencia
- [ ] Razón del cambio registrada en `AGENTS.md` sección "Historial de decisiones"

---

## 2. Verificación obligatoria antes de commit

### Pre-commit (manual o agente)
1. **Lint visual** del código modificado (no hay imports sin usar, indentación consistente)
2. **Smoke test**: ejecutar el script/módulo modificado al menos 1 vez
3. **Sin secretos**: revisar que no haya tokens, credenciales o paths privados
4. **Sin pesos pesados**: verificar que `.pth`, `.h5`, dataset, etc. NO estén en `git status`

### Comando de verificación rápida
```bash
# Antes de cada commit
git status --short                          # ¿qué se va a commitear?
python -c "from spine_segmentation.config import *; print('OK')"  # ¿el config sigue valido?
```

---

## 3. Reglas de descomposición del trabajo

Aplicación del **Principio 2 de Autonomía** (descomponer):

- **Una unidad de trabajo = un cambio coherente** (no mezclar feature + fix + refactor en un commit)
- **Cada unidad debe completarse en <8 horas** de trabajo enfocado
- **Si una tarea toma más de 8h**, debe descomponerse en sub-tareas con sus propios commits
- **Un ciclo (semanal/quincenal) agrupa varias unidades** y produce un artefacto en `docs/`

### Ejemplo de descomposición correcta
```
Ciclo 4: Despliegue
├── Unidad 4.1: Crear Dockerfile multi-stage (4h)
├── Unidad 4.2: Configurar Hetzner + IP fija (3h)
├── Unidad 4.3: Configurar Nginx + SSL (3h)
├── Unidad 4.4: Smoke test end-to-end (2h)
└── Unidad 4.5: Documentar deployment + actualizar AGENTS.md (1h)
```

---

## 4. Convenciones de commits

Formato: `<tipo>(<scope>): <descripcion corta>`

### Tipos válidos
- `feat`: nueva funcionalidad
- `fix`: corrección de bug
- `docs`: cambios solo en documentación
- `refactor`: cambios de código que no agregan features ni arreglan bugs
- `train`: nuevo modelo entrenado o re-entrenado
- `eval`: evaluación/métricas/análisis
- `deploy`: cambios de infraestructura/despliegue
- `chore`: tareas de mantenimiento

### Scopes válidos
- `data`, `models`, `training`, `evaluation`, `deployment`, `docs`, `ci`, `repo`

### Ejemplos
```
feat(deployment): add Hetzner deploy script with nginx config
fix(data): correct mask interpolation to nearest-neighbor
train(models): retrain efficientnet_b4 multiclass with bs=12 (Dice=0.2189)
docs(workflow): add WORKFLOW.md with policy and conventions
deploy(infra): containerize app for portable cloud deployment
```

### Cuerpo del commit (cuando aplica)
```
feat(deployment): add docker-compose for full stack

- Includes Gradio app + Nginx reverse proxy + SSL termination
- Bind-mounts checkpoints/ for hot-swap without rebuild
- Health check endpoint at /health for monitoring

Closes: <issue-id> (si hay)
Refs: docs/CICLO_4_DESPLIEGUE_BRIEF.md
```

---

## 5. Qué va en Git vs. qué va en OneDrive

### En Git (commiteable)
- ✅ Código (`spine_segmentation/`, `scripts/`)
- ✅ Notebooks (`notebooks/*.ipynb`) — pero sin outputs grandes
- ✅ Documentación (`*.md`, `docs/`)
- ✅ Configuración (`requirements.txt`, `Dockerfile`, `.gitignore`)
- ✅ Splits reproducibles (`data_splits.json`)

### En OneDrive (compartir por separado)
- ❌ Pesos entrenados (`.pth` >5MB) → `paquete_equipo_onedrive.zip`
- ❌ Dataset original (`MaIA_Scoliosis_Dataset/`) → propiedad U. Andes, no redistribuir
- ❌ MLflow runs (`mlruns/`) → se regeneran
- ❌ Outputs (`outputs/`) → se regeneran

### Razón
GitHub limita a 100 MB por archivo y desincentiva blobs grandes. Los pesos cambian frecuentemente, no encajan en versionado de código.

---

## 6. Estructura de ciclos (Pilar 6 de Spec-Driven Work)

**Cada ciclo de trabajo:**

```
Inicio del ciclo N:
  Input = docs/CICLO_(N-1)_ARTEFACTOS.md + AGENTS.md
  
Durante el ciclo N:
  - Descomponer trabajo en unidades (<8h cada una)
  - Commit por unidad
  - Verificar antes de cada commit
  
Fin del ciclo N:
  Output = docs/CICLO_N_ARTEFACTOS.md (input del ciclo N+1)
  AGENTS.md actualizado con métricas/decisiones nuevas
  Commit + push
```

### Ciclos definidos hasta ahora
| Ciclo | Tema | Estado | Artefacto de salida |
|-------|------|--------|---------------------|
| 1 | Infraestructura inicial | ✅ Completado | (informal, en AGENTS.md) |
| 2 | Entrenamiento + transformers | ✅ Completado | (informal, en AGENTS.md) |
| 3 | 5 modelos + paquete equipo | ✅ Completado | `docs/CICLO_3_ARTEFACTOS.md` |
| 4 | Despliegue | 🔜 Próximo | `docs/CICLO_4_DESPLIEGUE_BRIEF.md` (spec inicial) |

---

## 7. Quién verifica qué

### Humano (Elvis)
- Aprueba arquitectura y decisiones estratégicas
- Verifica resultados clínicos (interpretación de Cobb, severidad)
- Hace el commit/push final de cada ciclo
- Define rúbricas y acceptance criteria de cada ciclo

### Equipo (compañeros)
- Verifica que el notebook corra en CPU sin problemas
- Revisa el informe escrito
- Aporta a la interpretación de resultados

### Agente (Claude Code u otro)
- Implementa la descomposición de tareas
- Genera código siguiendo las convenciones
- Actualiza AGENTS.md y artefactos al cerrar cada ciclo
- NUNCA hace push sin aprobación humana explícita
- NUNCA modifica WORKFLOW.md sin discusión previa

### CI (futuro, opcional)
- Validación automática de Dockerfile builds
- Smoke test del módulo de inferencia

---

## 8. Reglas no negociables

1. **NUNCA** commitear pesos `.pth` directamente al repo
2. **NUNCA** commitear el dataset `MaIA_Scoliosis_Dataset/`
3. **NUNCA** hacer `git push --force` a `main`
4. **SIEMPRE** actualizar `AGENTS.md` al cerrar un ciclo
5. **SIEMPRE** generar el `docs/CICLO_N_ARTEFACTOS.md` al cerrar un ciclo
6. **SIEMPRE** preguntar antes de eliminar checkpoints
7. **DISCLAIMER médico obligatorio** en toda interfaz de usuario (notebook, app)

---

## 9. Referencias

- Six Pillars of Spec-Driven Work — Leonardo Gonzalez (2025)
- From Spec-Driven Work to Work Orchestration — Leonardo Gonzalez (2026)
- OpenSymphony: https://opensymphony.dev
- AGENTS.md: https://agents-md.io (formato comunitario)
