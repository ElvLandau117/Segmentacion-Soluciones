# Prompt para el Próximo Chat (Ciclo 4 — Despliegue)

> **Cómo usar este archivo:**
> 1. Copia el bloque de texto de abajo
> 2. Pega la nueva rúbrica de la universidad donde dice `[PEGAR AQUÍ LA RÚBRICA]`
> 3. Pega todo en el primer mensaje de un nuevo chat con Claude Code

---

## 📋 PROMPT PARA COPIAR

```text
Hola Claude. Soy Elvis Hernandez. Continuamos el proyecto de Segmentacion Vertebral
para diagnostico de escoliosis (Maestria en IA, Universidad de los Andes).

=== INSTRUCCIONES DE ONBOARDING (HAZ ESTO PRIMERO) ===

Lee estos archivos en este orden, son la memoria del proyecto:

1. C:\Users\User\Desktop\otro\Tercer Semestre\Trabajo final\AGENTS.md
   (memoria persistente, decisiones arquitectonicas, problemas conocidos)

2. C:\Users\User\Desktop\otro\Tercer Semestre\Trabajo final\WORKFLOW.md
   (reglas del repositorio, convenciones de commits, descomposicion del trabajo)

3. C:\Users\User\Desktop\otro\Tercer Semestre\Trabajo final\docs\RUTA_LECTURA.md
   (te confirma el orden, dale un vistazo)

4. C:\Users\User\Desktop\otro\Tercer Semestre\Trabajo final\docs\CICLO_3_ARTEFACTOS.md
   (resultados completos del ciclo anterior: 5 modelos, metricas finales)

5. C:\Users\User\Desktop\otro\Tercer Semestre\Trabajo final\docs\CICLO_4_DESPLIEGUE_BRIEF.md
   (spec inicial del ciclo actual: desplegar la app)

=== METODOLOGIA QUE SEGUIMOS ===

El proyecto sigue Spec-Driven Work + Work Orchestration
(Leonardo Gonzalez 2025-2026). En la practica:

- Cada ciclo de trabajo cierra con artefactos en docs/CICLO_N_ARTEFACTOS.md
- El siguiente ciclo lee ese artefacto como input
- AGENTS.md siempre se actualiza al cerrar un ciclo
- Cada commit es una unidad de trabajo coherente y verificable
- Pesos van por OneDrive (paquete_equipo_onedrive.zip), codigo va en GitHub

=== TAREA ACTUAL ===

Estamos arrancando el CICLO 4: DESPLIEGUE.

Objetivo: desplegar la app web Gradio (con explicabilidad + Cobb) en
un servidor Hetzner que ya tengo, accesible publicamente.

La universidad nos compartio una NUEVA RUBRICA DE EVALUACION para esta
entrega. La pego abajo. Por favor:

1. Lee primero los 5 archivos del onboarding
2. Lee la rubrica
3. Actualiza docs/CICLO_4_DESPLIEGUE_BRIEF.md alineando el plan de tareas
   con los criterios y pesos de la rubrica
4. Proponme un plan en formato de plan-mode antes de ejecutar nada
5. Cuando apruebe el plan, arrancamos con la primera unidad de trabajo

=== NUEVA RUBRICA DE EVALUACION ===

[PEGAR AQUI LA RUBRICA QUE COMPARTIO LA UNIVERSIDAD]

=== CONTEXTO ADICIONAL ===

- Estoy en Windows con WSL/Git Bash, Python 3.13 en C:/Python313/
- GPU: NVIDIA RTX 4060 Ti 16GB (uso bs=12)
- Tengo un server Hetzner con Docker instalado
- Quiero que el codigo limpio quede en GitHub
- Quiero que los pesos sigan compartiendose por OneDrive (paquete_equipo_onedrive.zip)
- No quiero co-autoria de IA en los commits (solo mi nombre: Elvis Hernandez)
- Sigue la metodologia spec-driven: descomponer el trabajo, verificar antes de commit,
  actualizar AGENTS.md al cerrar el ciclo, crear docs/CICLO_4_ARTEFACTOS.md al final

Empezamos.
```

---

## 🔍 Qué hace este prompt

1. **Onboarding automático:** Claude lee 5 archivos clave antes de hacer nada
2. **Contexto metodológico:** sabe que aplicamos spec-driven work
3. **Tarea clara:** desplegar la app, con rúbrica como guía
4. **Reglas explícitas:** sin co-autoría IA, pesos por OneDrive, código en Git
5. **Espera aprobación:** te muestra un plan antes de ejecutar

---

## ✅ Checklist antes de pegar el prompt

- [ ] Tienes la rúbrica oficial de la universidad
- [ ] La pegaste en `[PEGAR AQUI LA RUBRICA]`
- [ ] Verificaste que los archivos referenciados existen en las rutas dadas
- [ ] El nuevo chat está en plan mode (recomendado para que no haga cambios sin tu OK)

---

## 📝 Notas adicionales

### Si Claude no encuentra los archivos
Verifica las rutas absolutas. En Windows con bash usa `/c/Users/User/...` o usa
las rutas absolutas con backslashes que ya están en el prompt.

### Si el chat no tiene GPU disponible
Tampoco importa para el deployment. Las pruebas de inferencia se hacen en CPU.

### Si la rúbrica es muy larga
Está bien. Pégala completa. Claude la procesará y la usará para alinear el plan.

### Después del próximo chat
Al cerrar el Ciclo 4, deberá actualizar:
- `AGENTS.md` (estado actual, métricas de deployment, URL pública)
- `docs/CICLO_4_ARTEFACTOS.md` (artefacto formal del ciclo)
- `docs/CICLO_5_<tema>_BRIEF.md` (si hay siguiente ciclo)
- `docs/PROMPT_PROXIMO_CHAT.md` (este archivo, para el chat después del próximo)
