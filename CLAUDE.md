# Claude Code — Punto de entrada de contexto

> Este archivo lo lee automáticamente Claude Code al iniciar una sesión.
> Su único propósito es redirigir al artefacto público de memoria del proyecto.

**Antes de hacer cualquier cosa, lee `AGENTS.md`.**

→ [`AGENTS.md`](AGENTS.md) — memoria persistente del proyecto (estado, decisiones, métricas, historial)

Después, sigue la ruta de lectura formal:

→ [`docs/RUTA_LECTURA.md`](docs/RUTA_LECTURA.md)

## Por qué este archivo apunta a AGENTS.md

`CLAUDE.md` es la convención de Claude Code; `AGENTS.md` es la convención
[agents-md.io](https://agents-md.io) — un estándar legible para humanos y para
cualquier agente, no atado a una herramienta específica.

Mantenemos **una sola fuente de verdad** (`AGENTS.md`) para que:
- Los jurados y compañeros que no usen Claude Code puedan auditarla sin fricción.
- Los médicos que revisen el sistema entiendan el contexto sin tener que abrir un IDE.
- Cualquier agente futuro (Claude, Cursor, Aider, etc.) lea el mismo documento.

`CLAUDE.md` (este archivo) existe únicamente como puntero.
