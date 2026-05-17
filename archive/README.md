# Archive — artefactos legacy

Esta carpeta contiene **artefactos viejos que ya no se usan activamente** pero
se preservan por trazabilidad histórica.

Toda la carpeta está en `.gitignore`: **nada de aquí se commitea**. Vive solo
en el filesystem local del equipo.

## Contenido típico

- `paquete_equipo_onedrive.zip` — paquete antiguo de distribución de pesos via OneDrive.
  Reemplazado en Ciclo 4 por hosting en Hugging Face Hub.
- Backups manuales antes de refactors grandes.
- Versiones viejas de notebooks o scripts antes de cambios irreversibles.

## Regla

Si algo cumple **todas** estas condiciones, va aquí:
1. Ya no se usa en flujo activo.
2. Borrarlo sería irreversible y podría ser útil consultarlo en el futuro.
3. No vale la pena versionarlo en git (pesado o sensible).

Si no cumple las 3, **bórralo** o **inclúyelo en git** en la carpeta correcta.
