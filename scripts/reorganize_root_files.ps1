# =============================================================================
# reorganize_root_files.ps1 — Unidad 4.1 del Ciclo 4 (Despliegue)
#
# Mueve archivos sueltos de la raíz del proyecto a sus carpetas temáticas.
# Idempotente: si un archivo ya fue movido, lo saltea sin fallar.
#
# Uso:
#   cd "C:\Users\User\Desktop\otro\Tercer Semestre\Trabajo final"
#   powershell -ExecutionPolicy Bypass -File scripts\reorganize_root_files.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
Set-Location $root

Write-Host "Reorganizando archivos en: $root" -ForegroundColor Cyan
Write-Host ""

# Helper: mueve si existe y reporta
function Move-IfExists {
    param([string]$src, [string]$dst)
    if (Test-Path $src) {
        $dstDir = Split-Path $dst -Parent
        if (-not (Test-Path $dstDir)) {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        }
        Move-Item -Path $src -Destination $dst -Force
        Write-Host "[OK]   $src -> $dst" -ForegroundColor Green
    } else {
        Write-Host "[skip] $src (no existe)" -ForegroundColor DarkGray
    }
}

# === requisitos_universidad/ ===
Move-IfExists "Entregable y criterios de evaluación _ Coursera.pdf" `
              "requisitos_universidad\Entregable y criterios de evaluacion _ Coursera.pdf"
Move-IfExists "recomendacion despliegue.pdf" `
              "requisitos_universidad\recomendacion despliegue.pdf"

# === docs/metodologia/ ===
Move-IfExists "The Six Pillars of Spec-Driven Work - by Leonardo Gonzalez.pdf" `
              "docs\metodologia\The Six Pillars of Spec-Driven Work - by Leonardo Gonzalez.pdf"
Move-IfExists "From Spec-Driven Work to Work Orchestration.pdf" `
              "docs\metodologia\From Spec-Driven Work to Work Orchestration.pdf"
Move-IfExists "orchestantion agentic.txt" `
              "docs\metodologia\orchestration_agentic_transcript.txt"

# === notebooks/ ===
Move-IfExists "Laboratorio_sumativo_escoliosis_.ipynb" `
              "notebooks\Laboratorio_sumativo_escoliosis_.ipynb"

# === archive/ ===
Move-IfExists "paquete_equipo_onedrive.zip" `
              "archive\paquete_equipo_onedrive.zip"

# Paper del proyecto (referencia academica del semestre anterior)
Move-IfExists "Segmentación_automática_de_columna_vertebral_y_vértebras_en_radiografías_de_pacientes_sanos_y_con_es.pdf" `
              "docs\metodologia\Segmentacion_automatica_columna_vertebral_paper_referencia.pdf"

Write-Host ""
Write-Host "Reorganizacion completada." -ForegroundColor Cyan
Write-Host "Verifica con: git status" -ForegroundColor Yellow
