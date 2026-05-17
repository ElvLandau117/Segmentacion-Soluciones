"""
upload_weights.py — Subir pesos del proyecto a Hugging Face Hub.

Reemplazo del antiguo flujo de OneDrive. El repo de HF almacena solo los
archivos .pth necesarios para inferencia. El container los descarga al boot.

Uso típico (Ciclo 4 — primera subida):

    # 1. (una sola vez) Autenticarse en HF
    huggingface-cli login

    # 2. Subir el modelo ganador (DeepLabV3+ multiclase) y el binario
    python scripts/upload_weights.py \\
        --repo elvis/spine-checkpoints \\
        --file checkpoints/deeplabv3plus_resnet50_multiclass_best.pth

    python scripts/upload_weights.py \\
        --repo elvis/spine-checkpoints \\
        --file checkpoints/unet_resnet50_binary_best.pth

Si el repo no existe en HF, se crea automaticamente (tipo "model").
Usa --private si los pesos no deben ser publicos.

Ver: docs/HUGGINGFACE_SETUP.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sube un checkpoint .pth a un repo de Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("HF_REPO_ID"),
        help="ID del repo en HF (ej: 'elvis/spine-checkpoints'). "
             "Por defecto lee la variable de entorno HF_REPO_ID.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Ruta local al checkpoint .pth a subir.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nombre destino en el repo (default: el basename del archivo local).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crear/mantener el repo como privado (requiere HF_TOKEN en el container).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Mensaje del commit en HF Hub (default: 'Upload <name>').",
    )
    args = parser.parse_args()

    # Validaciones tempranas (boundary check)
    if not args.repo:
        print(
            "ERROR: falta --repo o la variable de entorno HF_REPO_ID.\n"
            "Ejemplo: --repo elvis/spine-checkpoints",
            file=sys.stderr,
        )
        return 2

    if not args.file.exists():
        print(f"ERROR: el archivo no existe: {args.file}", file=sys.stderr)
        return 2

    if args.file.suffix != ".pth":
        print(
            f"AVISO: extension inesperada '{args.file.suffix}'. "
            "Se esperaba '.pth'. Continuando de todos modos.",
            file=sys.stderr,
        )

    try:
        from huggingface_hub import HfApi, create_repo
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        print(
            "ERROR: huggingface_hub no esta instalado.\n"
            "Instala con: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 3

    api = HfApi()
    path_in_repo = args.name or args.file.name
    commit_msg = args.commit_message or f"Upload {path_in_repo}"

    # Crear repo si no existe (idempotente con exist_ok=True)
    try:
        api.repo_info(repo_id=args.repo, repo_type="model")
        print(f"[info] Repo existente: {args.repo}")
    except RepositoryNotFoundError:
        print(f"[info] Creando repo: {args.repo} (private={args.private})")
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

    # Subir archivo
    size_mb = args.file.stat().st_size / (1024 * 1024)
    print(f"[info] Subiendo {args.file} ({size_mb:.1f} MB) -> {args.repo}/{path_in_repo}")

    api.upload_file(
        path_or_fileobj=str(args.file),
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        repo_type="model",
        commit_message=commit_msg,
    )

    print(f"[ok] Subido. URL: https://huggingface.co/{args.repo}/blob/main/{path_in_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
