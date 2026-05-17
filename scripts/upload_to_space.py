"""
upload_to_space.py - Subir archivos a un repo de tipo "space" en Hugging Face Hub.

Hermano de scripts/upload_weights.py. Se usa cuando NO queremos hacer git push
al Space (porque ya tiene historia LFS limpia o porque el cambio es minimo).
Sube uno o varios archivos en un solo commit atomico via HfApi.upload_file().

Uso tipico (Ciclo 4 - fix del bug de gradio-client):

    python scripts/upload_to_space.py \\
        --repo ElvLandau/spine-segmentation \\
        --file README.md \\
        --file requirements.txt \\
        --commit-message "fix(deploy): upgrade gradio to 5.50.0"

Si pasas un solo --file, el script hace un upload_file simple. Con varios,
agrupa todo en un solo commit usando create_commit (mas eficiente que N
commits separados y mantiene el rebuild del Space como un evento atomico).

Requiere HF_TOKEN con permiso write (o `huggingface-cli login` previo).
Ver: docs/HF_SPACES_SETUP.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sube archivos a un Space de Hugging Face Hub via HfApi.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("HF_SPACE_ID"),
        help="ID del Space en HF (ej: 'ElvLandau/spine-segmentation'). "
             "Por defecto lee la variable de entorno HF_SPACE_ID.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        required=True,
        help="Ruta local al archivo a subir. Se puede repetir para varios "
             "archivos en un solo commit atomico.",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        action="append",
        default=None,
        help="Nombre destino dentro del Space para cada --file (en el mismo "
             "orden). Si se omite, se usa el basename del archivo local.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Mensaje del commit en HF (default: 'Update <lista de archivos>').",
    )
    args = parser.parse_args()

    if not args.repo:
        print(
            "ERROR: falta --repo o la variable de entorno HF_SPACE_ID.\n"
            "Ejemplo: --repo ElvLandau/spine-segmentation",
            file=sys.stderr,
        )
        return 2

    for f in args.file:
        if not f.exists():
            print(f"ERROR: el archivo no existe: {f}", file=sys.stderr)
            return 2

    # Resolver path-in-repo paralelo a --file (o default a basename)
    if args.path_in_repo is not None:
        if len(args.path_in_repo) != len(args.file):
            print(
                f"ERROR: --path-in-repo se paso {len(args.path_in_repo)} veces "
                f"pero --file se paso {len(args.file)} veces. Deben coincidir.",
                file=sys.stderr,
            )
            return 2
        paths_in_repo = args.path_in_repo
    else:
        paths_in_repo = [f.name for f in args.file]

    commit_msg = args.commit_message or f"Update {', '.join(paths_in_repo)}"

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        print(
            "ERROR: huggingface_hub no esta instalado.\n"
            "Instala con: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 3

    api = HfApi()

    # Verificar que el Space exista (no se crea automaticamente desde aqui;
    # se asume que ya fue creado en https://huggingface.co/new-space).
    try:
        api.repo_info(repo_id=args.repo, repo_type="space")
    except RepositoryNotFoundError:
        print(
            f"ERROR: el Space {args.repo} no existe. Crealo primero en "
            "https://huggingface.co/new-space (este script no crea Spaces para "
            "evitar provisioning accidental).",
            file=sys.stderr,
        )
        return 4

    # Reportar lo que se va a subir
    total_mb = sum(f.stat().st_size for f in args.file) / (1024 * 1024)
    print(f"[info] Repo: {args.repo} (space)")
    for f, p in zip(args.file, paths_in_repo):
        size_kb = f.stat().st_size / 1024
        print(f"[info]   {f} ({size_kb:.1f} KB) -> {p}")
    print(f"[info] Total: {total_mb:.2f} MB en {len(args.file)} archivo(s)")
    print(f"[info] Commit: {commit_msg}")

    # Single file -> upload_file simple. Multi-file -> create_commit atomico.
    if len(args.file) == 1:
        api.upload_file(
            path_or_fileobj=str(args.file[0]),
            path_in_repo=paths_in_repo[0],
            repo_id=args.repo,
            repo_type="space",
            commit_message=commit_msg,
        )
    else:
        operations = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f))
            for f, p in zip(args.file, paths_in_repo)
        ]
        api.create_commit(
            repo_id=args.repo,
            repo_type="space",
            operations=operations,
            commit_message=commit_msg,
        )

    print(f"[ok] Subido. URL: https://huggingface.co/spaces/{args.repo}")
    print(f"[ok] El Space se va a re-construir automaticamente (2-3 min).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
