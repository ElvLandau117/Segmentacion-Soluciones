"""
Exporta checkpoints completos a versiones reducidas solo para inferencia.
Remueve: optimizer_state_dict, scheduler_state_dict (no necesarios para cargar modelo).
Resultado: archivos ~3x mas pequenos.

Uso: python scripts/export_weights.py
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"
EXPORT_DIR = Path(__file__).parent.parent / "checkpoints_inference"
EXPORT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("EXPORTANDO CHECKPOINTS PARA INFERENCIA (sin optimizer/scheduler)")
    print("=" * 70)

    # Solo los multiclase (son los que se comparten al equipo)
    multiclass_ckpts = sorted(CHECKPOINTS_DIR.glob("*_multiclass_best.pth"))

    if not multiclass_ckpts:
        print("No se encontraron checkpoints multiclase. Verificar carpeta 'checkpoints/'")
        return

    total_orig, total_new = 0, 0

    for ckpt_path in multiclass_ckpts:
        print(f"\nProcesando: {ckpt_path.name}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        original_size = ckpt_path.stat().st_size / 1024**2

        # Solo lo necesario para inferencia
        inference_ckpt = {
            "model_state_dict": checkpoint["model_state_dict"],
            "model_name": checkpoint.get("model_name", "unknown"),
            "task": checkpoint.get("task", "multiclass"),
            "num_classes": checkpoint.get("num_classes", 24),
            "best_val_dice": float(checkpoint.get("best_val_dice", 0.0)),
            "epoch": int(checkpoint.get("epoch", 0)),
        }

        export_path = EXPORT_DIR / ckpt_path.name
        torch.save(inference_ckpt, export_path)
        new_size = export_path.stat().st_size / 1024**2

        reduction = (1 - new_size / original_size) * 100
        print(f"  Original: {original_size:.1f} MB")
        print(f"  Export:   {new_size:.1f} MB")
        print(f"  Reduccion: {reduction:.1f}%")

        total_orig += original_size
        total_new += new_size

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_orig:.1f} MB -> {total_new:.1f} MB (reduccion {(1-total_new/total_orig)*100:.1f}%)")
    print(f"Archivos exportados en: {EXPORT_DIR}")
    print('=' * 70)


if __name__ == "__main__":
    main()
