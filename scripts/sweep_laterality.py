"""sweep_laterality.py - Ciclo 6.1 visual validation tool.

Runs ``SpineSegmentationPipeline.predict()`` on a curated list of cases
from the MaIA dataset and emits a markdown table with the convexity
direction reported for each detected curve. Designed for the human-in-
the-loop review that closes the laterality bug.

The script intentionally bypasses Gradio so a sweep over 12 cases takes
seconds instead of minutes. Output is meant to be diff-able against a
clinician annotation table ("expected vs reported").

Usage:
    python scripts/sweep_laterality.py
    python scripts/sweep_laterality.py --cases S_22,S_158,S_100
    python scripts/sweep_laterality.py --markdown outputs/sweep.md --csv outputs/sweep.csv

Run from any cwd. Checkpoints and dataset paths default to the repo
roots; pass --binary-ckpt / --multiclass-ckpt / --dataset to override.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

# Make ``spine_segmentation`` importable regardless of where the script
# lives (matters when the script is invoked from a worktree but data
# lives in the main checkout).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spine_segmentation.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    DATASET_ROOT,
    DEFAULT_BINARY_MODEL,
    DEFAULT_BINARY_WEIGHT,
    DEFAULT_MULTICLASS_MODEL,
    DEFAULT_MULTICLASS_WEIGHT,
)
from spine_segmentation.deployment.inference import SpineSegmentationPipeline  # noqa: E402

DEFAULT_CASES = [
    "N_1", "N_61",
    "S_21", "S_22", "S_45", "S_77", "S_100",
    "S_120", "S_130", "S_150", "S_158", "S_200",
]


def resolve_image(case: str, dataset_root: Path) -> Path:
    """Return the path to a case image, trying Scoliosis/ first then Normal/."""
    for subdir in ("Scoliosis", "Normal"):
        candidate = dataset_root / subdir / f"{case}.jpg"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image not found for case {case} under {dataset_root}/Scoliosis or /Normal"
    )


def format_curve(curve) -> str:
    """Format a single curve as 'NN.N deg dir (Vup-Vlo)' or '-' if absent."""
    if not curve:
        return "-"
    angle = curve.get("cobb_angle_deg")
    direction = curve.get("direction", "?")
    upper = curve.get("upper_vertebra") or "?"
    lower = curve.get("lower_vertebra") or "?"
    return f"{angle:.1f} deg {direction} ({upper}-{lower})"


def run_sweep(
    cases,
    pipeline: SpineSegmentationPipeline,
    dataset_root: Path,
):
    """Process each case and return a list of result rows."""
    rows = []
    for case in cases:
        try:
            img_path = resolve_image(case, dataset_root)
        except FileNotFoundError as err:
            rows.append({
                "case": case,
                "n_curves": "-",
                "principal": "-",
                "secondary": "-",
                "tilt_deg": "-",
                "partial": "-",
                "notes": f"ERROR: {err}",
            })
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            rows.append({
                "case": case,
                "n_curves": "-",
                "principal": "-",
                "secondary": "-",
                "tilt_deg": "-",
                "partial": "-",
                "notes": f"ERROR: cv2.imread returned None for {img_path}",
            })
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = pipeline.predict(image)

        cobb = result.get("cobb_binary") or {}
        curves = cobb.get("curves") or []
        principal = curves[0] if len(curves) >= 1 else None
        secondary = curves[1] if len(curves) >= 2 else None

        orientation = result.get("orientation_info") or {}
        tilt = orientation.get("tilt_deg")
        coverage = result.get("coverage_info") or {}
        partial = coverage.get("is_partial")

        rows.append({
            "case": case,
            "n_curves": len(curves),
            "principal": format_curve(principal),
            "secondary": format_curve(secondary),
            "tilt_deg": f"{tilt:.1f}" if isinstance(tilt, (int, float)) else "-",
            "partial": "Y" if partial else "N",
            "notes": "",
        })
    return rows


def render_markdown(rows) -> str:
    """Render rows as a markdown table string."""
    lines = [
        "| case | n_curves | principal (deg, dir, vertebrae) | secondary (deg, dir, vertebrae) | tilt | partial | notes |",
        "|---|:-:|---|---|:-:|:-:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['n_curves']} | {row['principal']} | "
            f"{row['secondary']} | {row['tilt_deg']} | {row['partial']} | {row['notes']} |"
        )
    return "\n".join(lines)


def write_csv(rows, path: Path) -> None:
    """Write rows to CSV at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case", "n_curves", "principal", "secondary",
                "tilt_deg", "partial", "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cycle 6.1 laterality sweep over 12 dataset cases.",
    )
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help="Comma-separated case names (default: 12 cycle-6.1 fixtures)",
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT),
        help="Dataset root (must contain Scoliosis/ and Normal/)",
    )
    parser.add_argument(
        "--binary-ckpt",
        default=str(CHECKPOINTS_DIR / DEFAULT_BINARY_WEIGHT),
        help="Path to the binary segmentation checkpoint (.pth)",
    )
    parser.add_argument(
        "--multiclass-ckpt",
        default=str(CHECKPOINTS_DIR / DEFAULT_MULTICLASS_WEIGHT),
        help="Path to the multiclass segmentation checkpoint (.pth)",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional output path for the markdown table",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output path for the CSV table",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not (dataset / "Scoliosis").exists():
        print(
            f"ERROR: dataset folder Scoliosis/ not found under {dataset}.\n"
            "Pass --dataset to point to MaIA_Scoliosis_Dataset/.",
            file=sys.stderr,
        )
        return 2

    binary_ckpt = Path(args.binary_ckpt)
    multiclass_ckpt = Path(args.multiclass_ckpt)
    if not binary_ckpt.exists() or not multiclass_ckpt.exists():
        print(
            "ERROR: checkpoint files not found.\n"
            f"  binary: {binary_ckpt} "
            f"({'OK' if binary_ckpt.exists() else 'MISSING'})\n"
            f"  multiclass: {multiclass_ckpt} "
            f"({'OK' if multiclass_ckpt.exists() else 'MISSING'})\n"
            "Download with: python scripts/upload_weights.py --download",
            file=sys.stderr,
        )
        return 2

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    print(f"Loading pipeline ({DEFAULT_BINARY_MODEL} + {DEFAULT_MULTICLASS_MODEL})...",
          file=sys.stderr)
    pipeline = SpineSegmentationPipeline(
        binary_checkpoint=str(binary_ckpt),
        multiclass_checkpoint=str(multiclass_ckpt),
        binary_model_name=DEFAULT_BINARY_MODEL,
        multiclass_model_name=DEFAULT_MULTICLASS_MODEL,
    )

    print(f"Sweeping {len(cases)} cases...", file=sys.stderr)
    rows = run_sweep(cases, pipeline, dataset)
    markdown = render_markdown(rows)
    print(markdown)

    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nMarkdown written to {out}", file=sys.stderr)

    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"CSV written to {Path(args.csv)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
