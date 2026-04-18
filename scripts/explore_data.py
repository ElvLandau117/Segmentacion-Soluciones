"""
Exploratory Data Analysis (EDA) for the MaIA Scoliosis Dataset.
Generates statistics, distributions, and visualizations.

Usage: python scripts/explore_data.py
"""

import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from spine_segmentation.config import (
    DATASET_ROOT, DATASET_INDEX_CSV, RADIOGRAPH_METRICS_CSV,
    LABELS_DICT_JSON, OUTPUTS_DIR,
)
from spine_segmentation.data.class_mapping import get_class_names


def analyze_dataset():
    """Run complete EDA on the MaIA Scoliosis Dataset."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - MaIA Scoliosis Dataset")
    print("=" * 60)

    # Load metadata
    df = pd.read_csv(DATASET_INDEX_CSV)
    print(f"\nTotal samples: {len(df)}")
    print(f"Conditions: {df['split'].value_counts().to_dict()}")

    # 1. Image size analysis
    print("\n--- Image Size Analysis ---")
    heights, widths = [], []
    for _, row in df.iterrows():
        img_path = DATASET_ROOT / row["radiograph_path"]
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)

    print(f"Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
    print(f"Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"Aspect ratios: {np.mean(np.array(heights)/np.array(widths)):.2f} (H/W)")

    # Plot image size distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(heights, bins=20, color='steelblue', edgecolor='white')
    axes[0].set_title('Height Distribution', fontsize=12)
    axes[0].set_xlabel('Pixels')

    axes[1].hist(widths, bins=20, color='coral', edgecolor='white')
    axes[1].set_title('Width Distribution', fontsize=12)
    axes[1].set_xlabel('Pixels')

    aspects = np.array(heights) / np.array(widths)
    axes[2].hist(aspects, bins=20, color='mediumseagreen', edgecolor='white')
    axes[2].set_title('Aspect Ratio (H/W)', fontsize=12)

    plt.suptitle('Image Dimensions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "eda_image_sizes.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Class distribution analysis (multiclass masks)
    print("\n--- Multiclass Label Distribution ---")
    class_names = get_class_names("vertebrae_24")
    class_pixel_counts = Counter()
    class_image_counts = Counter()

    for _, row in df.iterrows():
        mask_path = DATASET_ROOT / row["multiclass_id_png"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        unique_classes = np.unique(mask)
        for c in unique_classes:
            class_pixel_counts[c] += (mask == c).sum()
            class_image_counts[c] += 1

    print(f"\nClasses found: {sorted(class_pixel_counts.keys())}")
    print(f"\nPer-class statistics:")
    print(f"{'ID':>4} {'Name':>20} {'Pixel Count':>15} {'Images':>8} {'Pixel %':>10}")
    print("-" * 62)
    total_pixels = sum(class_pixel_counts.values())
    for c in sorted(class_pixel_counts.keys()):
        name = class_names.get(c, f"Entity_{c}")
        pcount = class_pixel_counts[c]
        icount = class_image_counts[c]
        pct = pcount / total_pixels * 100
        print(f"{c:4d} {name:>20} {pcount:>15,} {icount:>8} {pct:>9.4f}%")

    # Plot class pixel distribution (excluding background)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pixel counts
    classes_sorted = [c for c in sorted(class_pixel_counts.keys()) if c > 0 and c < 23]
    pixel_vals = [class_pixel_counts[c] for c in classes_sorted]
    names = [class_names.get(c, str(c)) for c in classes_sorted]
    colors = []
    for c in classes_sorted:
        name = class_names.get(c, "")
        if name.startswith("C"):
            colors.append("#e74c3c")
        elif name.startswith("T"):
            colors.append("#2ecc71")
        elif name.startswith("L"):
            colors.append("#3498db")
        else:
            colors.append("#95a5a6")

    axes[0].barh(names, pixel_vals, color=colors)
    axes[0].set_xlabel('Total Pixel Count')
    axes[0].set_title('Pixel Distribution per Vertebra Class')

    # Image presence counts
    image_vals = [class_image_counts[c] for c in classes_sorted]
    axes[1].barh(names, image_vals, color=colors)
    axes[1].set_xlabel('Number of Images')
    axes[1].set_title('Image Presence per Vertebra Class')

    plt.suptitle('Class Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "eda_class_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Cobb angle distribution
    print("\n--- Cobb Angle Distribution ---")
    metrics_df = pd.read_csv(RADIOGRAPH_METRICS_CSV)
    cobb_angles = metrics_df["cobb_angle_deg"].values

    print(f"Cobb angles: n={len(cobb_angles)}")
    print(f"  Min: {cobb_angles.min():.1f} degrees")
    print(f"  Max: {cobb_angles.max():.1f} degrees")
    print(f"  Mean: {cobb_angles.mean():.1f} degrees")
    print(f"  Median: {np.median(cobb_angles):.1f} degrees")
    print(f"  Std: {cobb_angles.std():.1f} degrees")

    # Severity distribution
    mild = ((cobb_angles >= 10) & (cobb_angles < 25)).sum()
    moderate = ((cobb_angles >= 25) & (cobb_angles < 40)).sum()
    severe = (cobb_angles >= 40).sum()
    print(f"\n  Mild (10-25 deg): {mild} ({mild/len(cobb_angles)*100:.1f}%)")
    print(f"  Moderate (25-40 deg): {moderate} ({moderate/len(cobb_angles)*100:.1f}%)")
    print(f"  Severe (>40 deg): {severe} ({severe/len(cobb_angles)*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(cobb_angles, bins=20, color='mediumpurple', edgecolor='white')
    axes[0].axvline(x=10, color='green', linestyle='--', label='10 deg (mild threshold)')
    axes[0].axvline(x=25, color='orange', linestyle='--', label='25 deg (moderate)')
    axes[0].axvline(x=40, color='red', linestyle='--', label='40 deg (severe)')
    axes[0].set_xlabel('Cobb Angle (degrees)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Cobb Angle Distribution', fontsize=14)
    axes[0].legend(fontsize=9)

    # Pie chart of severity
    labels = ['Mild\n(10-25°)', 'Moderate\n(25-40°)', 'Severe\n(>40°)']
    sizes = [mild, moderate, severe]
    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
    axes[1].set_title('Scoliosis Severity Distribution', fontsize=14)

    plt.suptitle('Cobb Angle Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "eda_cobb_angles.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Sample visualizations
    print("\n--- Generating sample visualizations ---")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    samples = df.sample(n=8, random_state=42).reset_index(drop=True)
    for i, (_, row) in enumerate(samples.iterrows()):
        ax = axes[i // 4, i % 4]
        img = cv2.imread(str(DATASET_ROOT / row["radiograph_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(DATASET_ROOT / row["multiclass_id_png"]), cv2.IMREAD_UNCHANGED)
        if mask is not None and mask.ndim == 3:
            mask = mask[:, :, 0]

        # Create simple overlay
        if mask is not None:
            overlay = img.copy()
            colored = np.zeros_like(img)
            for c in range(1, 23):
                region = mask == c
                if region.any():
                    color = list({1:(255,0,0), 5:(255,100,0), 6:(255,200,0),
                                 12:(0,255,0), 17:(0,200,200), 18:(0,100,255),
                                 22:(0,0,255)}.get(c, (128,128,128)))
                    colored[region] = color
            overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
            ax.imshow(overlay)
        else:
            ax.imshow(img)

        ax.set_title(f"{row['image']} ({row['split']})", fontsize=9)
        ax.axis('off')

    plt.suptitle('Sample Radiographs with Segmentation Overlay', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "eda_samples.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll EDA visualizations saved to {OUTPUTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    analyze_dataset()
