"""
create_figure0_dataset_samples.py
====================================
Chạy: python -m scripts.create_figure0_dataset_samples
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parents[1]  # check again
UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

FIGURE_DIR = BASE / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def pick_representative_image(csv_path: Path, class_name: str, seed: int = 42) -> Path:
    df = pd.read_csv(csv_path)
    class_df = df[df["label"] == class_name].sample(frac=1.0, random_state=seed)
    mid_idx = len(class_df) // 2
    row = class_df.iloc[mid_idx]
    return BASE / row["filepath"]


def main():
    trashnet_csv = BASE / "data" / "splits" / "trashnet_test.csv"
    realwaste_csv = BASE / "data" / "splits" / "realwaste_test.csv"

    fig, axes = plt.subplots(2, len(UNIFIED_CLASSES), figsize=(15, 6.5))

    for col_idx, class_name in enumerate(UNIFIED_CLASSES):
        # TrashNet
        trashnet_img_path = pick_representative_image(trashnet_csv, class_name)
        trashnet_img = Image.open(trashnet_img_path).convert("RGB")
        axes[0, col_idx].imshow(trashnet_img)
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        axes[0, col_idx].set_title(class_name.capitalize(), fontsize=12, fontweight="bold")

        # RealWaste
        realwaste_img_path = pick_representative_image(realwaste_csv, class_name)
        realwaste_img = Image.open(realwaste_img_path).convert("RGB")
        axes[1, col_idx].imshow(realwaste_img)
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])

    # Labels
    axes[0, 0].set_ylabel("TrashNet\n(source)", fontsize=12, fontweight="bold",
                           rotation=0, ha="right", va="center", labelpad=40)
    axes[1, 0].set_ylabel("RealWaste\n(target)", fontsize=12, fontweight="bold",
                           rotation=0, ha="right", va="center", labelpad=40)

    fig.suptitle("Representative Samples: Source Domain vs Target Domain",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURE_DIR / "Figure0_dataset_samples.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()