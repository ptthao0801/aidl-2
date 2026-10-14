# Generate E3 confusion matrix
"""
E3_confusion_matrix.py
=========================
Chạy: python -m src.experiments.E3_zeroshot_shift.E3_confusion_matrix
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

BASE = Path(__file__).resolve().parents[3]
UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_e3_config() -> dict:
    config_path = Path(__file__).resolve().parent / "E3_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    e3_config = load_e3_config()
    table_dir = BASE / "outputs" / "tables"
    figure_dir = BASE / e3_config["output"]["figure_dir"]

    e1_cm_path = figure_dir / "E1_confusion_matrix_trashnet_raw.csv"
    e3_cm_path = figure_dir / "E3_confusion_matrix_realwaste_raw.csv"

    if not (e1_cm_path.exists() and e3_cm_path.exists()):
        print("[ERROR] Not enough data: confusion matrix CSV. Run:")
        print("        - E1_evaluate_trashnet_test.py")
        print("        - E3_evaluate_realwaste.py")
        return

    cm_e1 = pd.read_csv(e1_cm_path, index_col=0)
    cm_e3 = pd.read_csv(e3_cm_path, index_col=0)

    # Normalize 
    cm_e1_norm = cm_e1.div(cm_e1.sum(axis=1), axis=0) * 100
    cm_e3_norm = cm_e3.div(cm_e3.sum(axis=1), axis=0) * 100

    # Get Macro-F1 for titles
    e1_f1, e3_f1 = None, None
    e1_results_path = table_dir / "E1_trashnet_test_results.csv"
    e3_results_path = table_dir / "E3_realwaste_results.csv"
    if e1_results_path.exists():
        e1_df = pd.read_csv(e1_results_path)
        e1_f1 = e1_df[e1_df["metric"] == "macro_f1"]["mean"].values[0]
    if e3_results_path.exists():
        e3_df = pd.read_csv(e3_results_path)
        e3_f1 = e3_df[e3_df["metric"] == "macro_f1"]["mean"].values[0]

    e1_title = f"ResNet50 — TrashNet Test (E1, Ceiling)\nMacro-F1 = {e1_f1:.3f}" if e1_f1 else \
        "ResNet50 — TrashNet Test (E1, Ceiling)"
    e3_title = f"ResNet50 — RealWaste (E3, Zero-Shot)\nMacro-F1 = {e3_f1:.3f}" if e3_f1 else \
        "ResNet50 — RealWaste (E3, Zero-Shot)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm_e1_norm, annot=True, fmt=".1f", cmap="Blues",
        vmin=0, vmax=100, ax=axes[0], cbar_kws={"label": "% of true class"}
    )
    axes[0].set_title(e1_title)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_e3_norm, annot=True, fmt=".1f", cmap="Reds",
        vmin=0, vmax=100, ax=axes[1], cbar_kws={"label": "% of true class"}
    )
    axes[1].set_title(e3_title)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.suptitle("ResNet50: In-Domain Ceiling vs Zero-Shot Domain Shift",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = figure_dir / "E3_confusion_matrix_e1_vs_e3.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")

    # ---- Compare the top confusion pair for each model ----
    print("\n" + "=" * 60)
    print("TOP CONFUSION PAIR ANALYSIS")
    print("=" * 60)
    for name, cm_norm in [("E1 (TrashNet)", cm_e1_norm), ("E3 (RealWaste)", cm_e3_norm)]:
        cm_off_diag = cm_norm.copy()
        for c in UNIFIED_CLASSES:
            cm_off_diag.loc[c, c] = 0  
        max_confusion = cm_off_diag.stack().idxmax()
        max_value = cm_off_diag.stack().max()
        print(f"{name}: top confusion is '{max_confusion[0]}' -> "
              f"'{max_confusion[1]}' ({max_value:.1f}% of images in this class)")


if __name__ == "__main__":
    main()