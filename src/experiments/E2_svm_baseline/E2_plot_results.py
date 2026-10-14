"""
E2_plot_results.py
=====================
1. Confusion Matrix heatmap — in-domain vs cross-domain 
2. Bar chart Macro-F1/Accuracy/MCC — in-domain vs cross-domain
3. Permutation Importance bar chart — in-domain vs cross-domain 
4. Per-class F1 bar chart — in-domain vs cross-domain 

Chạy: python -m src.experiments.E2_svm_baseline.E2_plot_results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parents[3]
TABLE_DIR = BASE / "outputs" / "tables"
FIGURE_DIR = BASE / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_confusion_matrices_side_by_side():
    """Figure: Confusion Matrix — TrashNet (in-domain) vs RealWaste (cross-domain)."""
    indomain_cm_path = FIGURE_DIR / "E2_confusion_matrix_indomain_raw.csv"
    crossdomain_cm_path = FIGURE_DIR / "E2_confusion_matrix_crossdomain_raw.csv"

    if not (indomain_cm_path.exists() and crossdomain_cm_path.exists()):
        print("[SKIP] Not enough data from confusion matrix CSV — run E2_evaluate_indomain.py "
              "and E2_evaluate_crossdomain.py first.")
        return

    cm_indomain = pd.read_csv(indomain_cm_path, index_col=0)
    cm_crossdomain = pd.read_csv(crossdomain_cm_path, index_col=0)


    cm_indomain_norm = cm_indomain.div(cm_indomain.sum(axis=1), axis=0) * 100
    cm_crossdomain_norm = cm_crossdomain.div(cm_crossdomain.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm_indomain_norm, annot=True, fmt=".1f", cmap="Blues",
        vmin=0, vmax=100, ax=axes[0], cbar_kws={"label": "% of true class"}
    )
    axes[0].set_title("SVM — TrashNet Test (In-Domain)\nMacro-F1 = 0.837")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_crossdomain_norm, annot=True, fmt=".1f", cmap="Reds",
        vmin=0, vmax=100, ax=axes[1], cbar_kws={"label": "% of true class"}
    )
    axes[1].set_title("SVM — RealWaste (Cross-Domain, Zero-Shot)\nMacro-F1 = 0.177")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    output_path = FIGURE_DIR / "E2_confusion_matrix_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")


def plot_overall_metrics_comparison():
    """Figure: Bar chart Macro-F1/Accuracy/MCC — in-domain vs cross-domain."""
    table7_path = TABLE_DIR / "E2_table7_indomain_vs_crossdomain.csv"
    if not table7_path.exists():
        print("[SKIP] Table not found — run E2_evaluate_crossdomain.py.")
        return

    df = pd.read_csv(table7_path)

    metrics = ["macro_f1", "accuracy", "mcc"]
    metric_labels = ["Macro-F1", "Accuracy", "MCC"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    indomain_vals = df[df["domain"].str.contains("in-domain")][metrics].values[0]
    crossdomain_vals = df[df["domain"].str.contains("cross-domain")][metrics].values[0]

    bars1 = ax.bar(x - width / 2, indomain_vals, width, label="TrashNet (in-domain)",
                    color="#4C72B0")
    bars2 = ax.bar(x + width / 2, crossdomain_vals, width, label="RealWaste (cross-domain)",
                    color="#C44E52")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("SVM Baseline — Performance Degradation under Dataset Shift")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = FIGURE_DIR / "E2_overall_metrics_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")


def plot_permutation_importance_side_by_side():
    """Figure: Permutation Importance — in-domain vs cross-domain."""
    indomain_path = TABLE_DIR / "E2_permutation_importance_indomain.csv"
    crossdomain_path = TABLE_DIR / "E2_permutation_importance_crossdomain.csv"

    if not (indomain_path.exists() and crossdomain_path.exists()):
        print("[SKIP] Not enough data: permutation importance CSV.")
        return

    df_in = pd.read_csv(indomain_path).sort_values("f1_drop_mean", ascending=True)
    df_cross = pd.read_csv(crossdomain_path).sort_values("f1_drop_mean", ascending=True)

    group_colors = {"hsv": "#DD8452", "lbp": "#55A868", "hog": "#4C72B0"}
    group_labels = {"hsv": "HSV (color)", "lbp": "LBP (texture)", "hog": "HOG (shape)"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    max_drop = max(df_in["f1_drop_mean"].max(), df_cross["f1_drop_mean"].max())
    shared_xlim = (0, max_drop * 1.25)

    for ax, df, title in zip(
        axes, [df_in, df_cross],
        ["In-Domain (TrashNet test)\nBaseline Macro-F1 = 0.837",
         "Cross-Domain (RealWaste)\nBaseline Macro-F1 = 0.177"]
    ):
        colors = [group_colors[g] for g in df["feature_group"]]
        labels = [group_labels[g] for g in df["feature_group"]]
        bars = ax.barh(labels, df["f1_drop_mean"], xerr=df["f1_drop_std"],
                        color=colors, capsize=4)
        ax.set_xlabel("Mean Macro-F1 drop when shuffled")
        ax.set_title(title)
        ax.set_xlim(shared_xlim)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, df["f1_drop_mean"]):
            ax.annotate(f"{val:.3f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)

    fig.suptitle("Permutation Feature Importance — Which Feature Group Matters Most?",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = FIGURE_DIR / "E2_permutation_importance_comparison_sharedscale.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")

    # ---- Ver 2 ----

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title in zip(
        axes2, [df_in, df_cross],
        ["In-Domain (TrashNet test)\nBaseline Macro-F1 = 0.837",
         "Cross-Domain (RealWaste)\nBaseline Macro-F1 = 0.177"]
    ):
        colors = [group_colors[g] for g in df["feature_group"]]
        labels = [group_labels[g] for g in df["feature_group"]]
        bars = ax.barh(labels, df["f1_drop_mean"], xerr=df["f1_drop_std"],
                        color=colors, capsize=4)
        ax.set_xlabel("Mean Macro-F1 drop when shuffled")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, df["f1_drop_mean"]):
            ax.annotate(f"{val:.3f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)

    fig2.suptitle("Permutation Feature Importance — Relative Ranking within Each Domain",
                   fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path2 = FIGURE_DIR / "E2_permutation_importance_comparison_independentscale.png"
    plt.savefig(output_path2)
    plt.close()
    print(f"[SAVED] {output_path2}")


def plot_per_class_f1_comparison():
    """Figure: Per-class F1 — in-domain vs cross-domain (grouped bar chart)."""
    indomain_path = TABLE_DIR / "E2_indomain_per_class_results.csv"
    crossdomain_path = TABLE_DIR / "E2_crossdomain_per_class_results.csv"

    if not (indomain_path.exists() and crossdomain_path.exists()):
        print("[SKIP] Not enough data: per-class results CSV.")
        return

    df_in = pd.read_csv(indomain_path).set_index("class").reindex(UNIFIED_CLASSES)
    df_cross = pd.read_csv(crossdomain_path).set_index("class").reindex(UNIFIED_CLASSES)

    x = np.arange(len(UNIFIED_CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, df_in["f1"], width, label="TrashNet (in-domain)",
                    color="#4C72B0")
    bars2 = ax.bar(x + width / 2, df_cross["f1"], width, label="RealWaste (cross-domain)",
                    color="#C44E52")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in UNIFIED_CLASSES])
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.0)
    ax.set_title("SVM Per-Class F1 — In-Domain vs Cross-Domain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = FIGURE_DIR / "E2_per_class_f1_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")


def main():
    print("=" * 60)
    print("E2 — PLOT CHARTS")
    print("=" * 60)

    plot_confusion_matrices_side_by_side()
    plot_overall_metrics_comparison()
    plot_permutation_importance_side_by_side()
    plot_per_class_f1_comparison()

    print("\n[DONE] Charts saved to outputs/figures/")


if __name__ == "__main__":
    main()