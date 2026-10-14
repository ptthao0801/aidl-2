"""
E2_plot_gridsearch_heatmap.py
================================

Chạy: python -m src.experiments.E2_svm_baseline.E2_plot_gridsearch_heatmap
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

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    cv_results_path = TABLE_DIR / "E2_gridsearch_cv_results.csv"
    if not cv_results_path.exists():
        print(f"[ERROR] Cannot find {cv_results_path}. Run E2_train_svm.py first.")
        return

    df = pd.read_csv(cv_results_path)

    df["C"] = df["param_C"].astype(float)
    df["gamma"] = df["param_gamma"].astype(str)

    # Pivot to marix C x gamma, value = mean_test_score (Macro-F1)
    pivot = df.pivot(index="C", columns="gamma", values="mean_test_score")
    # sort 
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="viridis",
        cbar_kws={"label": "Cross-validation Macro-F1"},
        ax=ax, linewidths=0.5, linecolor="white"
    )

    best_idx = pivot.stack().idxmax()
    best_c, best_gamma = best_idx
    row_pos = list(pivot.index).index(best_c)
    col_pos = list(pivot.columns).index(best_gamma)
    ax.add_patch(plt.Rectangle((col_pos, row_pos), 1, 1, fill=False,
                                edgecolor="red", linewidth=3))

    ax.set_xlabel("γ (gamma)")
    ax.set_ylabel("C")
    ax.set_title("SVM Grid Search — Macro-F1 across All Candidate Settings\n"
                  f"Best: C={best_c:g}, γ={best_gamma} (red box)")

    plt.tight_layout()
    output_path = FIGURE_DIR / "E2_gridsearch_heatmap.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")

    # In bảng đầy đủ để đối chiếu
    print("\nGrid search results (Macro-F1):")
    print(pivot.to_string())


if __name__ == "__main__":
    main()