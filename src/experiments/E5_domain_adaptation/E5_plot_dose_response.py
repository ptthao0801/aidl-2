"""
E5_plot_dose_response.py
===========================
Chạy: python -m src.experiments.E5_domain_adaptation.E5_plot_dose_response
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
TABLE_DIR = BASE / "outputs" / "tables"
FIGURE_DIR = BASE / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DOSE_RESPONSE_VARIANTS = {
    "combined_weak": 0.3,
    "combined": 0.5,
    "combined_strong": 0.8,
    "combined_max": 1.0,
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    summary_path = TABLE_DIR / "E5_variants_comparison_summary.csv"
    if not summary_path.exists():
        print(f"[ERROR] Cannot find {summary_path}. Run E5_evaluate_variants.py first.")
        return

    df = pd.read_csv(summary_path)

    dose_df = df[df["variant"].isin(DOSE_RESPONSE_VARIANTS.keys())].copy()
    dose_df["intensity"] = dose_df["variant"].map(DOSE_RESPONSE_VARIANTS)
    dose_df = dose_df.sort_values("intensity").reset_index(drop=True)

    if len(dose_df) < 2:
        print("[ERROR] Needs at least 2 points dose-response to plot. "
              "Check on E5_variants_comparison_summary.csv.")
        return

    intensities = dose_df["intensity"].values
    recovery = dose_df["recovery_ratio_pct"].values
    retention = dose_df["retention_ratio_pct"].values


    e1_f1 = 0.9490  # from E1_trashnet_test_results.csv
    e3_f1 = 0.4208  # from E3_realwaste_results.csv
    recovery_std_pct = dose_df["realwaste_macro_f1_std"].values / (e1_f1 - e3_f1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Left panel: Recovery ratio by intensity ----
    axes[0].errorbar(
        intensities, recovery, yerr=recovery_std_pct,
        marker="o", markersize=8, linewidth=2, capsize=5,
        color="#C44E52", label="Recovery ratio"
    )
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("ColorJitter intensity (brightness/contrast/saturation)")
    axes[0].set_ylabel("Recovery ratio (%)")
    axes[0].set_title("How Much of the Performance Gap\nDoes Augmentation Recover?")
    axes[0].grid(alpha=0.3)
    for x, y in zip(intensities, recovery):
        axes[0].annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 12),
                          textcoords="offset points", ha="center", fontsize=9)

    # Mark the peak point
    peak_idx = np.argmax(recovery)
    axes[0].scatter([intensities[peak_idx]], [recovery[peak_idx]],
                     s=200, facecolors="none", edgecolors="black", linewidths=2, zorder=5)

    # ---- Right panel phải: Retention ratio by intensity ----
    axes[1].plot(
        intensities, retention,
        marker="s", markersize=8, linewidth=2,
        color="#4C72B0", label="Retention ratio"
    )
    axes[1].axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                     label="No regression (100%)")
    axes[1].set_xlabel("ColorJitter intensity (brightness/contrast/saturation)")
    axes[1].set_ylabel("Retention ratio (%)")
    axes[1].set_title("Does Stronger Augmentation Hurt\nOriginal-Domain Performance?")
    axes[1].set_ylim(95, 101)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=9)
    for x, y in zip(intensities, retention):
        axes[1].annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, -15),
                          textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle("Dose-Response of Colour Augmentation Intensity on Domain Shift Recovery",
                  fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    output_path = FIGURE_DIR / "E5_dose_response_curve.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] {output_path}")

    # ---- Summary tablet ----
    print("\nDose-Response Summary:")
    print(dose_df[["variant", "intensity", "recovery_ratio_pct", "retention_ratio_pct"]]
          .to_string(index=False))

    print(f"\n[PEAK] Highest recovery ratio at intensity {intensities[peak_idx]}: "
          f"{recovery[peak_idx]:.1f}%")
    if peak_idx < len(intensities) - 1:
        print(f"[NOTE] Recovery slightly decreases after this point (at intensity {intensities[-1]}: "
              f"{recovery[-1]:.1f}%)")


if __name__ == "__main__":
    main()