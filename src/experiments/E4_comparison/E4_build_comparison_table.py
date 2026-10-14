# Merge E1/E2/E3 results into Table 10
"""
E4_build_comparison_table.py
===============================
Chạy: python -m src.experiments.E4_comparison.E4_build_comparison_table
"""

from pathlib import Path

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "E4_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    table_dir = BASE / config["output"]["table_dir"]

    print("=" * 60)
    print("E4 — STRUCTURED COMPARISON TABLE (Table 10)")
    print("=" * 60)

    # ---- Read results from E1/E2/E3 ----
    required_files = {
        "e1": table_dir / "E1_trashnet_test_results.csv",
        "e2_indomain": table_dir / "E2_indomain_overall_results.csv",
        "e2_crossdomain": table_dir / "E2_crossdomain_overall_results.csv",
        "e3": table_dir / "E3_realwaste_results.csv",
    }
    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        print(f"[ERROR] Not found results from: {missing}")
        print("        Run E1/E2/E3 first.")
        return

    e1_df = pd.read_csv(required_files["e1"])
    e2_in_df = pd.read_csv(required_files["e2_indomain"])
    e2_cross_df = pd.read_csv(required_files["e2_crossdomain"])
    e3_df = pd.read_csv(required_files["e3"])

    def get_metric(df, metric_name, value_col="mean", fallback_col=None):
        if "metric" in df.columns:
            row = df[df["metric"] == metric_name]
            if row.empty:
                return None, None
            mean_val = row[value_col].values[0]
            std_val = row["std"].values[0] if "std" in df.columns else 0.0
            return mean_val, std_val
        else:
            if metric_name not in df.columns:
                return None, None
            return df[metric_name].values[0], 0.0  

    # ---- Result ----
    rows = []
    for model_name, in_df, cross_df in [
        ("SVM (hand-engineered)", e2_in_df, e2_cross_df),
        ("ResNet50 (learned)", e1_df, e3_df),
    ]:
        in_f1, in_f1_std = get_metric(in_df, "macro_f1")
        cross_f1, cross_f1_std = get_metric(cross_df, "macro_f1")
        in_mcc, _ = get_metric(in_df, "mcc")
        cross_mcc, _ = get_metric(cross_df, "mcc")

        abs_drop = in_f1 - cross_f1
        rel_drop_pct = abs_drop / in_f1 * 100

        rows.append({
            "model": model_name,
            "trashnet_macro_f1": in_f1,
            "trashnet_macro_f1_std": in_f1_std,
            "realwaste_macro_f1": cross_f1,
            "realwaste_macro_f1_std": cross_f1_std,
            "absolute_drop": abs_drop,
            "relative_decline_pct": rel_drop_pct,
            "trashnet_mcc": in_mcc,
            "realwaste_mcc": cross_mcc,
        })

    table10 = pd.DataFrame(rows)
    table10_path = table_dir / "E4_comparison_table.csv"
    table10.to_csv(table10_path, index=False)

    print(f"\n[SAVED] Table 10: {table10_path}\n")
    print(table10.to_string(index=False))

    # ---- for RQ ----
    svm_decline = table10[table10["model"].str.contains("SVM")]["relative_decline_pct"].values[0]
    resnet_decline = table10[table10["model"].str.contains("ResNet")]["relative_decline_pct"].values[0]
    gap = svm_decline - resnet_decline

    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")
    print(f"SVM relative decline:      {svm_decline:.1f}%")
    print(f"ResNet50 relative decline: {resnet_decline:.1f}%")
    print(f"Gap (SVM degrade vs ResNet50): {gap:.1f} %")
    print(f"\n-> ResNet50 degrade {'LESS THAN' if resnet_decline < svm_decline else 'MORE THAN'} "
          f"SVM.")


if __name__ == "__main__":
    main()