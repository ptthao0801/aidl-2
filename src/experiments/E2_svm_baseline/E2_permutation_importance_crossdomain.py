# Cross-domain permutation importance
"""
E2_permutation_importance_crossdomain.py
===========================================

Chạy: python -m src.experiments.E2_svm_baseline.E2_permutation_importance_crossdomain
"""

from pathlib import Path

import pandas as pd

from src.features.feature_pipeline import build_feature_matrix
from src.experiments.E2_svm_baseline.E2_evaluate_indomain import (
    load_svm_and_scaler, load_config, BASE
)
from src.experiments.E2_svm_baseline.E2_permutation_importance_indomain import (
    compute_permutation_importance, N_REPEATS, RANDOM_SEED
)


def main():
    config = load_config()

    print("=" * 60)
    print("E2 — PERMUTATION IMPORTANCE (CROSS-DOMAIN, RealWaste)")
    print("=" * 60)

    model, scaler = load_svm_and_scaler(config)

    crossdomain_csv = BASE / config["data"]["crossdomain_csv"]
    cache_dir = BASE / config["data"]["feature_cache_dir"]
    crossdomain_cache = cache_dir / "E2_features_realwaste_test.npz"
    X_cross, y_cross, _ = build_feature_matrix(crossdomain_csv, cache_path=crossdomain_cache)

    results = compute_permutation_importance(
        model, scaler, X_cross, y_cross, n_repeats=N_REPEATS, seed=RANDOM_SEED
    )

    # ---- Overall ----
    rows = []
    for group_name, r in results.items():
        rows.append({
            "feature_group": group_name,
            "baseline_macro_f1": r["baseline_f1"],
            "f1_drop_mean": r["drop_mean"],
            "f1_drop_std": r["drop_std"],
        })
    df = pd.DataFrame(rows).sort_values("f1_drop_mean", ascending=False)

    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    output_path = table_dir / "E2_permutation_importance_crossdomain.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")
    print(df.to_string(index=False))

    # ---- Comparison vs in-domain (if already ran) ----
    indomain_path = table_dir / "E2_permutation_importance_indomain.csv"
    if indomain_path.exists():
        indomain_df = pd.read_csv(indomain_path)
        indomain_df["rank_indomain"] = indomain_df["f1_drop_mean"].rank(ascending=False)
        df["rank_crossdomain"] = df["f1_drop_mean"].rank(ascending=False)

        comparison = indomain_df[["feature_group", "rank_indomain"]].merge(
            df[["feature_group", "rank_crossdomain"]], on="feature_group"
        )
        comparison["rank_change"] = comparison["rank_indomain"] - comparison["rank_crossdomain"]

        comparison_path = table_dir / "E2_permutation_importance_rank_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\n[SAVED] Comparison rank in-domain vs cross-domain: {comparison_path}")
        print(comparison.to_string(index=False))
        print("\n(rank_change > 0: group gains importance under domain shift")
        print(" rank_change < 0: group loses importance under domain shift)")
    else:
        print("\n[WARN] Cannot find results in-domain — run E2_permutation_importance_indomain.py first.")


if __name__ == "__main__":
    main()