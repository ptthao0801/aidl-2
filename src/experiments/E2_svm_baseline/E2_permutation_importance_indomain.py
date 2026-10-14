# In-domain permutation importance
"""
E2_permutation_importance_indomain.py
========================================

Chạy: python -m src.experiments.E2_svm_baseline.E2_permutation_importance_indomain
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features.feature_pipeline import build_feature_matrix, FEATURE_GROUP_SLICES
from src.experiments.E2_svm_baseline.E2_evaluate_indomain import (
    load_svm_and_scaler, load_config, BASE
)

N_REPEATS = 20 
RANDOM_SEED = 42


def compute_permutation_importance(model, scaler, X: np.ndarray, y: np.ndarray,
                                    n_repeats: int = N_REPEATS, seed: int = RANDOM_SEED) -> dict:
    rng = np.random.RandomState(seed)

    X_scaled = scaler.transform(X)
    y_pred_baseline = model.predict(X_scaled)
    baseline_f1 = f1_score(y, y_pred_baseline, average="macro", zero_division=0)
    print(f"[BASELINE] Macro-F1 (no shuffle): {baseline_f1:.4f}")

    results = {}
    for group_name, group_slice in FEATURE_GROUP_SLICES.items():
        drops = []
        for repeat in range(n_repeats):
            X_permuted = X.copy()
            # Shuffle toàn bộ cột trong group_slice qua các hàng (mẫu),
            # phá vỡ mối liên hệ feature-label CHỈ ở nhóm này
            permuted_indices = rng.permutation(len(X_permuted))
            X_permuted[:, group_slice] = X_permuted[permuted_indices][:, group_slice]

            X_permuted_scaled = scaler.transform(X_permuted)
            y_pred_permuted = model.predict(X_permuted_scaled)
            permuted_f1 = f1_score(y, y_pred_permuted, average="macro", zero_division=0)

            drop = baseline_f1 - permuted_f1
            drops.append(drop)

        results[group_name] = {
            "baseline_f1": baseline_f1,
            "drop_mean": np.mean(drops),
            "drop_std": np.std(drops),
        }
        print(f"[{group_name.upper()}] Mean F1 drop after shuffle: "
              f"{np.mean(drops):.4f} +/- {np.std(drops):.4f}")

    return results


def main():
    config = load_config()

    print("=" * 60)
    print("E2 — PERMUTATION IMPORTANCE (IN-DOMAIN, TrashNet test)")
    print("=" * 60)

    model, scaler = load_svm_and_scaler(config)

    test_csv = BASE / config["data"]["test_csv"]
    cache_dir = BASE / config["data"]["feature_cache_dir"]
    test_cache = cache_dir / "E2_features_trashnet_test.npz"
    X_test, y_test, _ = build_feature_matrix(test_csv, cache_path=test_cache)

    results = compute_permutation_importance(model, scaler, X_test, y_test)

    # ---- PI results ----
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
    output_path = table_dir / "E2_permutation_importance_indomain.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()