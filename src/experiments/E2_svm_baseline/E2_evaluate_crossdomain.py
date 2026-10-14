# Test SVM on RealWaste (zero-shot)
"""
E2_evaluate_crossdomain.py
=============================

Chạy: python -m src.experiments.E2_svm_baseline.E2_evaluate_crossdomain
"""

from pathlib import Path

import pandas as pd
import yaml

from src.features.feature_pipeline import build_feature_matrix
from src.data_prep.class_mapping import UNIFIED_CLASSES
from src.experiments.E2_svm_baseline.E2_evaluate_indomain import (
    evaluate_svm_on_dataset, load_svm_and_scaler, load_config, BASE
)


def main():
    config = load_config()

    print("=" * 60)
    print("E2 — EVALUATE SVM ON REALWASTE (cross-domain)")
    print("=" * 60)

    model, scaler = load_svm_and_scaler(config)

    crossdomain_csv = BASE / config["data"]["crossdomain_csv"]
    cache_dir = BASE / config["data"]["feature_cache_dir"]
    crossdomain_cache = cache_dir / "E2_features_realwaste_test.npz"

    X_cross, y_cross, _ = build_feature_matrix(crossdomain_csv, cache_path=crossdomain_cache)

    result = evaluate_svm_on_dataset(model, scaler, X_cross, y_cross)

    print(f"\nMacro-F1:  {result['macro_f1']:.4f}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"MCC:       {result['mcc']:.4f}")

    # ---- Overall ----
    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame([{
        "domain": "RealWaste (cross-domain)",
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "mcc": result["mcc"],
    }])
    overall_path = table_dir / "E2_crossdomain_overall_results.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"\n[SAVED] {overall_path}")

    # ---- Per-class ----
    per_class_df = pd.DataFrame({
        "class": UNIFIED_CLASSES,
        "precision": result["per_class_precision"],
        "recall": result["per_class_recall"],
        "f1": result["per_class_f1"],
    })
    per_class_path = table_dir / "E2_crossdomain_per_class_results.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"[SAVED] {per_class_path}")
    print(per_class_df.to_string(index=False))

    # ---- Confusion matrix ----
    figure_dir = BASE / config["output"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(result["confusion_matrix"], index=UNIFIED_CLASSES, columns=UNIFIED_CLASSES)
    cm_path = figure_dir / "E2_confusion_matrix_crossdomain_raw.csv"
    cm_df.to_csv(cm_path)
    print(f"[SAVED] Confusion matrix: {cm_path}")
    print("\nConfusion Matrix (row=truth, column=pred):")
    print(cm_df.to_string())

    # ---- In-domain + cross-domain table ----
    indomain_path = table_dir / "E2_indomain_overall_results.csv"
    if indomain_path.exists():
        indomain_df = pd.read_csv(indomain_path)
        table7 = pd.concat([indomain_df, overall_df], ignore_index=True)
        table7_path = table_dir / "E2_table7_indomain_vs_crossdomain.csv"
        table7.to_csv(table7_path, index=False)
        print(f"\n[SAVED] Table 7 (so sánh in-domain vs cross-domain): {table7_path}")
        print(table7.to_string(index=False))
    else:
        print("\n[WARN] Cannot find results in-domain — run E2_evaluate_indomain.py first.")


if __name__ == "__main__":
    main()