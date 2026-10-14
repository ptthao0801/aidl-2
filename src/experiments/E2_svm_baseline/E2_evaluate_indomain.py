# Test SVM on TrashNet test
"""
E2_evaluate_indomain.py
=========================

Load model + scaler trained in E2_train_svm.py, test on TrashNet test,
calculate Macro-F1, Accuracy, MCC, per-class Precision/Recall/F1.

Chạy: python -m src.experiments.E2_svm_baseline.E2_evaluate_indomain
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix
)

from src.features.feature_pipeline import build_feature_matrix
from src.data_prep.class_mapping import UNIFIED_CLASSES

BASE = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "E2_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_svm_on_dataset(model, scaler, X: np.ndarray, y_true: np.ndarray) -> dict:
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=UNIFIED_CLASSES, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=UNIFIED_CLASSES)

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "mcc": mcc,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1_per_class,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_true": y_true,
    }


def load_svm_and_scaler(config: dict):
    model_dir = BASE / config["output"]["model_dir"]
    with open(model_dir / "E2_svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "E2_svm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def main():
    config = load_config()

    print("=" * 60)
    print("E2 — EVALUATE SVM ON TRASHNET TEST (in-domain)")
    print("=" * 60)

    model, scaler = load_svm_and_scaler(config)

    test_csv = BASE / config["data"]["test_csv"]
    cache_dir = BASE / config["data"]["feature_cache_dir"]
    test_cache = cache_dir / "E2_features_trashnet_test.npz"
    X_test, y_test, _ = build_feature_matrix(test_csv, cache_path=test_cache)

    result = evaluate_svm_on_dataset(model, scaler, X_test, y_test)

    print(f"\nMacro-F1:  {result['macro_f1']:.4f}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"MCC:       {result['mcc']:.4f}")

    # ---- Overall ----
    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame([{
        "domain": "TrashNet (in-domain)",
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "mcc": result["mcc"],
    }])
    overall_path = table_dir / "E2_indomain_overall_results.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"\n[SAVED] {overall_path}")

    # ---- Per-class ----
    per_class_df = pd.DataFrame({
        "class": UNIFIED_CLASSES,
        "precision": result["per_class_precision"],
        "recall": result["per_class_recall"],
        "f1": result["per_class_f1"],
    })
    per_class_path = table_dir / "E2_indomain_per_class_results.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"[SAVED] {per_class_path}")
    print(per_class_df.to_string(index=False))

    # ---- Confusion matrix ----
    figure_dir = BASE / config["output"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(result["confusion_matrix"], index=UNIFIED_CLASSES, columns=UNIFIED_CLASSES)
    cm_path = figure_dir / "E2_confusion_matrix_indomain_raw.csv"
    cm_df.to_csv(cm_path)
    print(f"[SAVED] Confusion matrix: {cm_path}")
    print("\nConfusion Matrix (row=truth, column=pred):")
    print(cm_df.to_string())


if __name__ == "__main__":
    main()