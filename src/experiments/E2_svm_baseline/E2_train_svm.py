# GridSearchCV + fit SVM on TrashNet train
"""
E2_train_svm.py
=================

Chạy: python -m src.experiments.E2_svm_baseline.E2_train_svm
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import build_feature_matrix, TOTAL_FEATURE_DIM

BASE = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "E2_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    train_csv = BASE / config["data"]["train_csv"]
    cache_dir = BASE / config["data"]["feature_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E2 — TRAIN SVM BASELINE")
    print("=" * 60)

    # ---- 1. Feature ----
    train_cache = cache_dir / "E2_features_trashnet_train.npz"
    X_train, y_train, _ = build_feature_matrix(train_csv, cache_path=train_cache)
    print(f"\n[INFO] Feature matrix train: X={X_train.shape}, y={y_train.shape}")
    assert X_train.shape[1] == TOTAL_FEATURE_DIM, "Feature dimension does not match the 195-dim design"

    # ---- 2. StandardScaler ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ---- 3. GridSearchCV ----
    param_grid = {
        "C": config["svm"]["gridsearch"]["C"],
        "gamma": config["svm"]["gridsearch"]["gamma"],
    }
    cv_folds = config["svm"]["cv_folds"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    svm = SVC(
        kernel=config["svm"]["kernel"],
        class_weight=config["svm"]["class_weight"],
        probability=False,  
    )

    print(f"\n[INFO] Run GridSearchCV: C={param_grid['C']}, gamma={param_grid['gamma']}, "
          f"{cv_folds}-fold stratified CV...")
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n[RESULT] Best params: {grid_search.best_params_}")
    print(f"[RESULT] Best CV Macro-F1: {grid_search.best_score_:.4f}")

    best_svm = grid_search.best_estimator_

    # ---- 4. Save model + scaler ----
    model_dir = BASE / config["output"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "E2_svm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_svm, f)
    print(f"\n[SAVED] SVM model: {model_path}")

    scaler_path = model_dir / "E2_svm_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[SAVED] StandardScaler: {scaler_path}")

    # ---- 5. Save GridSearchCV result table ----
    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_path = table_dir / "E2_gridsearch_cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"[SAVED] CV results: {cv_results_path}")

    best_params_summary = pd.DataFrame([{
        "best_C": grid_search.best_params_["C"],
        "best_gamma": grid_search.best_params_["gamma"],
        "best_cv_macro_f1": grid_search.best_score_,
        "kernel": config["svm"]["kernel"],
        "class_weight": config["svm"]["class_weight"],
    }])
    best_params_path = table_dir / "E2_gridsearch_best_params.csv"
    best_params_summary.to_csv(best_params_path, index=False)
    print(f"[SAVED] Table 6 (best params summary): {best_params_path}")
    print(best_params_summary.to_string(index=False))


if __name__ == "__main__":
    main()