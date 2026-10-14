# Compute 95% Bootstrap CI
"""
E4_bootstrap_ci.py
=====================
Chạy: python -m src.experiments.E4_comparison.E4_bootstrap_ci
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models

BASE = Path(__file__).resolve().parents[3]
UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "E4_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bootstrap_macro_f1_ci(y_true: np.ndarray, y_pred: np.ndarray,
                           n_resamples: int, confidence_level: float,
                           seed: int) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)

    macro_f1_samples = []
    per_class_f1_samples = {c: [] for c in UNIFIED_CLASSES}

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)  # resample with replacement
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]

        macro_f1 = f1_score(y_true_resampled, y_pred_resampled,
                             average="macro", zero_division=0)
        macro_f1_samples.append(macro_f1)

        per_class = f1_score(y_true_resampled, y_pred_resampled,
                              labels=UNIFIED_CLASSES, average=None, zero_division=0)
        for i, c in enumerate(UNIFIED_CLASSES):
            per_class_f1_samples[c].append(per_class[i])

    alpha = 1 - confidence_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    result = {
        "macro_f1_mean": np.mean(macro_f1_samples),
        "macro_f1_ci_lower": np.percentile(macro_f1_samples, lower_pct),
        "macro_f1_ci_upper": np.percentile(macro_f1_samples, upper_pct),
    }
    for c in UNIFIED_CLASSES:
        result[f"{c}_f1_mean"] = np.mean(per_class_f1_samples[c])
        result[f"{c}_f1_ci_lower"] = np.percentile(per_class_f1_samples[c], lower_pct)
        result[f"{c}_f1_ci_upper"] = np.percentile(per_class_f1_samples[c], upper_pct)

    return result


def get_svm_predictions(feature_cache_path: Path, model_path: Path, scaler_path: Path):
    data = np.load(feature_cache_path, allow_pickle=True)
    X, y_true = data["X"], data["y"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_true, y_pred


def get_resnet_predictions(csv_path: Path, checkpoint_path: Path, image_size: int,
                            batch_size: int, device):
    from src.experiments.E1_ceiling.E1_train_resnet50 import (
        WasteImageDataset, build_transforms, CLASS_TO_IDX
    )

    _, eval_transform = build_transforms(image_size, {})
    ds = WasteImageDataset(csv_path, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(UNIFIED_CLASSES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels_idx = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_idx.extend(labels.numpy())

    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    y_true = np.array([idx_to_class[i] for i in all_labels_idx])
    y_pred = np.array([idx_to_class[i] for i in all_preds])
    return y_true, y_pred


def main():
    config = load_config()
    bootstrap_cfg = config["bootstrap"]
    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E4 — BOOTSTRAP 95% CONFIDENCE INTERVAL")
    print("=" * 60)

    results = {}

    # ---- SVM: in-domain + cross-domain ----
    svm_model_path = BASE / "outputs" / "checkpoints" / "E2_svm_model.pkl"
    svm_scaler_path = BASE / "outputs" / "checkpoints" / "E2_svm_scaler.pkl"

    if svm_model_path.exists():
        print("\n[SVM in-domain] Bootstrap CI...")
        y_true, y_pred = get_svm_predictions(
            BASE / config["data"]["trashnet_test_features_cache"],
            svm_model_path, svm_scaler_path
        )
        results["SVM (TrashNet, in-domain)"] = bootstrap_macro_f1_ci(
            y_true, y_pred, bootstrap_cfg["n_resamples"],
            bootstrap_cfg["confidence_level"], bootstrap_cfg["random_seed"]
        )

        print("[SVM cross-domain] Bootstrap CI...")
        y_true, y_pred = get_svm_predictions(
            BASE / config["data"]["realwaste_features_cache"],
            svm_model_path, svm_scaler_path
        )
        results["SVM (RealWaste, cross-domain)"] = bootstrap_macro_f1_ci(
            y_true, y_pred, bootstrap_cfg["n_resamples"],
            bootstrap_cfg["confidence_level"], bootstrap_cfg["random_seed"]
        )
    else:
        print("[WARN] Not found: SVM model — skip bootstrap CI for SVM.")

    # ---- ResNet50: in-domain + cross-domain ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_checkpoint = BASE / "outputs" / "checkpoints" / "E1_resnet50_seed42.pt"

    if resnet_checkpoint.exists():
        print("\n[ResNet50 in-domain, seed 42] Bootstrap CI...")
        y_true, y_pred = get_resnet_predictions(
            BASE / "data" / "splits" / "trashnet_test.csv",
            resnet_checkpoint, 224, 32, device
        )
        results["ResNet50 (TrashNet, in-domain)"] = bootstrap_macro_f1_ci(
            y_true, y_pred, bootstrap_cfg["n_resamples"],
            bootstrap_cfg["confidence_level"], bootstrap_cfg["random_seed"]
        )

        print("[ResNet50 cross-domain, seed 42] Bootstrap CI...")
        y_true, y_pred = get_resnet_predictions(
            BASE / "data" / "splits" / "realwaste_test.csv",
            resnet_checkpoint, 224, 32, device
        )
        results["ResNet50 (RealWaste, cross-domain)"] = bootstrap_macro_f1_ci(
            y_true, y_pred, bootstrap_cfg["n_resamples"],
            bootstrap_cfg["confidence_level"], bootstrap_cfg["random_seed"]
        )
    else:
        print("[WARN] Not found: ResNet50 checkpoint — skip bootstrap CI.")

    # ---- Save results ----
    rows = []
    for config_name, r in results.items():
        row = {"configuration": config_name}
        row.update(r)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = table_dir / "E4_bootstrap_ci.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")

    summary_cols = ["configuration", "macro_f1_mean", "macro_f1_ci_lower", "macro_f1_ci_upper"]
    print("\nMacro-F1 with 95% CI:")
    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()