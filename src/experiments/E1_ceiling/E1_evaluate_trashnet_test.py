"""
E1_evaluate_trashnet_test.py
==============================
E1 — Evaluate on TrashNet held-out test set (in-domain, ceiling performance)

Load 3 checkpoint (seed 42/123/456) trained in E1_train_resnet50.py,
test on TrashNet test set.

Chạy: python -m src.experiments.E1_ceiling.E1_evaluate_trashnet_test
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    accuracy_score, confusion_matrix, matthews_corrcoef
)
from torch.utils.data import DataLoader
from torchvision import models

from src.experiments.E1_ceiling.E1_train_resnet50 import (
    WasteImageDataset, build_transforms, UNIFIED_CLASSES, CLASS_TO_IDX, BASE
)

CONFIG_PATH = Path(__file__).resolve().parent / "E1_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, num_classes: int, device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_all(model, dataloader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = config["data"]["image_size"]
    _, eval_transform = build_transforms(image_size, config["augmentation"])

    test_csv = BASE / config["data"]["test_csv"]
    test_ds = WasteImageDataset(test_csv, transform=eval_transform)
    batch_size = config["training"]["batch_size"]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    checkpoint_dir = BASE / config["output"]["checkpoint_dir"]
    seeds = config["seeds"]

    per_seed_results = []
    all_confusion_matrices = []

    print("=" * 60)
    print("E1 — EVALUATE ON TRASHNET TEST")
    print("=" * 60)

    for seed in seeds:
        checkpoint_path = checkpoint_dir / f"E1_resnet50_seed{seed}.pt"
        if not checkpoint_path.exists():
            print(f"[ERROR] Cannot find checkpoint: {checkpoint_path}")
            print("        Run E1_train_resnet50.py first.")
            continue

        model = load_model(checkpoint_path, config["model"]["num_classes"], device)
        preds, labels = predict_all(model, test_loader, device)

        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        accuracy = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            labels, preds, labels=range(len(UNIFIED_CLASSES)), zero_division=0
        )

        cm = confusion_matrix(labels, preds, labels=range(len(UNIFIED_CLASSES)))
        all_confusion_matrices.append(cm)

        print(f"\nSeed {seed}: Macro-F1={macro_f1:.4f}, Accuracy={accuracy:.4f}")

        per_seed_results.append({
            "seed": seed,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "mcc": mcc,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1_per_class,
        })

    if not per_seed_results:
        print("[ERROR] Cannot find checkpoint to evaluate. Stopped.")
        return

    # ---- Metric table (3 seeds) ----
    macro_f1_scores = [r["macro_f1"] for r in per_seed_results]
    accuracy_scores = [r["accuracy"] for r in per_seed_results]
    mcc_scores = [r["mcc"] for r in per_seed_results]

    table4 = pd.DataFrame({
        "metric": ["macro_f1", "accuracy", "mcc"],
        "mean": [np.mean(macro_f1_scores), np.mean(accuracy_scores), np.mean(mcc_scores)],
        "std": [np.std(macro_f1_scores), np.std(accuracy_scores), np.std(mcc_scores)],
    })

    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    table4_path = table_dir / "E1_trashnet_test_results.csv"
    table4.to_csv(table4_path, index=False)
    print(f"\n[SAVED] Table 4 (overall): {table4_path}")
    print(table4.to_string(index=False))

    # ---- Metric table (per-class) (3 seeds) ----
    per_class_rows = []
    for i, class_name in enumerate(UNIFIED_CLASSES):
        precisions = [r["per_class_precision"][i] for r in per_seed_results]
        recalls = [r["per_class_recall"][i] for r in per_seed_results]
        f1s = [r["per_class_f1"][i] for r in per_seed_results]
        per_class_rows.append({
            "class": class_name,
            "precision_mean": np.mean(precisions), "precision_std": np.std(precisions),
            "recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        })

    table5 = pd.DataFrame(per_class_rows)
    table5_path = table_dir / "E1_per_class_metrics.csv"
    table5.to_csv(table5_path, index=False)
    print(f"\n[SAVED] Table 5 (per-class): {table5_path}")
    print(table5.to_string(index=False))

    # ---- Confusion Matrix (avg, 3 seeds) ----
    avg_cm = np.mean(all_confusion_matrices, axis=0)
    cm_df = pd.DataFrame(avg_cm, index=UNIFIED_CLASSES, columns=UNIFIED_CLASSES)
    figure_dir = BASE / config["output"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    cm_csv_path = figure_dir / "E1_confusion_matrix_trashnet_raw.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"\n[SAVED] Confusion matrix (avg, 3 seeds): {cm_csv_path}")
    print("\nConfusion Matrix (avg 3 seeds, row=truth, column=pred):")
    print(cm_df.round(1).to_string())


if __name__ == "__main__":
    main()