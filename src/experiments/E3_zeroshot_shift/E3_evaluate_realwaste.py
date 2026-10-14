"""
E3_evaluate_realwaste.py
===========================
Chạy: python -m src.experiments.E3_zeroshot_shift.E3_evaluate_realwaste
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support, confusion_matrix
)
from torch.utils.data import DataLoader
from torchvision import models

from src.experiments.E1_ceiling.E1_train_resnet50 import (
    WasteImageDataset, build_transforms, UNIFIED_CLASSES, CLASS_TO_IDX, BASE
)

E1_CONFIG_PATH = Path(__file__).resolve().parents[1] / "E1_ceiling" / "E1_config.yaml"
E3_OUTPUT_CONFIG_PATH = Path(__file__).resolve().parent / "E3_config.yaml"


def load_e1_config() -> dict:
    with open(E1_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_e3_config() -> dict:
    with open(E3_OUTPUT_CONFIG_PATH, "r", encoding="utf-8") as f:
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
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    e1_config = load_e1_config()
    e3_config = load_e3_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("E3 — ZERO-SHOT DOMAIN SHIFT TEST (RealWaste)")
    print("=" * 60)
    print(f"Device: {device}")
    print("[Note] No retain. Use checkpoint from E1.\n")

    image_size = e1_config["data"]["image_size"]
    _, eval_transform = build_transforms(image_size, e1_config["augmentation"])

    realwaste_csv = BASE / e3_config["data"]["realwaste_csv"]
    realwaste_ds = WasteImageDataset(realwaste_csv, transform=eval_transform)
    batch_size = e1_config["training"]["batch_size"]
    num_workers = e1_config["training"].get("num_workers", 4)
    realwaste_loader = DataLoader(
        realwaste_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    checkpoint_dir = BASE / e1_config["output"]["checkpoint_dir"]
    seeds = e1_config["seeds"]

    per_seed_results = []
    all_confusion_matrices = []

    for seed in seeds:
        checkpoint_path = checkpoint_dir / f"E1_resnet50_seed{seed}.pt"
        if not checkpoint_path.exists():
            print(f"[ERROR] Cannot find checkpoint: {checkpoint_path}")
            print("        Run E1_train_resnet50.py first.")
            continue

        print(f"[SEED {seed}] Load checkpoint and test on RealWaste...")
        model = load_model(checkpoint_path, e1_config["model"]["num_classes"], device)
        preds, labels = predict_all(model, realwaste_loader, device)

        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        accuracy = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            labels, preds, labels=range(len(UNIFIED_CLASSES)), zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=range(len(UNIFIED_CLASSES)))
        all_confusion_matrices.append(cm)

        print(f"    Macro-F1={macro_f1:.4f}, Accuracy={accuracy:.4f}, MCC={mcc:.4f}")

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
        print("[ERROR] No checkpoint to evaluate. Stopped.")
        return

    # ---- Overall (3 seeds) ----
    macro_f1_scores = [r["macro_f1"] for r in per_seed_results]
    accuracy_scores = [r["accuracy"] for r in per_seed_results]
    mcc_scores = [r["mcc"] for r in per_seed_results]

    table8 = pd.DataFrame({
        "metric": ["macro_f1", "accuracy", "mcc"],
        "mean": [np.mean(macro_f1_scores), np.mean(accuracy_scores), np.mean(mcc_scores)],
        "std": [np.std(macro_f1_scores), np.std(accuracy_scores), np.std(mcc_scores)],
    })

    table_dir = BASE / e3_config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    table8_path = table_dir / "E3_realwaste_results.csv"
    table8.to_csv(table8_path, index=False)
    print(f"\n[SAVED] Table 8 (overall): {table8_path}")
    print(table8.to_string(index=False))

    # ---- Per-class (3 seed) ----
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

    table9 = pd.DataFrame(per_class_rows)
    table9_path = table_dir / "E3_per_class_metrics.csv"
    table9.to_csv(table9_path, index=False)
    print(f"\n[SAVED] Table 9 (per-class): {table9_path}")
    print(table9.to_string(index=False))

    # ---- Confusion Matrix 3 seed ----
    avg_cm = np.mean(all_confusion_matrices, axis=0)
    cm_df = pd.DataFrame(avg_cm, index=UNIFIED_CLASSES, columns=UNIFIED_CLASSES)
    figure_dir = BASE / e3_config["output"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    cm_csv_path = figure_dir / "E3_confusion_matrix_realwaste_raw.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"\n[SAVED] Confusion matrix (avg 3 seed): {cm_csv_path}")
    print("\nConfusion Matrix (avg 3 seed, row=truth, column=pred):")
    print(cm_df.round(1).to_string())

    # ---- Comparison vs E1 (in-domain) ----
    e1_results_path = BASE / e1_config["output"]["table_dir"] / "E1_trashnet_test_results.csv"
    if e1_results_path.exists():
        e1_df = pd.read_csv(e1_results_path)
        e1_macro_f1 = e1_df[e1_df["metric"] == "macro_f1"]["mean"].values[0]
        e3_macro_f1 = table8[table8["metric"] == "macro_f1"]["mean"].values[0]
        abs_drop = e1_macro_f1 - e3_macro_f1
        rel_drop = abs_drop / e1_macro_f1 * 100
        print(f"\n{'=' * 60}")
        print("COMPARISON VS E1")
        print(f"{'=' * 60}")
        print(f"E1 (TrashNet, in-domain):    Macro-F1 = {e1_macro_f1:.4f}")
        print(f"E3 (RealWaste, cross-domain):   Macro-F1 = {e3_macro_f1:.4f}")
        print(f"Absolute drop: {abs_drop:.4f} ({rel_drop:.1f}% relative decline)")
    else:
        print("\n[INFO] E1 results not found — "
              "run E1_evaluate_trashnet_test.py first.")


if __name__ == "__main__":
    main()