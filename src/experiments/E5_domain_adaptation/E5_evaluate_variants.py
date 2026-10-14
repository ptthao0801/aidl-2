"""
E5_evaluate_variants.py
==========================

Chạy: python -m src.experiments.E5_domain_adaptation.E5_evaluate_variants
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader
from torchvision import models

from src.experiments.E1_ceiling.E1_train_resnet50 import (
    WasteImageDataset, build_transforms, UNIFIED_CLASSES, BASE
)

CONFIG_PATH = Path(__file__).resolve().parent / "E5_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, num_classes: int, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(model, dataloader, device) -> dict:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    return {"macro_f1": macro_f1, "accuracy": accuracy, "mcc": mcc}


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("E5 — EVALUATE VARIANTS (TrashNet retention + RealWaste recovery)")
    print("=" * 60)

    image_size = config["data"]["image_size"]
    _, eval_transform = build_transforms(image_size, {})

    trashnet_test_csv = BASE / config["data"]["test_csv"]
    realwaste_csv = BASE / config["data"]["realwaste_csv"]

    trashnet_ds = WasteImageDataset(trashnet_test_csv, transform=eval_transform)
    realwaste_ds = WasteImageDataset(realwaste_csv, transform=eval_transform)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    trashnet_loader = DataLoader(trashnet_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)
    realwaste_loader = DataLoader(realwaste_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers)

    checkpoint_dir = BASE / config["output"]["checkpoint_dir"]
    seeds = config["seeds"]
    variant_names = list(config["variants"].keys())

    all_rows = []
    for variant_name in variant_names:
        for seed in seeds:
            checkpoint_path = checkpoint_dir / f"E5_{variant_name}_resnet50_seed{seed}.pt"
            if not checkpoint_path.exists():
                print(f"[WARN] Not found checkpoint: {checkpoint_path}.")
                continue

            print(f"\n[{variant_name}, seed {seed}] Evaluating...")
            model = load_model(checkpoint_path, config["model"]["num_classes"], device)

            trashnet_result = evaluate_checkpoint(model, trashnet_loader, device)
            realwaste_result = evaluate_checkpoint(model, realwaste_loader, device)

            print(f"    TrashNet:  Macro-F1={trashnet_result['macro_f1']:.4f}")
            print(f"    RealWaste: Macro-F1={realwaste_result['macro_f1']:.4f}")

            all_rows.append({
                "variant": variant_name, "seed": seed,
                "trashnet_macro_f1": trashnet_result["macro_f1"],
                "trashnet_accuracy": trashnet_result["accuracy"],
                "trashnet_mcc": trashnet_result["mcc"],
                "realwaste_macro_f1": realwaste_result["macro_f1"],
                "realwaste_accuracy": realwaste_result["accuracy"],
                "realwaste_mcc": realwaste_result["mcc"],
            })

    if not all_rows:
        print("[ERROR] Cannot find any checkpoint — run E5_train_augmented.py first.")
        return

    raw_df = pd.DataFrame(all_rows)
    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    raw_path = table_dir / "E5_raw_results_per_seed.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[SAVED] {raw_path}")

    # ---- Summarize mean across 3 seeds for each variant ----
    summary_rows = []
    for variant_name in variant_names:
        variant_df = raw_df[raw_df["variant"] == variant_name]
        if variant_df.empty:
            continue
        summary_rows.append({
            "variant": variant_name,
            "trashnet_macro_f1_mean": variant_df["trashnet_macro_f1"].mean(),
            "trashnet_macro_f1_std": variant_df["trashnet_macro_f1"].std(),
            "realwaste_macro_f1_mean": variant_df["realwaste_macro_f1"].mean(),
            "realwaste_macro_f1_std": variant_df["realwaste_macro_f1"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ---- Read E1 and E3 ----
    e1_path = table_dir / "E1_trashnet_test_results.csv"
    e3_path = table_dir / "E3_realwaste_results.csv"

    if e1_path.exists() and e3_path.exists():
        e1_df = pd.read_csv(e1_path)
        e3_df = pd.read_csv(e3_path)
        e1_f1 = e1_df[e1_df["metric"] == "macro_f1"]["mean"].values[0]
        e3_f1 = e3_df[e3_df["metric"] == "macro_f1"]["mean"].values[0]

        summary_df["recovery_ratio_pct"] = (
            (summary_df["realwaste_macro_f1_mean"] - e3_f1) / (e1_f1 - e3_f1) * 100
        )
        summary_df["retention_ratio_pct"] = (
            summary_df["trashnet_macro_f1_mean"] / e1_f1 * 100
        )

        print(f"\n[REFERENCE] E1 (ceiling, no augmentation): {e1_f1:.4f}")
        print(f"[REFERENCE] E3 (degraded, no adaptation):   {e3_f1:.4f}")
    else:
        print("\n[WARN] Not found: E1/E3 results"
              "Run E1_evaluate_trashnet_test.py and E3_evaluate_realwaste.py first.")

    summary_path = table_dir / "E5_variants_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}\n")
    print(summary_df.to_string(index=False))

    if "recovery_ratio_pct" in summary_df.columns:
        best_variant = summary_df.loc[summary_df["recovery_ratio_pct"].idxmax()]
        print(f"\n{'=' * 60}")
        print("BEST VARIANT (by recovery ratio)")
        print(f"{'=' * 60}")
        print(f"Variant: {best_variant['variant']}")
        print(f"Recovery ratio: {best_variant['recovery_ratio_pct']:.1f}%")
        print(f"Retention ratio: {best_variant['retention_ratio_pct']:.1f}%")

if __name__ == "__main__":
    main()