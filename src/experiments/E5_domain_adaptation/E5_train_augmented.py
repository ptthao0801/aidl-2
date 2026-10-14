"""
E5_train_augmented.py
========================
Chạy: python -m src.experiments.E5_domain_adaptation.E5_train_augmented
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.experiments.E1_ceiling.E1_train_resnet50 import (
    WasteImageDataset, UNIFIED_CLASSES, CLASS_TO_IDX,
    IMAGENET_MEAN, IMAGENET_STD, compute_class_weights, build_model
)

BASE = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "E5_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_variant_transform(image_size: int, variant_cfg: dict):
    transform_list = [transforms.Resize((image_size, image_size))]

    b = variant_cfg.get("color_jitter_brightness", 0)
    c = variant_cfg.get("color_jitter_contrast", 0)
    s = variant_cfg.get("color_jitter_saturation", 0)
    h = variant_cfg.get("color_jitter_hue", 0)
    if any([b, c, s, h]):
        transform_list.append(transforms.ColorJitter(
            brightness=b, contrast=c, saturation=s, hue=h,
        ))

    if variant_cfg.get("random_perspective", False):
        transform_list.append(transforms.RandomPerspective(
            distortion_scale=variant_cfg.get("random_perspective_distortion", 0.3),
            p=0.5,
        ))

    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.RandomRotation(15))

    if variant_cfg.get("gaussian_blur", False):
        transform_list.append(transforms.GaussianBlur(
            kernel_size=variant_cfg.get("gaussian_blur_kernel", 5),
            sigma=tuple(variant_cfg.get("gaussian_blur_sigma", [0.1, 2.0])),
        ))

    transform_list.append(transforms.ToTensor())

    if variant_cfg.get("random_erasing", False):
        transform_list.append(transforms.RandomErasing(
            p=variant_cfg.get("random_erasing_prob", 0.3),
            scale=tuple(variant_cfg.get("random_erasing_scale", [0.02, 0.15])),
        ))

    transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    train_transform = transforms.Compose(transform_list)

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def train_one_seed(variant_name: str, seed: int, config: dict, log_lines: list) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"E5-{variant_name.upper()} — Training ResNet50, seed={seed}, device={device}")
    print(f"{'=' * 60}")
    log_lines.append(f"\n=== VARIANT {variant_name}, SEED {seed} ===")

    image_size = config["data"]["image_size"]
    variant_cfg = config["variants"][variant_name]
    train_transform, eval_transform = build_variant_transform(image_size, variant_cfg)

    train_csv = BASE / config["data"]["train_csv"]
    val_csv = BASE / config["data"]["val_csv"]

    train_ds = WasteImageDataset(train_csv, transform=train_transform)
    val_ds = WasteImageDataset(val_csv, transform=eval_transform)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0),
    )

    model = build_model(
        config["model"]["backbone"], config["model"]["num_classes"], config["model"]["pretrained"]
    ).to(device)

    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    patience = config["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    best_val_f1_so_far = -1.0
    epochs_no_improve = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        scheduler.step()

        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        val_loss = val_running_loss / len(val_ds)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        log_line = (f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | val_macro_f1={val_macro_f1:.4f}")
        print(log_line)
        log_lines.append(log_line)

        if val_macro_f1 > best_val_f1_so_far:
            best_val_f1_so_far = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            msg = f"[EARLY STOP] Dừng tại epoch {epoch}"
            print(msg)
            log_lines.append(msg)
            break

    model.load_state_dict(best_model_state)

    checkpoint_dir = BASE / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"E5_{variant_name}_resnet50_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SAVED] Checkpoint: {checkpoint_path} (best val_macro_f1={best_val_f1_so_far:.4f})")
    log_lines.append(f"[SAVED] {checkpoint_path} (best val_macro_f1={best_val_f1_so_far:.4f})")

    return {"variant": variant_name, "seed": seed, "best_val_macro_f1": best_val_f1_so_far}


def main():
    config = load_config()
    log_lines = ["E5 TRAINING LOG — Augmentation-Based Domain Adaptation", f"Config: {CONFIG_PATH}"]

    seeds = config["seeds"]
    all_variant_names = list(config["variants"].keys())

    checkpoint_dir = BASE / config["output"]["checkpoint_dir"]
    variant_names = []
    for v in all_variant_names:
        already_done = all(
            (checkpoint_dir / f"E5_{v}_resnet50_seed{s}.pt").exists() for s in seeds
        )
        if already_done:
            print(f"[SKIP] '{v}' Already have {len(seeds)} checkpoint.")
        else:
            variant_names.append(v)

    if not variant_names:
        print("[INFO] All variants already have complete checkpoints. Nothing to train.")
        return

    print(f"[INFO] Variants to be trained: {variant_names}")
    all_results = []

    start_time = time.time()
    for variant_name in variant_names:
        for seed in seeds:
            result = train_one_seed(variant_name, seed, config, log_lines)
            all_results.append(result)
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"E5 TRAINING COMPLETED ({len(variant_names)} new variants x {len(seeds)} seed) — {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame(all_results)
    for variant_name in variant_names:
        variant_scores = results_df[results_df["variant"] == variant_name]["best_val_macro_f1"]
        print(f"  {variant_name}: mean val Macro-F1 = {variant_scores.mean():.4f} "
              f"+/- {variant_scores.std():.4f}")

    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / "E5_training_summary.csv"
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        results_df = pd.concat([old_df, results_df], ignore_index=True)
    results_df.to_csv(summary_path, index=False)

    log_dir = BASE / config["output"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "E5_train_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n[SAVED] Full log: {log_dir / 'E5_train_log.txt'}")


if __name__ == "__main__":
    main()