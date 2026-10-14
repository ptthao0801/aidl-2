"""
E1_train_resnet50.py
======================
Train ResNet50 on 3 seeds (42, 123, 456)

Chạy: python -m src.experiments.E1_ceiling.E1_train_resnet50
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "E1_config.yaml"

UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
CLASS_TO_IDX = {c: i for i, c in enumerate(UNIFIED_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class WasteImageDataset(Dataset):
    """Read image + label from file CSV generated from build_splits.py."""

    def __init__(self, csv_path: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = BASE / row["filepath"]
        img = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(image_size: int, augment: dict):
    train_transform_list = [
        transforms.Resize((image_size, image_size)),
    ]
    if augment.get("random_horizontal_flip", False):
        train_transform_list.append(transforms.RandomHorizontalFlip())
    if augment.get("random_rotation_degrees", 0) > 0:
        train_transform_list.append(transforms.RandomRotation(augment["random_rotation_degrees"]))
    if augment.get("color_jitter", False):
        strength = augment.get("color_jitter_strength", 0.1)
        train_transform_list.append(
            transforms.ColorJitter(brightness=strength, contrast=strength, saturation=strength)
        )
    train_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train_transform = transforms.Compose(train_transform_list)

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def compute_class_weights(csv_path: Path) -> torch.Tensor:
    """
    Handle class imbalance
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    weights = []
    for c in UNIFIED_CLASSES:
        freq = counts.get(c, 1)
        weights.append(1.0 / freq)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(UNIFIED_CLASSES)  # normalize
    return weights


def build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    if backbone != "resnet50":
        raise ValueError(f"E1 chỉ hỗ trợ resnet50, nhận: {backbone}")

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    # Thay fully-connected cuối cho khớp 5 lớp
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, dataloader, device) -> dict:
    """Return Macro-F1 + predictions/labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"macro_f1": macro_f1, "preds": all_preds, "labels": all_labels}


def train_one_seed(seed: int, config: dict, log_lines: list) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"E1 — Training ResNet50, seed={seed}, device={device}")
    print(f"{'=' * 60}")
    log_lines.append(f"\n=== SEED {seed} ===")

    image_size = config["data"]["image_size"]
    train_transform, eval_transform = build_transforms(image_size, config["augmentation"])

    train_csv = BASE / config["data"]["train_csv"]
    val_csv = BASE / config["data"]["val_csv"]

    train_ds = WasteImageDataset(train_csv, transform=train_transform)
    val_ds = WasteImageDataset(val_csv, transform=eval_transform)

    batch_size = config["training"]["batch_size"]
    # num_workers cao hơn để tránh CPU nghẽn khi GPU xử lý nhanh hơn tốc độ
    # load+decode+augment ảnh. pin_memory + non_blocking giúp copy CPU->GPU
    # bất đồng bộ, không block GPU chờ. persistent_workers tránh overhead
    # khởi tạo lại worker process ở đầu mỗi epoch.
    num_workers = config["training"].get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
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
    best_val_loss = float("inf")       # when to stop
    best_val_f1_so_far = -1.0          # checkpoint
    epochs_no_improve = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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

        # ---- Tách biệt 2 vai trò ----
        # (a) CHECKPOINT
        if val_macro_f1 > best_val_f1_so_far:
            best_val_f1_so_far = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
            log_lines.append(f"    -> New best val_macro_f1: {best_val_f1_so_far:.4f} (checkpoint updated)")

        # (b) EARLY STOP
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            msg = f"[EARLY STOP] Dừng tại epoch {epoch} (val_loss không cải thiện {patience} epoch liên tiếp)"
            print(msg)
            log_lines.append(msg)
            break


    model.load_state_dict(best_model_state)

    checkpoint_dir = BASE / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"E1_resnet50_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[SAVED] Checkpoint: {checkpoint_path} (best val_macro_f1={best_val_f1_so_far:.4f})")
    log_lines.append(f"[SAVED] {checkpoint_path} (best val_macro_f1={best_val_f1_so_far:.4f})")

    # Training history
    table_dir = BASE / config["output"]["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    history_df["seed"] = seed
    history_csv = table_dir / f"E1_training_history_seed{seed}.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"[SAVED] Training history: {history_csv}")

    return {"seed": seed, "best_val_macro_f1": best_val_f1_so_far, "checkpoint": str(checkpoint_path)}


def main():
    config = load_config()
    log_lines = ["E1 TRAINING LOG — ResNet50 Ceiling Performance", f"Config: {CONFIG_PATH}"]

    seeds = config["seeds"]
    results = []

    start_time = time.time()
    for seed in seeds:
        result = train_one_seed(seed, config, log_lines)
        results.append(result)
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"E1 TRAINING COMPLETED — total time: {elapsed / 60:.1f} mins")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  Seed {r['seed']}: best val_macro_f1 = {r['best_val_macro_f1']:.4f}")

    val_f1_scores = [r["best_val_macro_f1"] for r in results]
    print(f"\n  Mean val Macro-F1: {np.mean(val_f1_scores):.4f} +/- {np.std(val_f1_scores):.4f}")

    log_lines.append(f"\nTotal time: {elapsed / 60:.1f} mins")
    log_lines.append(f"Mean val Macro-F1: {np.mean(val_f1_scores):.4f} +/- {np.std(val_f1_scores):.4f}")

    log_dir = BASE / config["output"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "E1_train_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n[SAVED] Full log: {log_path}")


if __name__ == "__main__":
    main()