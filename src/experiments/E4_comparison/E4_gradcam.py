# Grad-CAM for ResNet50 layer4
"""
E4_gradcam.py
================
Chạy: python -m src.experiments.E4_comparison.E4_gradcam
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.experiments.E1_ceiling.E1_train_resnet50 import (
    build_transforms, UNIFIED_CLASSES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD
)

BASE = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "E4_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(UNIFIED_CLASSES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def find_examples(csv_path: Path, model, transform, device, want_correct: bool, n_samples: int):
    df = pd.read_csv(csv_path)
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    found = []
    for _, row in df.sample(frac=1.0, random_state=42).iterrows():  # shuffle
        img_path = BASE / row["filepath"]
        true_label = row["label"]

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
        pred_label = idx_to_class[pred_idx]

        is_correct = (pred_label == true_label)
        if is_correct == want_correct:
            found.append((img_path, true_label, pred_label))
        if len(found) >= n_samples:
            break

    return found


def generate_gradcam_overlay(model, target_layer, img_path: Path, transform, device):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    img_tensor = transform(img).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return overlay


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("E4 — GRAD-CAM (ResNet50)")
    print("=" * 60)

    checkpoint_path = BASE / "outputs" / "checkpoints" / "E1_resnet50_seed42.pt"
    if not checkpoint_path.exists():
        print(f"[ERROR] Cannot find checkpoint: {checkpoint_path}")
        return

    model = load_model(checkpoint_path, device)
    target_layer = model.layer4[-1] 

    _, eval_transform = build_transforms(224, {})

    trashnet_test_csv = BASE / "data" / "splits" / "trashnet_test.csv"
    realwaste_csv = BASE / "data" / "splits" / "realwaste_test.csv"

    n_samples = config["gradcam"]["n_samples_per_panel"]

    print("\n[INFO] Finding illustration images for the 3 columns...")
    col1_examples = find_examples(trashnet_test_csv, model, eval_transform, device,
                                   want_correct=True, n_samples=n_samples)
    col2_examples = find_examples(realwaste_csv, model, eval_transform, device,
                                   want_correct=True, n_samples=n_samples)
    col3_examples = find_examples(realwaste_csv, model, eval_transform, device,
                                   want_correct=False, n_samples=n_samples)

    print(f"    Column 1 (TrashNet, correct): {len(col1_examples)} images")
    print(f"    Column 2 (RealWaste, correct): {len(col2_examples)} images")
    print(f"    Column 3 (RealWaste, wrong): {len(col3_examples)} images")

    n_rows = max(len(col1_examples), len(col2_examples), len(col3_examples))
    if n_rows == 0:
        print("[ERROR] No images found. Check again on checkpoint/data.")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = [
        "TrashNet — Correct\n(In-Domain)",
        "RealWaste — Correct\n(Cross-Domain)",
        "RealWaste — Incorrect\n(Domain Shift Error)",
    ]

    for col_idx, (examples, title) in enumerate(
        zip([col1_examples, col2_examples, col3_examples], col_titles)
    ):
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            if row_idx < len(examples):
                img_path, true_label, pred_label = examples[row_idx]
                overlay = generate_gradcam_overlay(model, target_layer, img_path,
                                                    eval_transform, device)
                ax.imshow(overlay)
                status = "✓" if true_label == pred_label else "✗"
                ax.set_xlabel(f"True: {true_label} | Pred: {pred_label} {status}", fontsize=9)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Grad-CAM: Where Does ResNet50 Look Under Domain Shift?",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    figure_dir = BASE / config["output"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "E4_gradcam_3col.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    main()