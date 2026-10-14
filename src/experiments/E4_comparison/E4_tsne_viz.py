# t-SNE latent feature space visualization
"""
E4_tsne_viz.py
==================
Chạy: python -m src.experiments.E4_comparison.E4_tsne_viz
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import models

from src.experiments.E1_ceiling.E1_train_resnet50 import (
    WasteImageDataset, build_transforms, UNIFIED_CLASSES, CLASS_TO_IDX
)

BASE = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "E4_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_gap_features(model, dataloader, device):
    features_list, labels_list = [], []

    activation = {}

    def hook_fn(module, input, output):
        activation["gap_output"] = output.detach()

    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images) 
            gap_features = activation["gap_output"].squeeze(-1).squeeze(-1)  
            features_list.append(gap_features.cpu().numpy())
            labels_list.extend(labels.numpy())

    hook_handle.remove()

    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    return features, labels


def subsample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    if len(X) <= max_samples:
        return X, y
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    return X[indices], y[indices]


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsne_cfg = config["tsne"]

    print("=" * 60)
    print("E4 — t-SNE VISUALIZATION (ResNet50 feature space)")
    print("=" * 60)

    checkpoint_path = BASE / "outputs" / "checkpoints" / "E1_resnet50_seed42.pt"
    if not checkpoint_path.exists():
        print(f"[ERROR] Cannot find checkpoint: {checkpoint_path}")
        return

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(UNIFIED_CLASSES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    _, eval_transform = build_transforms(224, {})

    print("\n[INFO] Extracting GAP feature (2048-dim) from TrashNet test...")
    trashnet_ds = WasteImageDataset(BASE / "data" / "splits" / "trashnet_test.csv",
                                     transform=eval_transform)
    trashnet_loader = DataLoader(trashnet_ds, batch_size=32, shuffle=False, num_workers=2)
    X_trashnet, y_trashnet_idx = extract_gap_features(model, trashnet_loader, device)

    print("[INFO] Extracting GAP feature (2048-dim) from RealWaste...")
    realwaste_ds = WasteImageDataset(BASE / "data" / "splits" / "realwaste_test.csv",
                                      transform=eval_transform)
    realwaste_loader = DataLoader(realwaste_ds, batch_size=32, shuffle=False, num_workers=2)
    X_realwaste, y_realwaste_idx = extract_gap_features(model, realwaste_loader, device)

    max_samples = tsne_cfg["max_samples_per_domain"]
    seed = tsne_cfg["random_seed"]
    X_trashnet, y_trashnet_idx = subsample(X_trashnet, y_trashnet_idx, max_samples, seed)
    X_realwaste, y_realwaste_idx = subsample(X_realwaste, y_realwaste_idx, max_samples, seed)

    print(f"[INFO] After subsample: TrashNet={len(X_trashnet)}, RealWaste={len(X_realwaste)}")

    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    y_trashnet_class = np.array([idx_to_class[i] for i in y_trashnet_idx])
    y_realwaste_class = np.array([idx_to_class[i] for i in y_realwaste_idx])


    X_combined = np.concatenate([X_trashnet, X_realwaste], axis=0)
    domain_labels = np.array(["TrashNet"] * len(X_trashnet) + ["RealWaste"] * len(X_realwaste))
    class_labels = np.concatenate([y_trashnet_class, y_realwaste_class])

    print(f"\n[INFO] Running t-SNE on {len(X_combined)} points (perplexity="
      f"{tsne_cfg['perplexity']})... this may take a few minutes.")
    tsne = TSNE(n_components=2, perplexity=tsne_cfg["perplexity"],
                random_state=seed, init="pca")
    embedding = tsne.fit_transform(X_combined)

    # ---- Plotting ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    class_colors = plt.cm.tab10(np.linspace(0, 1, len(UNIFIED_CLASSES)))
    for i, c in enumerate(UNIFIED_CLASSES):
        mask = class_labels == c
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                         label=c, alpha=0.5, s=15, color=class_colors[i])
    axes[0].set_title("Colored by Class", fontsize=12, fontweight="bold")
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")

    domain_colors = {"TrashNet": "#4C72B0", "RealWaste": "#C44E52"}
    for domain_name, color in domain_colors.items():
        mask = domain_labels == domain_name
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1],
                         label=domain_name, alpha=0.5, s=15, color=color)
    axes[1].set_title("Colored by Domain", fontsize=12, fontweight="bold")
    axes[1].legend(markerscale=2, fontsize=9)
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    fig.suptitle("t-SNE of ResNet50 Feature Space (2048-dim GAP output)",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    figure_dir = BASE / config["output"]["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "E4_tsne_class_vs_domain.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    main()