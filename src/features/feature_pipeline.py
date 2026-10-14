# Combine features into a single feature vector
"""
feature_pipeline.py
=====================

Chạy: python -m src.features.feature_pipeline
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.hsv_features import extract_hsv_histogram, FEATURE_DIM as HSV_DIM
from src.features.lbp_features import extract_lbp_features, FEATURE_DIM as LBP_DIM
from src.features.hog_features import extract_hog_features, FEATURE_DIM as HOG_DIM

BASE = Path(__file__).resolve().parents[2]

TOTAL_FEATURE_DIM = HSV_DIM + LBP_DIM + HOG_DIM  # 96 + 18 + 81 = 195

FEATURE_GROUP_SLICES = {
    "hsv": slice(0, HSV_DIM),                          # cột 0-95
    "lbp": slice(HSV_DIM, HSV_DIM + LBP_DIM),           # cột 96-113
    "hog": slice(HSV_DIM + LBP_DIM, TOTAL_FEATURE_DIM),  # cột 114-194
}


def extract_all_features(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    hsv_feat = extract_hsv_histogram(image_bgr)
    lbp_feat = extract_lbp_features(image_rgb)
    hog_feat = extract_hog_features(image_rgb)

    return np.concatenate([hsv_feat, lbp_feat, hog_feat]) 


def build_feature_matrix(csv_path: Path, cache_path: Path = None) -> tuple:
    if cache_path and cache_path.exists():
        print(f"[CACHE] Load feature matrix có sẵn: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["filepaths"]

    df = pd.read_csv(csv_path)
    n = len(df)
    X = np.zeros((n, TOTAL_FEATURE_DIM), dtype=np.float64)
    y = df["label"].values
    filepaths = df["filepath"].values

    print(f"[EXTRACT] Extract feature for {n} images from {csv_path.name}...")
    failed_indices = []
    for i in tqdm(range(n), desc=f"Feature extraction ({csv_path.stem})"):
        img_path = BASE / filepaths[i]
        try:
            X[i] = extract_all_features(img_path)
        except Exception as e:
            print(f"\n[ERROR] Error extracting feature {img_path}: {e}")
            failed_indices.append(i)

    if failed_indices:
        print(f"[WARN] {len(failed_indices)} failed image.")
        mask = np.ones(n, dtype=bool)
        mask[failed_indices] = False
        X, y, filepaths = X[mask], y[mask], filepaths[mask]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, filepaths=filepaths)
        print(f"[SAVED] Cache: {cache_path}")

    return X, y, filepaths


if __name__ == "__main__":
    print(f"Total feature dimension: {TOTAL_FEATURE_DIM} "
          f"(HSV={HSV_DIM} + LBP={LBP_DIM} + HOG={HOG_DIM})")
    print(f"Feature group slices: {FEATURE_GROUP_SLICES}")

    # Quick test
    import tempfile
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, dummy_image)
        features = extract_all_features(Path(tmp.name))
        print(f"\nTest 1 image: feature shape = {features.shape} "
              f"(expect {TOTAL_FEATURE_DIM},)")