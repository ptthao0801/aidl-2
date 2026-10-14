# Extract Local Binary Pattern features
"""
lbp_features.py
=================
"""

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage import img_as_ubyte


LBP_RADIUS_1 = 1
LBP_POINTS_1 = 8 * LBP_RADIUS_1  
LBP_RADIUS_2 = 3
LBP_POINTS_2 = 8 * LBP_RADIUS_2  

FEATURE_DIM_1 = LBP_POINTS_1 + 1  
FEATURE_DIM_2_REDUCED = 8         

N_BINS_1 = LBP_POINTS_1 + 2   
N_BINS_2_REDUCED = 8

FEATURE_DIM = N_BINS_1 + N_BINS_2_REDUCED  


def extract_lbp_features(image_rgb: np.ndarray) -> np.ndarray:

    gray_float = rgb2gray(image_rgb)  
    gray = img_as_ubyte(gray_float)

    # --- LBP radius=1 ---
    lbp1 = local_binary_pattern(gray, LBP_POINTS_1, LBP_RADIUS_1, method="uniform")
    hist1, _ = np.histogram(lbp1, bins=N_BINS_1, range=(0, N_BINS_1))
    hist1 = hist1.astype(np.float64)
    hist1 = hist1 / (hist1.sum() + 1e-7)

    # --- LBP radius=3 ---
    lbp2 = local_binary_pattern(gray, LBP_POINTS_2, LBP_RADIUS_2, method="uniform")
    n_bins_2_full = LBP_POINTS_2 + 2  
    hist2_full, _ = np.histogram(lbp2, bins=n_bins_2_full, range=(0, n_bins_2_full))
    hist2_full = hist2_full.astype(np.float64)
    hist2_full = hist2_full / (hist2_full.sum() + 1e-7)

    hist2_reduced = np.array([
        hist2_full[i::N_BINS_2_REDUCED].sum()
        for i in range(N_BINS_2_REDUCED)
    ])
    hist2_reduced = hist2_reduced / (hist2_reduced.sum() + 1e-7)

    result = np.concatenate([hist1, hist2_reduced])
    assert result.shape[0] == FEATURE_DIM, (
        f"LBP feature dim mismatch: got {result.shape[0]}, expected {FEATURE_DIM}"
    )
    return result


if __name__ == "__main__":
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    features = extract_lbp_features(dummy_image)
    print(f"LBP feature shape: {features.shape}")  
    print(f"Sum hist1 (~1.0): {features[:N_BINS_1].sum():.3f}")
    print(f"Sum hist2_reduced (~1.0): {features[N_BINS_1:].sum():.3f}")