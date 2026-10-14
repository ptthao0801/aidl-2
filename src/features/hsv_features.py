# Extract HSV histogram features
"""
hsv_features.py
=================
"""

import cv2
import numpy as np


N_BINS = 32
N_CHANNELS = 3
FEATURE_DIM = N_BINS * N_CHANNELS  # 96


def extract_hsv_histogram(image_bgr: np.ndarray) -> np.ndarray:

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist_features = []
    ranges = [(0, 180), (0, 256), (0, 256)] 

    for channel_idx in range(N_CHANNELS):
        channel_range = ranges[channel_idx]
        hist = cv2.calcHist(
            [image_hsv], [channel_idx], None,
            [N_BINS], channel_range
        )
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        hist_features.append(hist)

    return np.concatenate(hist_features)  # shape (96,)


if __name__ == "__main__":
    # Quick test
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    features = extract_hsv_histogram(dummy_image)
    print(f"HSV feature shape: {features.shape}") 
    print(f"Sum per channel: "
          f"{features[:32].sum():.3f}, {features[32:64].sum():.3f}, {features[64:].sum():.3f}")