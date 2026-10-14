# Extract Histogram of Oriented Gradients features
"""
hog_features.py
=================
"""

import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize


HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (56, 56)   
HOG_CELLS_PER_BLOCK = (2, 2)      
HOG_CELLS_PER_BLOCK_FINAL = (1, 1) 
HOG_PIXELS_PER_CELL_81D = (74, 74) 
FEATURE_DIM = 81


def extract_hog_features(image_rgb: np.ndarray) -> np.ndarray:

    gray = rgb2gray(image_rgb)

    if gray.shape != (224, 224):
        gray = resize(gray, (224, 224), anti_aliasing=True)

    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL_81D,
        cells_per_block=(1, 1), 
        block_norm="L2-Hys",
        feature_vector=True,
    )

    assert features.shape[0] == FEATURE_DIM, (
        f"HOG feature dim mismatch: got {features.shape[0]}, expected {FEATURE_DIM}. "
        f"Check HOG_PIXELS_PER_CELL_81D if the input size changes."
    )

    return features


if __name__ == "__main__":
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    features = extract_hog_features(dummy_image)
    print(f"HOG feature shape: {features.shape}") 