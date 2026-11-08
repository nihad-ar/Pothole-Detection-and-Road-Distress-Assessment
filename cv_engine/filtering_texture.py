"""
Module: cv_engine.filtering_texture
Directly builds upon reference experiments:
 - Exp 2 (2_and_3.py):
     - Gaussian Blur (spatial smoothing)
     - Median Filtering (impulse/salt-and-pepper noise reduction)
     - Laplacian Operator (second-derivative edge detection)
     - Sobel Operator (directional gradient extraction dx=1, dy=1)
     - Canny Edge Detection (multi-stage hysteresis thresholding)
     - Local Binary Pattern (LBP) texture extraction
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def apply_gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Gaussian smoothing filter with odd kernel size."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    ksize = max(1, ksize)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter for non-linear impulse noise removal."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    ksize = max(1, ksize)
    return cv2.medianBlur(img, ksize)

def apply_laplacian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian second-order derivative filter for isotropic edge detection.
    Normalizes 64-bit float output to 8-bit unsigned display range.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap_abs = np.absolute(lap)
    lap_norm = np.uint8(255 * (lap_abs / (np.max(lap_abs) + 1e-8)))
    return lap_norm

def apply_sobel(img: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 5) -> np.ndarray:
    """
    Sobel gradient operator for directional or combined edge extraction.
    Computes horizontal (dx=1, dy=0), vertical (dx=0, dy=1), or magnitude (dx=1, dy=1).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    ksize = min(7, max(1, ksize))

    if dx > 0 and dy > 0:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_norm = np.uint8(255 * (magnitude / (np.max(magnitude) + 1e-8)))
        return sobel_norm
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx=dx, dy=dy, ksize=ksize)
        sobel_abs = np.absolute(sobel)
        sobel_norm = np.uint8(255 * (sobel_abs / (np.max(sobel_abs) + 1e-8)))
        return sobel_norm

def apply_canny(img: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Canny edge detector with dual hysteresis thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.Canny(gray, threshold1=int(threshold1), threshold2=int(threshold2))

def apply_lbp(img: np.ndarray, num_points: int = 8, radius: int = 1, method: str = "uniform") -> np.ndarray:
    """
    Local Binary Pattern (LBP) texture representation.
    Extracts micro-texture patterns invariant to monotonic illumination changes.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lbp = local_binary_pattern(gray, P=int(num_points), R=int(radius), method=method)
    # Normalize LBP output to 0..255 for visual rendering
    lbp_norm = np.uint8(255 * (lbp / (np.max(lbp) + 1e-8)))
    return lbp_norm

