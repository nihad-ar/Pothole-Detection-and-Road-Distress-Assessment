"""
Module: cv_engine.enhancement
Directly builds upon reference experiments:
 - Exp 1 (1.py): Negative, Logarithmic transformation, Power-law (Gamma) correction, Global Histogram Equalization.
 - Exp 4 (4.py): CLAHE (Contrast-Limited Adaptive Histogram Equalization) and comparative enhancement.
"""

import cv2
import numpy as np

def apply_negative(img: np.ndarray) -> np.ndarray:
    """Invert pixel intensities (255 - pixel). Preserves color channels if BGR."""
    return 255 - img

def apply_log_transform(img: np.ndarray, c: float = None) -> np.ndarray:
    """
    Logarithmic Transformation: s = c * log(1 + r).
    Expands dark pixels, compresses high intensity values.
    """
    is_color = len(img.shape) == 3
    if is_color:
        # Convert to HSV or YCrCb, transform value channel to preserve chrominance
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)
        log_y = np.log(1.0 + y)
        scale = 255.0 / np.max(log_y) if c is None else c
        ycrcb[:, :, 0] = np.uint8(np.clip(log_y * scale, 0, 255))
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        gray_f = img.astype(np.float32)
        log_t = np.log(1.0 + gray_f)
        scale = 255.0 / np.max(log_t) if c is None else c
        return np.uint8(np.clip(log_t * scale, 0, 255))

def apply_gamma_transform(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Power-Law / Gamma Transformation: s = c * (r / 255.0) ^ gamma * 255.
    gamma < 1.0: Brightens shadows (under-exposed scenes)
    gamma > 1.0: Darkens highlights (over-exposed scenes)
    """
    gamma = max(0.01, float(gamma))
    inv_gamma = gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization.
    If image is BGR, equalizes luminance channel (Y in YCrCb) to avoid color distortion.
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Limits noise amplification in flat regions while maximizing local contrast.
    """
    tile_size = max(2, int(tile_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_size, tile_size))
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)

