"""
Module: cv_engine.segmentation
Directly builds upon reference experiments:
 - Exp 4 (4.py):
     - Otsu's Automatic Thresholding (minimizes intra-class variance)
     - Raw vs CLAHE-Enhanced Segmentation Comparison
 - Defect & Object Morphometry:
     - Contour Detection (cv2.findContours)
     - Connected Component Analysis & Bounding Boxes
     - Defect / Object Area, Perimeter, and Centroid Extraction
"""

import cv2
import numpy as np
from .enhancement import apply_clahe

def apply_otsu_threshold(img: np.ndarray, use_clahe_first: bool = False, clip_limit: float = 2.0) -> tuple[np.ndarray, float]:
    """
    Apply Otsu's Binarization.
    Optionally pre-enhances using CLAHE (as explored in 4.py).
    Returns (binary_image, calculated_otsu_threshold).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    if use_clahe_first:
        gray = apply_clahe(gray, clip_limit=clip_limit)
        
    otsu_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(otsu_val)

def compare_otsu_enhancement(img: np.ndarray, clip_limit: float = 2.0) -> dict:
    """
    Direct replication and enhancement of Experiment 4 (4.py):
    Compares Otsu's threshold on original image vs Otsu's on CLAHE-enhanced image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe_img = apply_clahe(gray, clip_limit=clip_limit)

    thresh_orig, bin_orig = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_clahe, bin_clahe = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "original_gray": gray,
        "clahe_enhanced": clahe_img,
        "otsu_original": bin_orig,
        "otsu_clahe": bin_clahe,
        "threshold_original": float(thresh_orig),
        "threshold_clahe": float(thresh_clahe)
    }

def detect_defects_and_objects(
    img: np.ndarray,
    min_area: int = 50,
    max_area: int = 50000,
    use_clahe: bool = True,
    invert_binary: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Industrial inspection & defect segmentation pipeline:
    1. Grayscale & CLAHE contrast enhancement
    2. Otsu thresholding
    3. Morphological opening/closing to clean noise
    4. Contour detection & bounding box annotation
    5. Morphometric measurement (area, centroid, aspect ratio)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    if use_clahe:
        processed = apply_clahe(gray, clip_limit=2.5)
    else:
        processed = gray

    # Otsu thresholding
    thresh_flag = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
    _, binary = cv2.threshold(processed, 0, 255, thresh_flag + cv2.THRESH_OTSU)

    # Morphological noise filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detected_items = []
    total_defect_area = 0

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            total_defect_area += area
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Moments for centroid
            M = cv2.moments(c)
            cx = int(M["m10"] / (M["m00"] + 1e-5))
            cy = int(M["m01"] / (M["m00"] + 1e-5))

            # Draw bounding box and centroid
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"#{len(detected_items) + 1}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            detected_items.append({
                "id": len(detected_items) + 1,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area_px": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
                "centroid": (cx, cy)
            })

    total_pixels = gray.shape[0] * gray.shape[1]
    coverage_pct = round((total_defect_area / total_pixels) * 100, 2)

    stats = {
        "count": len(detected_items),
        "total_area_px": int(total_defect_area),
        "coverage_percentage": coverage_pct,
        "items": detected_items[:20]  # top 20 items in summary
    }

    return annotated, stats

