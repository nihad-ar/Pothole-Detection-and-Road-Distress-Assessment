"""
Module: cv_engine.face_biometrics
Directly builds upon reference experiment:
 - Exp 6 (6.py): Face detection, LBP texture extraction, and facial verification.
Supports dual backend:
 1. Haar Cascades (when cv2.CascadeClassifier is available)
 2. Kovac-Chai-Ngan YCrCb Chrominance Segmentation & Aspect-Ratio Filter (universal across OpenCV 3/4/5)
"""

import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern

def detect_faces(img: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> tuple[np.ndarray, list]:
    """
    Detect faces using either Haar Cascades or YCrCb Skin Chrominance Segmentation.
    Annotates detected faces with bounding boxes and confidence markers.
    Returns (annotated_image, face_boxes).
    """
    annotated = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results = []

    # Backend 1: Haar Cascades if supported in current OpenCV build
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                cascade = cv2.CascadeClassifier(cascade_path)
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=float(scale_factor),
                    minNeighbors=int(min_neighbors),
                    minSize=(40, 40)
                )
                if len(faces) > 0:
                    for i, (x, y, w, h) in enumerate(faces):
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            annotated,
                            f"Face #{i+1} ({w}x{h})",
                            (x, max(15, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )
                        results.append({"face_id": i + 1, "x": int(x), "y": int(y), "width": int(w), "height": int(h), "backend": "Haar Cascade"})
                    return annotated, results
            except Exception:
                pass  # Fall through to Chrominance detector

    # Backend 2: YCrCb Chrominance & Morphological Aspect Ratio Face Detector
    # Classic Chai-Ngan / Kovac Skin Color Model for face localization
    img_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h_img, w_img = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # Standard human skin chrominance bounds
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Morphological filtering to clean noise and bridge facial segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_face_area = int(w_img * h_img * 0.015)
    max_face_area = int(w_img * h_img * 0.75)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_face_area <= area <= max_face_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            # Human face bounding box aspect ratio is typically between 0.55 and 1.25
            if 0.55 <= aspect_ratio <= 1.25 and y + h <= h_img:
                candidates.append((area, x, y, w, h))

    # Sort largest candidates first (main faces)
    candidates.sort(key=lambda item: item[0], reverse=True)

    for i, (_, x, y, w, h) in enumerate(candidates[:5]):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Face #{i+1} ({w}x{h})",
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        results.append({
            "face_id": i + 1,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "backend": "YCrCb Chrominance"
        })

    # If no skin region passed the strict bounds, provide fallback center ROI
    if len(results) == 0:
        cx, cy = w_img // 2, h_img // 2
        fw, fh = int(w_img * 0.35), int(h_img * 0.35)
        x, y = max(0, cx - fw // 2), max(0, cy - fh // 2)
        cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "Target ROI",
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )
        results.append({"face_id": 1, "x": x, "y": y, "width": fw, "height": fh, "backend": "Center Prior"})

    return annotated, results

def extract_face_lbp_descriptor(face_crop: np.ndarray, num_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Extract normalized LBP spatial histogram for facial biometric comparison (Exp 6 concept).
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    resized = cv2.resize(gray, (100, 100))
    lbp = local_binary_pattern(resized, P=num_points, R=radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=int(num_points * (num_points - 1) + 3), range=(0, int(num_points * (num_points - 1) + 2)))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def verify_faces_lbp(face1: np.ndarray, face2: np.ndarray, threshold: float = 0.25) -> dict:
    """
    Verify whether two face crops belong to the same identity using Chi-Square distance between LBP histograms.
    """
    desc1 = extract_face_lbp_descriptor(face1)
    desc2 = extract_face_lbp_descriptor(face2)

    # Chi-square distance
    chi_dist = 0.5 * np.sum(((desc1 - desc2) ** 2) / (desc1 + desc2 + 1e-10))
    verified = bool(chi_dist < threshold)

    return {
        "verified": verified,
        "chi_square_distance": round(float(chi_dist), 4),
        "match_confidence": round(max(0.0, min(100.0, (1.0 - chi_dist / (threshold * 2)) * 100)), 1)
    }
