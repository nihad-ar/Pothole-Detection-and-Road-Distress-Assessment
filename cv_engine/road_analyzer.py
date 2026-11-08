"""
RoadScan AI: Core Road Pavement Distress & Pothole Assessment Engine
Connects classical CV foundations:
 - Radiometric Enhancement: CLAHE, Gamma Correction (Exp 1 & 4)
 - Spatial Denoising: Gaussian & Median Spatial Filtering (Exp 2)
 - Edge & Structural Extraction: Sobel Gradients & Canny with Length-Based Noise Rejection (Exp 2)
 - Texture Profiling: Local Binary Patterns (LBP) & Multi-Branch Intersect Analysis (Exp 2)
 - Cavity Segmentation: Adaptive Darkness Binarization & Contour Morphometry (Exp 4)
 - Civil Decision Matrix: Pavement Condition Index (PCI 0-100) & Maintenance Work Orders
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from .enhancement import apply_clahe, apply_gamma_transform
from .filtering_texture import apply_median_filter, apply_sobel, apply_canny
from .utils import to_base64, calc_histogram, compute_image_stats

def analyze_road_pavement(
    img: np.ndarray,
    gamma: float = 0.7,
    clahe_clip: float = 2.0,
    canny_thresh1: int = 60,
    canny_thresh2: int = 140,
    min_pothole_area: int = 1500,
    max_pothole_area: int = 80000
) -> dict:
    """
    End-to-End Automated Road Pavement Inspection Pipeline:
    1. Preprocessing (CLAHE + Gamma)
    2. Spatial Denoising (Gaussian + Median)
    3. Crack Network Extraction (Sobel Magnitude + Canny with Length-Filtering)
    4. Surface Fatigue Characterization (LBP Texture + Multi-Branch Intersect Count)
    5. Pothole Cavity Segmentation (Depth/Darkness Segmentation + Elliptical Morphology)
    6. Infrastructure Rating (PCI Score 0-100 & Action Recommendation)
    """
    h, w = img.shape[:2]
    total_road_pixels = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # ---------------------------------------------------------
    # STAGE 1: Illumination Normalization & Contrast Boost
    # ---------------------------------------------------------
    gamma_corr = apply_gamma_transform(gray, gamma=gamma)
    enhanced = apply_clahe(gamma_corr, clip_limit=clahe_clip, tile_size=8)

    # ---------------------------------------------------------
    # STAGE 2: Spatial Denoising (Gaussian + Median)
    # ---------------------------------------------------------
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    denoised = cv2.medianBlur(smooth, 5)

    # ---------------------------------------------------------
    # STAGE 3: Crack Network Extraction with Noise Rejection
    # ---------------------------------------------------------
    sobel_mag = apply_sobel(denoised, dx=1, dy=1, ksize=3)
    raw_canny = apply_canny(smooth, threshold1=canny_thresh1, threshold2=canny_thresh2)

    # Filter isolated mineral noise: retain continuous crack contours with length > 65 px
    cnts, _ = cv2.findContours(raw_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_mask = np.zeros_like(raw_canny)
    total_crack_len_px = 0
    valid_crack_branches = 0

    for c in cnts:
        l = cv2.arcLength(c, False)
        if l > 65:
            cv2.drawContours(crack_mask, [c], -1, 255, 1)
            total_crack_len_px += l
            valid_crack_branches += 1

    # Connect nearby crack segments using morphological cross structuring element
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel_cross)

    crack_pixel_count = int(np.sum(crack_mask > 0))
    crack_density_pct = round((crack_pixel_count / total_road_pixels) * 100, 2)

    # ---------------------------------------------------------
    # STAGE 4: Texture Profiling (Local Binary Patterns)
    # ---------------------------------------------------------
    lbp = local_binary_pattern(denoised, P=8, R=1, method="uniform")
    # Alligator cracking produces a dense web of branches (> 40) and high crack volume
    is_fatigue_alligator = bool(valid_crack_branches >= 40 and crack_pixel_count > 8000)

    # ---------------------------------------------------------
    # STAGE 5: Pothole Cavity Segmentation & Morphometry
    # ---------------------------------------------------------
    # Physical principle: A true pothole cavity has deep darkness below the road baseline
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    darkness_cutoff = min(50.0, mean_val - 1.35 * std_val)

    cavity_raw = (gray < darkness_cutoff).astype(np.uint8) * 255

    # Clean cavity mask with elliptical morphology
    kernel_pothole = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean_pothole_mask = cv2.morphologyEx(cavity_raw, cv2.MORPH_OPEN, kernel_pothole)
    clean_pothole_mask = cv2.morphologyEx(clean_pothole_mask, cv2.MORPH_CLOSE, kernel_pothole)

    contours, _ = cv2.findContours(clean_pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_potholes = []
    total_pothole_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if min_pothole_area <= area <= max_pothole_area:
            x, y, pw, ph = cv2.boundingRect(c)
            aspect_ratio = float(pw) / ph if ph > 0 else 0
            perimeter = cv2.arcLength(c, True)

            circularity = (4 * np.pi * area) / (perimeter**2 + 1e-5)

            # Potholes are round/oval depressions (not long thin lines)
            if 0.40 <= aspect_ratio <= 2.5 and circularity > 0.08:
                total_pothole_area += area
                M = cv2.moments(c)
                cx = int(M["m10"] / (M["m00"] + 1e-5))
                cy = int(M["m01"] / (M["m00"] + 1e-5))

                approx_width_cm = round(pw * 0.08, 1)
                approx_len_cm = round(ph * 0.08, 1)
                severity = "Severe" if area > 8000 else "Moderate" if area > 3000 else "Minor"

                detected_potholes.append({
                    "id": len(detected_potholes) + 1,
                    "x": int(x), "y": int(y),
                    "width": int(pw), "height": int(ph),
                    "area_px": int(area),
                    "approx_size_cm": f"{approx_width_cm} x {approx_len_cm} cm",
                    "centroid": (cx, cy),
                    "circularity": round(circularity, 2),
                    "severity": severity
                })

    pothole_coverage_pct = round((total_pothole_area / total_road_pixels) * 100, 2)

    # ---------------------------------------------------------
    # STAGE 6: Pavement Condition Index (PCI) & Civil Verdict
    # ---------------------------------------------------------
    pci_score = 100.0

    if len(detected_potholes) > 0:
        # Potholes are catastrophic hazards
        pci_score = 30.0 - min(25.0, len(detected_potholes) * 5.0 + pothole_coverage_pct * 1.5)
    elif is_fatigue_alligator:
        # Alligator cracking represents structural base fatigue
        pci_score = 52.0 - min(15.0, (crack_pixel_count / 1500.0))
    elif crack_pixel_count > 800:
        # Linear hairline cracks
        pci_score = 76.0 - min(12.0, (crack_pixel_count / 800.0))
    else:
        # Sound healthy road
        pci_score = 98.0

    pci_score = max(0, min(100, int(round(pci_score))))

    # Classification Categories & Action Recommendations
    if pci_score >= 88:
        condition_rating = "Good (Normal Service)"
        status_color = "emerald"
        action_recommendation = "Road pavement is in sound structural condition. Regular scheduled monitoring recommended."
    elif pci_score >= 68:
        condition_rating = "Fair (Minor Distress)"
        status_color = "teal"
        action_recommendation = "Preventative surface seal coating recommended to arrest hairline crack moisture penetration."
    elif pci_score >= 42:
        condition_rating = "Poor (Warning - Moderate Damage)"
        status_color = "amber"
        action_recommendation = "Alligator fatigue detected. Crack routing & rubberized asphalt sealant required before pothole formation."
    else:
        condition_rating = "Critical / Failed (Immediate Repair)"
        status_color = "rose"
        action_recommendation = "CRITICAL HAZARD: Deep cavity depression detected. Immediate cold/hot asphalt patch filling required. Mill and overlay recommended."

    # ---------------------------------------------------------
    # STAGE 7: Generate Annotated Visual Diagnostic Map
    # ---------------------------------------------------------
    annotated = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Highlight crack network in bright yellow/amber
    crack_overlay = np.zeros_like(annotated)
    crack_overlay[crack_mask > 0] = [0, 220, 255]
    annotated = cv2.addWeighted(annotated, 0.85, crack_overlay, 0.9, 0)

    # Highlight potholes with high-visibility red bounding box & centroid crosshair
    for p in detected_potholes:
        x, y, pw, ph = p["x"], p["y"], p["width"], p["height"]
        cx, cy = p["centroid"]

        # Semi-transparent red highlight over pothole cavity
        sub_roi = annotated[y:y+ph, x:x+pw]
        red_tint = np.full_like(sub_roi, (0, 0, 220))
        cv2.addWeighted(sub_roi, 0.65, red_tint, 0.35, 0, dst=sub_roi)

        color = (0, 0, 255) if p["severity"] == "Severe" else (0, 140, 255)
        cv2.rectangle(annotated, (x, y), (x + pw, y + ph), color, 2)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)

        label = f"Pothole #{p['id']} [{p['severity']}]"
        cv2.putText(annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(annotated, p["approx_size_cm"], (x, y + ph + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Top HUD banner
    hud_bg = np.zeros((45, w, 3), dtype=np.uint8)
    annotated[0:45, 0:w] = cv2.addWeighted(annotated[0:45, 0:w], 0.3, hud_bg, 0.7, 0)
    pci_text = f"ROADSCAN AI | PCI: {pci_score}/100 [{condition_rating}]"
    cv2.putText(annotated, pci_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    lbp_norm = np.uint8(255 * (lbp / (np.max(lbp) + 1e-8)))

    return {
        "annotated_image": to_base64(annotated),
        "raw_image": to_base64(img),
        "stages": {
            "clahe_enhanced": to_base64(enhanced),
            "denoised_median": to_base64(denoised),
            "sobel_gradients": to_base64(sobel_mag),
            "crack_network": to_base64(crack_mask),
            "lbp_texture": to_base64(lbp_norm),
            "pothole_mask": to_base64(clean_pothole_mask)
        },
        "metrics": {
            "pci_score": pci_score,
            "condition_rating": condition_rating,
            "status_color": status_color,
            "pothole_count": len(detected_potholes),
            "total_pothole_area_px": int(total_pothole_area),
            "pothole_coverage_pct": pothole_coverage_pct,
            "crack_density_pct": crack_density_pct,
            "is_fatigue_alligator": is_fatigue_alligator,
            "action_recommendation": action_recommendation,
            "potholes": detected_potholes
        }
    }
