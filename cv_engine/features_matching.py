"""
Module: cv_engine.features_matching
Directly builds upon reference experiments:
 - Exp 3 (2_and_3.py):
     - Harris Corner Detection (cornerness response R = det(M) - k * (trace(M))^2)
     - ORB Keypoint Detection & Compute (oFAST keypoints + rBRIEF descriptors)
 - orb.py:
     - Query vs Scene Feature Matching using BFMatcher with NORM_HAMMING
     - Distance-based match sorting and cv2.drawMatches visualization
"""

import cv2
import numpy as np

def apply_harris_corners(
    img: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01
) -> tuple[np.ndarray, int]:
    """
    Harris Corner Detection.
    Marks detected corner interest points with red overlays on the image.
    Returns (annotated_image, corner_count).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    color_output = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ksize = ksize if ksize % 2 == 1 else ksize + 1
    harris_response = cv2.cornerHarris(
        gray,
        blockSize=int(block_size),
        ksize=int(ksize),
        k=float(k)
    )

    # Dilate response for clearer visual marks
    harris_dilated = cv2.dilate(harris_response, None)
    
    threshold = float(threshold_ratio) * harris_dilated.max()
    corner_mask = harris_dilated > threshold
    corner_count = int(np.sum(corner_mask))

    # Mark corners with glowing red markers
    color_output[corner_mask] = [0, 0, 255]
    
    # Also draw small circles on strong corners for visibility
    y_coords, x_coords = np.where(corner_mask)
    # Subsample if too dense
    step = max(1, len(x_coords) // 200)
    for x, y in zip(x_coords[::step], y_coords[::step]):
        cv2.circle(color_output, (int(x), int(y)), 3, (0, 0, 255), 1)

    return color_output, corner_count

def extract_orb_keypoints(img: np.ndarray, n_features: int = 500) -> tuple[np.ndarray, int, list]:
    """
    Extract ORB (Oriented FAST and Rotated BRIEF) keypoints.
    Draws rich keypoints with orientation circles and scale.
    Returns (annotated_image, keypoint_count, keypoint_coords).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    orb = cv2.ORB_create(nfeatures=int(n_features))
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    drawn_img = cv2.drawKeypoints(
        img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    kp_data = [{"x": round(kp.pt[0], 1), "y": round(kp.pt[1], 1), "size": round(kp.size, 1), "angle": round(kp.angle, 1)} for kp in keypoints[:50]]
    return drawn_img, len(keypoints), kp_data

def match_features_orb(
    query_img: np.ndarray,
    scene_img: np.ndarray,
    n_features: int = 500,
    top_matches: int = 30
) -> tuple[np.ndarray, dict]:
    """
    Match ORB features between Query image and Scene image using BFMatcher with NORM_HAMMING.
    Directly parallels the implementation in orb.py and 2_and_3.py.
    """
    q_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY) if len(query_img.shape) == 3 else query_img
    s_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY) if len(scene_img.shape) == 3 else scene_img

    orb = cv2.ORB_create(nfeatures=int(n_features))
    kp1, des1 = orb.detectAndCompute(q_gray, None)
    kp2, des2 = orb.detectAndCompute(s_gray, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        # Fallback if no descriptors found
        combined = np.hstack([
            cv2.resize(query_img, (400, 300)),
            cv2.resize(scene_img, (400, 300))
        ])
        return combined, {"matches_found": 0, "status": "Insufficient keypoints for matching"}

    # BFMatcher with NORM_HAMMING and cross-check verification
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    limit = min(int(top_matches), len(matches))
    good_matches = matches[:limit]

    # Calculate match statistics
    avg_distance = float(np.mean([m.distance for m in good_matches])) if good_matches else 0.0

    matched_visual = cv2.drawMatches(
        query_img, kp1,
        scene_img, kp2,
        good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    stats = {
        "query_keypoints": len(kp1),
        "scene_keypoints": len(kp2),
        "total_matches": len(matches),
        "displayed_matches": limit,
        "avg_hamming_distance": round(avg_distance, 2),
        "match_quality": "High Confidence" if avg_distance < 35 else "Moderate Confidence" if avg_distance < 55 else "Low Confidence"
    }

    return matched_visual, stats

