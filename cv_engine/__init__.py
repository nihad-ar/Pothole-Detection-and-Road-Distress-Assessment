"""
ForensiCV & RoadScan AI - Core Engine Package
Unifies foundational CV techniques for automated road infrastructure & pavement damage inspection.
"""

from .enhancement import (
    apply_negative,
    apply_log_transform,
    apply_gamma_transform,
    apply_histogram_equalization,
    apply_clahe
)
from .filtering_texture import (
    apply_gaussian_blur,
    apply_median_filter,
    apply_laplacian,
    apply_sobel,
    apply_canny,
    apply_lbp
)
from .features_matching import (
    apply_harris_corners,
    extract_orb_keypoints,
    match_features_orb
)
from .segmentation import (
    apply_otsu_threshold,
    compare_otsu_enhancement,
    detect_defects_and_objects
)
from .face_biometrics import (
    detect_faces,
    verify_faces_lbp
)
from .road_analyzer import (
    analyze_road_pavement
)
from .utils import (
    to_base64,
    from_base64,
    calc_histogram,
    compute_image_stats
)
