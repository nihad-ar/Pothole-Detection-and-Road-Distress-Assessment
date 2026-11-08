# ForensiCV Studio - Mini-Project Execution Report

## Overview
This automated demonstration successfully executed all computer vision pipeline stages, integrating syllabus experiments 1 through 6 into a cohesive digital forensics and quality inspection suite.

## Pipeline Execution Summary

| Module Ref | Operation / Stage | Mathematical Formula / Principle | Output Image Artifact |
|------------|-------------------|----------------------------------|-----------------------|
| **Exp 1** | Image Negative (Inversion) | `s = 255 - r` | [01_negative.jpg](./01_negative.jpg) |
| **Exp 1** | Logarithmic Transform | `s = c · log(1+r)` | [02_log_transform.jpg](./02_log_transform.jpg) |
| **Exp 1** | Power-Law Gamma (γ=0.5) | `s = c · r^γ` | [03_gamma_0.5.jpg](./03_gamma_0.5.jpg) |
| **Exp 1** | Histogram Equalization | `T(r) = (L-1)·CDF` | [04_hist_equalized.jpg](./04_hist_equalized.jpg) |
| **Exp 4** | CLAHE Adaptive Contrast | `Local Grid CDF Clip` | [05_clahe_contrast.jpg](./05_clahe_contrast.jpg) |
| **Exp 2** | Gaussian Blur (5x5) | `Spatial Bell Kernel` | [06_gaussian_blur.jpg](./06_gaussian_blur.jpg) |
| **Exp 2** | Median Filter (5x5) | `Non-linear Rank Denoise` | [07_median_filter.jpg](./07_median_filter.jpg) |
| **Exp 2** | Laplacian Operator | `Isotropic 2nd-Deriv` | [08_laplacian_edge.jpg](./08_laplacian_edge.jpg) |
| **Exp 2** | Sobel Combined Gradients | `G = sqrt(Gx^2 + Gy^2)` | [09_sobel_gradient.jpg](./09_sobel_gradient.jpg) |
| **Exp 2** | Canny Edge Detection | `Hysteresis (100, 200)` | [10_canny_edges.jpg](./10_canny_edges.jpg) |
| **Exp 2** | LBP Texture Representation | `Uniform (P=8, R=1)` | [11_lbp_texture.jpg](./11_lbp_texture.jpg) |
| **Exp 3** | Harris Corners (7101 pts) | `R = det(M) - k(Tr(M))^2` | [12_harris_corners.jpg](./12_harris_corners.jpg) |
| **Exp 3** | ORB Keypoints (500 pts) | `oFAST + rBRIEF` | [13_orb_keypoints.jpg](./13_orb_keypoints.jpg) |
| **orb.py** | ORB BFMatcher (115 matches) | `NORM_HAMMING + CrossCheck` | [14_orb_feature_matching.jpg](./14_orb_feature_matching.jpg) |
| **Exp 4** | 4-Quadrant Replication Plate | `4.py Exact Reproduction` | [15_exp4_quadrant_replication.jpg](./15_exp4_quadrant_replication.jpg) |
| **Exp 4+Contours** | Defect Morphometry (103 items) | `Otsu + Contours + Moments` | [16_defect_morphometry.jpg](./16_defect_morphometry.jpg) |
| **Exp 6** | Haar Face Detection (2 face) | `Viola-Jones Cascades` | [17_face_detection.jpg](./17_face_detection.jpg) |

## Key Analytical Findings
- **Exp 4 Contrast Impact**: Raw Otsu threshold was 115.0, while CLAHE Otsu threshold adapted to 123.0, demonstrating superior local foreground isolation.
- **Exp 3 Feature Matching**: ORB identified 115 valid descriptor matches with an average Hamming distance of 1.43 (High Confidence).
- **Defect Morphometry**: Successfully localized 103 distinct surface regions accounting for 15.73% of the image plane.
