# RoadScan AI: Automated Pavement Distress & Pothole Assessment Report

## 1. Executive Summary
This automated inspection analyzed **2** real-world road pavement images from the `Input Images/` folder using our computer vision pipeline combining CLAHE, Median spatial filtering, Sobel/Canny crack extraction, LBP fatigue texture, and Otsu pothole cavity segmentation.

## 2. Assessment Results & Pavement Condition Index (PCI)

| Image Filename | PCI Score (0-100) | Condition Rating | Potholes Detected | Total Pothole Area | Crack Density | Surface Fatigue | Civil Maintenance Work Order | Diagnostic Artifact |
|----------------|-------------------|------------------|-------------------|--------------------|---------------|-----------------|------------------------------|---------------------|
| `Broken_Roads_in_India's_Capital_New_Delhi.jpg` | **5/100** | Critical / Failed (Immediate Repair) | 5 | 41,148 px (3.81%) | 8.96% | Alligator Fatigue | CRITICAL HAZARD: Deep cavity depression detected. Immediate cold/hot asphalt patch filling required. Mill and overlay recommended. | [diagnostic_Broken_Roads_in_Indias_Capital_New_Delhi.jpg](./diagnostic_Broken_Roads_in_Indias_Capital_New_Delhi.jpg) |
| `Roads_deformed_T_munnekollala_Bengaluru.jpg` | **38/100** | Critical / Failed (Immediate Repair) | 0 | 0 px (0.0%) | 2.64% | Alligator Fatigue | CRITICAL HAZARD: Deep cavity depression detected. Immediate cold/hot asphalt patch filling required. Mill and overlay recommended. | [diagnostic_Roads_deformed_T_munnekollala_Bengaluru.jpg](./diagnostic_Roads_deformed_T_munnekollala_Bengaluru.jpg) |

## 3. Computer Vision Pipeline Mapping (Syllabus Connections)

1. **Stage 1: CLAHE Contrast Normalization (Exp 4 & 1)**: Overcomes uneven sunlight, shadows, and asphalt glare across the road surface.
2. **Stage 2: Median Spatial Denoising (Exp 2)**: Eliminates gravel aggregate speckle noise without blurring sharp crack fissure edges.
3. **Stage 3: Sobel & Canny Crack Extraction (Exp 2)**: Computes gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ and non-maximum suppression to isolate linear crack networks with length-based noise rejection.
4. **Stage 4: LBP Texture Representation (Exp 2)**: Quantifies high-entropy alligator fatigue deterioration on the asphalt surface.
5. **Stage 5: Cavity Depth Segmentation (Exp 4)**: Unsupervised binarization isolates dark sunken pothole cavities with contour area and centroid tracking.
