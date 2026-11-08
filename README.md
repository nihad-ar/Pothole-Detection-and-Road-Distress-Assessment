# RoadScan AI: Automated Pavement Distress & Pothole Assessment System
### Computer Vision Mini-Project | Real-World Infrastructure Quality Monitoring

---

## 1. Project Abstract & Real-World Motivation

Transportation infrastructure and road networks are critical economic lifelines. Manual road surveys by civil engineers are hazardous, subjective, slow, and expensive. As a result, minor asphalt micro-cracks often go unnoticed until rainwater penetrates the sub-base, creating catastrophic potholes and structural roadway failure.

**RoadScan AI** is an automated, real-time Computer Vision system designed to assess asphalt road pavement distress directly from camera imagery (smartphone, dashcam, or inspection vehicle). 

Rather than presenting isolated lab exercises, this project synthesizes classical Computer Vision techniques into a complete, automated end-to-end civil diagnostic system that:
1. Normalizes uneven sunlight, shadows, and asphalt glare.
2. Denoises surface gravel speckles while preserving sharp fissure edges.
3. Extracts and reconstructs interconnected crack networks.
4. Profiles micro-texture using Local Binary Patterns (LBP) to identify alligator fatigue.
5. Segments dark sunken pothole cavities and measures real-world dimensions ($cm^2$).
6. Calculates an automated **Pavement Condition Index (PCI: 0–100)** and issues a prioritized civil maintenance work order.

---

## 2. Direct Mapping to Syllabus Reference Experiments

| Pipeline Stage | Computer Vision Technique | Reference Lab Experiment | Real-World Engineering Function |
|---|---|---|---|
| **Stage 1: Illumination Normalization** | Contrast-Limited Adaptive Histogram Equalization (**CLAHE**) & **Gamma Correction** | **Exp 1 & Exp 4** (`1.py`, `4.py`) | Expands shadow regions and balances bright direct sunlight vs tree/vehicle shadows across asphalt. |
| **Stage 2: Spatial Denoising** | Dual **Gaussian & Median Filtering** | **Exp 2** (`2_and_3.py`) | Gaussian filtering smooths fine bitumen grains; Median rank filtering eliminates salt-and-pepper gravel speckles without blurring crack edges. |
| **Stage 3: Crack Network Extraction** | **Sobel Gradients & Canny Hysteresis Edge Tracking** with Length-Based Noise Rejection | **Exp 2** (`2_and_3.py`) | Computes gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ and hysteresis thresholds to trace hairline and longitudinal cracks while rejecting isolated gravel noise. |
| **Stage 4: Surface Fatigue Characterization** | **Local Binary Patterns (LBP)** & Branch Intersect Analysis | **Exp 2** (`2_and_3.py`) | Quantifies micro-texture roughness and flags high-entropy alligator fatigue deterioration on the asphalt surface. |
| **Stage 5: Cavity Segmentation & Morphometry** | **Adaptive Darkness Binarization & Contour Moments** | **Exp 4** (`4.py`) + Contours | Segments dark sunken pothole craters, computes area in $px^2$ & estimated $cm^2$, perimeter, centroid $(c_x, c_y)$, and circularity ($4\pi A / P^2$). |
| **Stage 6: Decision Matrix & Reporting** | **Pavement Condition Index (PCI 0–100)** | Domain Logic | Automatically assigns condition rating (**Good, Fair, Poor, Critical Hazard**) and generates civil repair recommendations. |

---

## 3. System Architecture & Directory Structure

```
CV ASSIGNMENT/
├── app.py                      # Interactive Web App (Flask REST backend + responsive UI)
├── run_demo.py                 # Automated Batch Runner (evaluates all 4 scenarios & generates audit report)
├── generate_road_samples.py    # High-fidelity procedural road pavement texture & defect generator
├── requirements.txt            # Package dependencies
├── README.md                   # Complete documentation, theory & formulas
│
├── cv_engine/                  # Modular Processing Engine
│   ├── __init__.py             # Unified package exports
│   ├── road_analyzer.py        # Core end-to-end RoadScan inspection pipeline
│   ├── enhancement.py          # Exp 1 & 4: CLAHE, Gamma, Log, Hist Eq
│   ├── filtering_texture.py    # Exp 2: Gaussian, Median, Sobel, Canny, LBP
│   ├── features_matching.py    # Exp 3 & orb.py: Harris corners, ORB detectors, BFMatcher
│   ├── segmentation.py         # Exp 4: Otsu binarization, defect morphometry, contours
│   └── utils.py                # Base64 enc/dec, 256-bin histograms, image analytics
│
├── templates/
│   └── index.html              # Sleek Dark-Mode Dashboard (Tailwind, Lucide, Interactive Stages)
│
├── static/
│   ├── css/style.css           # Custom styling and animations
│   ├── js/app.js               # Reactive parameter controls, API client, live diagnostic rendering
│   └── sample_images/          # 4 Curated Road Test Scenarios (Pothole, Alligator, Linear, Healthy)
│
└── demo_output/                # Diagnostic outputs, intermediate stage images, and audit reports
    ├── diagnostic_road_pothole_severe.jpg
    ├── diagnostic_road_alligator_crack.jpg
    ├── diagnostic_road_linear_crack.jpg
    ├── diagnostic_road_healthy.jpg
    ├── stage1_clahe.jpg to stage6_otsu_cavity.jpg
    └── ROAD_INSPECTION_AUDIT_REPORT.md
```

---

## 4. How to Run the Working Demonstrations

### Prerequisites
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Option A: Interactive Web Application (Live Demo / Presentation)
1. In PowerShell / Terminal, navigate to `CV ASSIGNMENT` and run:
   ```bash
   cd "CV ASSIGNMENT"
   python app.py
   ```
2. Open your web browser and visit:
   ```
   http://127.0.0.1:5000
   ```
3. **Interactive Features**:
   - **One-Click Scenario Selector**: Switch between *Severe Pothole*, *Alligator Cracks*, *Linear Cracks*, and *Sound Asphalt*.
   - **Side-by-Side Diagnostic Map**: Compares input road photo against the color-coded diagnostic overlay (Potholes in Red, Cracks in Yellow).
   - **Live Metrics Scorecard**: Pavement Condition Index (PCI), Pothole Area, Crack Density %, Surface Fatigue status, and Civil Maintenance Work Order.
   - **Under-The-Hood 6-Stage Pipeline Inspector**: Interactive thumbnails showing each computer vision transformation step (CLAHE $\to$ Median $\to$ Sobel $\to$ Canny $\to$ LBP $\to$ Otsu Mask).
   - **Parameter Sensitivity Sliders**: Tune CLAHE clip limit, Gamma, Canny thresholds, and minimum pothole area in real time.
   - **Custom Upload & Live Camera**: Test with your own road photos or capture snapshots with a webcam.
   - **Download Diagnostic Image**: Export annotated forensic maps.

---

### Option B: Standalone Automated CLI Demo
To run the automated batch pipeline and generate all diagnostic artifacts:
```bash
cd "CV ASSIGNMENT"
python run_demo.py
```
This processes all 4 scenarios, logs results to the terminal, and saves diagnostic images and `ROAD_INSPECTION_AUDIT_REPORT.md` into `demo_output/`.

---

## 5. Experimental Results Across Test Scenarios

| Test Scenario | Detected Potholes | Crack Density (%) | Pavement Condition Index (PCI) | Condition Rating | Automated Civil Maintenance Work Order |
|---|---|---|---|---|---|
| **Severe Pothole Cavity** | **1 Cavity** ($28,581 px^2$, ~5.95%) | $1.39\%$ | **16 / 100** | **Critical / Failed** | ⚠️ CRITICAL HAZARD: Deep cavity depression detected. Immediate cold/hot asphalt patch filling required. Mill and overlay recommended. |
| **Alligator Fatigue Cracking** | 0 Cavities | $4.31\%$ (Dense Web) | **38 / 100** | **Critical / Failed** | ⚠️ CRITICAL HAZARD: Alligator fatigue detected. Crack routing & rubberized asphalt sealant required before pothole formation. |
| **Longitudinal & Transverse Cracks** | 0 Cavities | $0.91\%$ (Hairlines) | **71 / 100** | **Fair (Minor Distress)** | 🛠️ Preventative surface seal coating recommended to arrest hairline crack moisture penetration. |
| **Sound Intact Asphalt** | 0 Cavities | $0.08\%$ (Baseline) | **98 / 100** | **Good (Normal Service)** | ✅ Road pavement is in sound structural condition. Regular scheduled monitoring recommended. |

---

## 6. Key Mathematical Formulations & Theory (Viva Reference)

### 1. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Divides the road image into non-overlapping contextual tiles ($8 \times 8$). In each tile, the histogram is clipped at threshold $\beta = 2.0$ to prevent over-amplifying uniform aggregate noise:
$$\beta = \frac{N}{M} \left(1 + \frac{\alpha}{100} (S_{\text{max}} - 1)\right)$$
Bilinear interpolation removes artificial boundaries between tiles.

### 2. Dual-Stage Spatial Denoising
Gaussian convolution filters high-frequency bitumen variation:
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
Followed by a $5 \times 5$ Median rank filter that eliminates salt-and-pepper gravel flecks without blurring structural fissure edges:
$$I_{\text{median}}(x, y) = \text{median}\left\{ I(x + i, y + j) \mid (i, j) \in W \right\}$$

### 3. Sobel Directional Gradients & Canny Edge Tracking
Computes gradient magnitude and direction:
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I, \quad |G| = \sqrt{G_x^2 + G_y^2}$$
Non-maximum suppression thins gradients into 1-pixel wide ridges. Hysteresis thresholding ($T_{\text{low}} = 60, T_{\text{high}} = 140$) retains authentic cracks while contour length filtering ($L > 65$ px) rejects isolated gravel noise.

### 4. Pothole Cavity Depth Segmentation
Potholes are sunken dark cavities. The threshold is dynamically calibrated against the road's baseline intensity:
$$T_{\text{cavity}} = \min(50, \mu_{\text{road}} - 1.35 \cdot \sigma_{\text{road}})$$
Morphological opening ($\circ$) and closing ($\bullet$) with an elliptical structuring element ($11 \times 11$) eliminates false positives:
$$M_{\text{clean}} = (M_{\text{raw}} \circ K) \bullet K$$

### 5. Pothole Circularity & Shape Metric
To prevent long linear cracks from being classified as potholes, the circularity metric is computed:
$$C = \frac{4\pi A}{P^2}$$
Authentic circular/oval impact craters maintain $C > 0.08$ and aspect ratio $0.40 \le \frac{w}{h} \le 2.5$.
