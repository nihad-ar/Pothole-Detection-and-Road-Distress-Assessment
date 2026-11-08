"""
RoadScan AI - Automated Pavement Distress & Pothole Assessment Runner
Evaluates all road pavement images located in 'Input Images/' directory,
saves annotated diagnostic outputs and intermediate CV pipeline stages to demo_output/,
and produces a Civil Engineering Inspection Audit Report.
"""

import os
import sys
import cv2

# Ensure cv_engine is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cv_engine import analyze_road_pavement, from_base64

OUTPUT_DIR = os.path.join(current_dir, "demo_output")
INPUT_DIR = os.path.join(current_dir, "Input Images")
SAMPLE_DIR = os.path.join(current_dir, "static", "sample_images")

def run_road_inspections():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 76)
    print("   ROADSCAN AI - AUTOMATED ROAD DAMAGE & POTHOLE ASSESSMENT SYSTEM")
    print("   Evaluating Images from: Input Images/")
    print("=" * 76)

    # 1. Discover target images in Input Images
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = []

    if os.path.exists(INPUT_DIR):
        for f in sorted(os.listdir(INPUT_DIR)):
            if f.lower().endswith(valid_exts):
                image_files.append((os.path.join(INPUT_DIR, f), f))

    # Fallback to SAMPLE_DIR if Input Images has no files
    if len(image_files) == 0 and os.path.exists(SAMPLE_DIR):
        print("[INFO] 'Input Images/' is empty. Falling back to sample directory.")
        for f in sorted(os.listdir(SAMPLE_DIR)):
            if f.lower().endswith(valid_exts):
                image_files.append((os.path.join(SAMPLE_DIR, f), f))

    if len(image_files) == 0:
        print("[ERROR] No image files found to inspect. Please place .jpg/.png files in 'Input Images/'.")
        return

    print(f" Found {len(image_files)} road image(s) to inspect in 'Input Images/':")
    for _, fname in image_files:
        print(f"   -> {fname}")
    print("-" * 76)

    report_rows = []

    for idx, (filepath, filename) in enumerate(image_files, 1):
        img = cv2.imread(filepath)
        if img is None:
            print(f"[WARN] Failed to read image: {filepath}")
            continue

        h, w = img.shape[:2]
        # Resize large phone/camera images for responsive inspection
        if max(h, w) > 1200:
            scale = 1200.0 / max(h, w)
            small = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            small = img.copy()

        print(f"\n[{idx}/{len(image_files)}] Analyzing: {filename} ({w}x{h} px -> {small.shape[1]}x{small.shape[0]} px)")

        # Run Core RoadScan Engine
        results = analyze_road_pavement(small)
        m = results["metrics"]

        # Save annotated diagnostic image
        safe_name = filename.replace("'", "").replace(" ", "_")
        out_filename = f"diagnostic_{safe_name}"
        if not out_filename.lower().endswith(".jpg"):
            out_filename += ".jpg"

        annotated_bgr = from_base64(results["annotated_image"])
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_filename), annotated_bgr)

        # Save intermediate pipeline stages for the first image
        if idx == 1:
            stages = results["stages"]
            cv2.imwrite(os.path.join(OUTPUT_DIR, "stage1_clahe.jpg"), from_base64(stages["clahe_enhanced"]))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "stage2_median.jpg"), from_base64(stages["denoised_median"]))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "stage3_sobel.jpg"), from_base64(stages["sobel_gradients"]))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "stage4_canny.jpg"), from_base64(stages["crack_network"]))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "stage5_lbp.jpg"), from_base64(stages["lbp_texture"]))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "stage6_otsu_cavity.jpg"), from_base64(stages["pothole_mask"]))
            print("  [OK] Saved 6 intermediate CV pipeline stages to demo_output/")

        print(f"  -> PCI Score:         {m['pci_score']} / 100 [{m['condition_rating']}]")
        print(f"  -> Potholes Detected:  {m['pothole_count']} ({m['total_pothole_area_px']:,} px area, {m['pothole_coverage_pct']}%)")
        print(f"  -> Crack Density:      {m['crack_density_pct']}% of road surface")
        print(f"  -> Surface Fatigue:    {'Detected (Alligator)' if m['is_fatigue_alligator'] else 'Normal / Intact'}")
        print(f"  -> Recommendation:     {m['action_recommendation']}")

        report_rows.append({
            "filename": filename,
            "pci": m["pci_score"],
            "rating": m["condition_rating"],
            "potholes": m["pothole_count"],
            "pothole_area": f"{m['total_pothole_area_px']:,} px ({m['pothole_coverage_pct']}%)",
            "crack_pct": f"{m['crack_density_pct']}%",
            "fatigue": "Alligator Fatigue" if m["is_fatigue_alligator"] else "Intact",
            "action": m["action_recommendation"],
            "artifact": out_filename
        })

    # Generate Markdown Report
    report_file = os.path.join(OUTPUT_DIR, "ROAD_INSPECTION_AUDIT_REPORT.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# RoadScan AI: Automated Pavement Distress & Pothole Assessment Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"This automated inspection analyzed **{len(report_rows)}** real-world road pavement images from the `Input Images/` folder using our computer vision pipeline combining CLAHE, Median spatial filtering, Sobel/Canny crack extraction, LBP fatigue texture, and Otsu pothole cavity segmentation.\n\n")
        f.write("## 2. Assessment Results & Pavement Condition Index (PCI)\n\n")
        f.write("| Image Filename | PCI Score (0-100) | Condition Rating | Potholes Detected | Total Pothole Area | Crack Density | Surface Fatigue | Civil Maintenance Work Order | Diagnostic Artifact |\n")
        f.write("|----------------|-------------------|------------------|-------------------|--------------------|---------------|-----------------|------------------------------|---------------------|\n")
        for r in report_rows:
            f.write(f"| `{r['filename']}` | **{r['pci']}/100** | {r['rating']} | {r['potholes']} | {r['pothole_area']} | {r['crack_pct']} | {r['fatigue']} | {r['action']} | [{r['artifact']}](./{r['artifact']}) |\n")

        f.write("\n## 3. Computer Vision Pipeline Mapping (Syllabus Connections)\n\n")
        f.write("1. **Stage 1: CLAHE Contrast Normalization (Exp 4 & 1)**: Overcomes uneven sunlight, shadows, and asphalt glare across the road surface.\n")
        f.write("2. **Stage 2: Median Spatial Denoising (Exp 2)**: Eliminates gravel aggregate speckle noise without blurring sharp crack fissure edges.\n")
        f.write("3. **Stage 3: Sobel & Canny Crack Extraction (Exp 2)**: Computes gradient magnitude $G = \\sqrt{G_x^2 + G_y^2}$ and non-maximum suppression to isolate linear crack networks with length-based noise rejection.\n")
        f.write("4. **Stage 4: LBP Texture Representation (Exp 2)**: Quantifies high-entropy alligator fatigue deterioration on the asphalt surface.\n")
        f.write("5. **Stage 5: Cavity Depth Segmentation (Exp 4)**: Unsupervised binarization isolates dark sunken pothole cavities with contour area and centroid tracking.\n")

    print("\n" + "=" * 76)
    print(f"   ALL {len(report_rows)} IMAGES FROM 'Input Images/' PROCESSED SUCCESSFULLY!")
    print(f"   Diagnostic Outputs: {OUTPUT_DIR}")
    print(f"   Audit Report:       {report_file}")
    print("=" * 76)

if __name__ == "__main__":
    run_road_inspections()
