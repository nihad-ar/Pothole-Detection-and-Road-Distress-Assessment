"""
RoadScan AI: Automated Pavement Distress & Pothole Damage Assessment
A real-world Computer Vision Mini-Project for Road Infrastructure Quality Monitoring.
Directly loads and evaluates images from 'Input Images/' directory.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from cv_engine import (
    analyze_road_pavement,
    to_base64,
    from_base64,
    calc_histogram,
    compute_image_stats
)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Input Images")
SAMPLE_DIR = os.path.join(BASE_DIR, "static", "sample_images")

os.makedirs(INPUT_DIR, exist_ok=True)

def get_available_images():
    """Scan 'Input Images' folder, prioritizing featured custom images, followed by dataset images."""
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    featured = []
    dataset = []

    if os.path.exists(INPUT_DIR):
        for f in sorted(os.listdir(INPUT_DIR)):
            if f.lower().endswith(valid_exts):
                clean_label = f.replace("_", " ").replace("-", " ")
                for ext in valid_exts:
                    if clean_label.lower().endswith(ext):
                        clean_label = clean_label[:-len(ext)]

                is_featured = "delhi" in f.lower() or "bengaluru" in f.lower() or "broken" in f.lower() or "deformed" in f.lower()
                entry = {
                    "filename": f,
                    "source": "input_folder",
                    "is_featured": is_featured,
                    "label": clean_label[:45] + ("..." if len(clean_label) > 45 else ""),
                    "full_path": os.path.join(INPUT_DIR, f)
                }
                if is_featured:
                    featured.append(entry)
                else:
                    dataset.append(entry)

    # Secondary fallback if Input Images is completely empty
    if len(featured) == 0 and len(dataset) == 0 and os.path.exists(SAMPLE_DIR):
        for f in sorted(os.listdir(SAMPLE_DIR)):
            if f.lower().endswith(valid_exts):
                dataset.append({
                    "filename": f,
                    "source": "sample_folder",
                    "is_featured": True,
                    "label": f,
                    "full_path": os.path.join(SAMPLE_DIR, f)
                })

    return featured + dataset

def load_input_image(req_data) -> tuple[np.ndarray, str]:
    """Load image from Base64 string or filename from Input Images."""
    img_data = req_data.get("image")
    sample_name = req_data.get("sample_name")

    if sample_name:
        path_input = os.path.join(INPUT_DIR, os.path.basename(sample_name))
        path_sample = os.path.join(SAMPLE_DIR, os.path.basename(sample_name))
        target_path = path_input if os.path.exists(path_input) else (path_sample if os.path.exists(path_sample) else None)

        if target_path and os.path.exists(target_path):
            img = cv2.imread(target_path)
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > 1200:
                    scale = 1200.0 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                return img, os.path.basename(target_path)

    if img_data:
        return from_base64(img_data), "uploaded_custom_image.jpg"

    # Fallback to first available
    avail = get_available_images()
    if avail:
        fallback_path = avail[0]["full_path"]
        img = cv2.imread(fallback_path)
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > 1200:
                scale = 1200.0 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            return img, avail[0]["filename"]

    return None, ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/input_images/<path:filename>")
def serve_input_image(filename):
    """Directly serve images from Input Images folder."""
    return send_from_directory(INPUT_DIR, filename)

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return all available images from the 'Input Images' folder."""
    images = get_available_images()
    return jsonify({
        "total_count": len(images),
        "featured": [img for img in images if img.get("is_featured")],
        "dataset_sample": [img for img in images if not img.get("is_featured")][:250],
        "all_filenames": [img["filename"] for img in images]
    })

@app.route("/api/inspect_road", methods=["POST"])
def inspect_road():
    """
    Main Real-World Inspection Endpoint:
    Runs the full RoadScan AI pipeline on the requested road image.
    """
    data = request.get_json(force=True)
    img, active_filename = load_input_image(data)

    if img is None:
        return jsonify({"error": "No valid road pavement image found in 'Input Images/' or upload"}), 400

    gamma = float(data.get("gamma", 0.7))
    clahe_clip = float(data.get("clahe_clip", 2.0))
    canny1 = int(data.get("canny1", 60))
    canny2 = int(data.get("canny2", 140))
    min_area = int(data.get("min_pothole_area", 1500))

    results = analyze_road_pavement(
        img,
        gamma=gamma,
        clahe_clip=clahe_clip,
        canny_thresh1=canny1,
        canny_thresh2=canny2,
        min_pothole_area=min_area
    )

    stats = compute_image_stats(img)
    hist = calc_histogram(img)
    results["image_stats"] = stats
    results["histogram"] = hist
    results["active_filename"] = active_filename

    return jsonify(results)

if __name__ == "__main__":
    print("=" * 70)
    print("  ROADSCAN AI - AUTOMATED ROAD DAMAGE & POTHOLE ASSESSMENT SYSTEM")
    print("  Dataset Folder: Input Images/")
    print("  Server Running: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=True)
