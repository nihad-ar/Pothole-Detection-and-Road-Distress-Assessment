import base64
import cv2
import numpy as np

def to_base64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode an OpenCV BGR or Grayscale image into a Base64 data URL string."""
    if img is None:
        return ""
    success, buffer = cv2.imencode(ext, img)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"

def from_base64(b64_string: str) -> np.ndarray:
    """Decode a Base64 data URL or raw Base64 string into an OpenCV BGR image."""
    if not b64_string:
        return None
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def calc_histogram(img: np.ndarray) -> dict:
    """Compute 256-bin normalized histogram for grayscale or BGR images."""
    if img is None:
        return {}
    
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return {"gray": hist.flatten().tolist()}
    elif len(img.shape) == 3:
        b_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
        r_hist = cv2.calcHist([img], [2], None, [256], [0, 256])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return {
            "gray": gray_hist.flatten().tolist(),
            "b": b_hist.flatten().tolist(),
            "g": g_hist.flatten().tolist(),
            "r": r_hist.flatten().tolist(),
        }
    return {}

def compute_image_stats(img: np.ndarray) -> dict:
    """Calculate key photographic & quality metrics: Mean Brightness, RMS Contrast, Entropy, Dynamic Range."""
    if img is None:
        return {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    
    # Entropy calculation
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    hist_nonzero = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))
    
    # Laplacian blur variance (Sharpness / Focus metric)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "dimensions": f"{w} x {h} px",
        "mean_brightness": round(mean_val, 2),
        "rms_contrast": round(std_val, 2),
        "dynamic_range": f"[{min_val}, {max_val}]",
        "entropy": round(entropy, 3),
        "sharpness_score": round(laplacian_var, 2)
    }

