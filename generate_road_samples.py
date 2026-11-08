"""
Generates high-fidelity realistic asphalt road pavement test samples:
1. Pothole with depth crater & jagged rim (sample_pothole.jpg)
2. Alligator / Fatigue crack network (sample_alligator_crack.jpg)
3. Longitudinal & Transverse hairline cracks (sample_linear_crack.jpg)
4. Smooth, healthy asphalt road (sample_healthy_road.jpg)
"""

import cv2
import numpy as np
import os
import random

def create_base_asphalt(width=800, height=600, base_tone=75) -> np.ndarray:
    """Generate realistic asphalt aggregate texture using multi-scale noise."""
    # Base gray layer
    asphalt = np.full((height, width, 3), base_tone, dtype=np.float32)

    # Coarse aggregate (mineral gravel chunks)
    coarse_noise = np.random.normal(0, 18, (height // 4, width // 4, 3)).astype(np.float32)
    coarse_noise = cv2.resize(coarse_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    asphalt += coarse_noise

    # Medium aggregate
    med_noise = np.random.normal(0, 12, (height // 2, width // 2, 3)).astype(np.float32)
    med_noise = cv2.resize(med_noise, (width, height), interpolation=cv2.INTER_LINEAR)
    asphalt += med_noise

    # Fine bitumen sand grain
    fine_noise = np.random.normal(0, 8, (height, width, 3)).astype(np.float32)
    asphalt += fine_noise

    # Random mineral flecks (bright white/yellow aggregate specks)
    num_specks = int(width * height * 0.003)
    speck_x = np.random.randint(0, width, num_specks)
    speck_y = np.random.randint(0, height, num_specks)
    asphalt[speck_y, speck_x] = np.random.uniform(130, 185, (num_specks, 3))

    # Dark tar flecks
    num_dark = int(width * height * 0.005)
    dark_x = np.random.randint(0, width, num_dark)
    dark_y = np.random.randint(0, height, num_dark)
    asphalt[dark_y, dark_x] = np.random.uniform(30, 48, (num_dark, 3))

    return np.clip(asphalt, 0, 255).astype(np.uint8)

def draw_crack_branch(img, start_pt, length=300, angle=-np.pi/2, thickness=2):
    """Draw realistic jagged crack using brownian random walk."""
    curr_x, curr_y = start_pt
    pts = [(int(curr_x), int(curr_y))]
    h, w = img.shape[:2]

    for _ in range(length):
        angle += np.random.uniform(-0.35, 0.35)
        step = np.random.uniform(2.0, 5.0)
        curr_x += np.cos(angle) * step
        curr_y += np.sin(angle) * step

        if not (10 < curr_x < w - 10 and 10 < curr_y < h - 10):
            break
        pts.append((int(curr_x), int(curr_y)))

        # Sub-branching probability
        if random.random() < 0.02 and length > 40:
            branch_angle = angle + random.choice([-1, 1]) * random.uniform(0.5, 1.2)
            draw_crack_branch(img, (curr_x, curr_y), length=length // 2, angle=branch_angle, thickness=max(1, thickness - 1))

    for i in range(len(pts) - 1):
        color = int(np.random.uniform(15, 30))
        cv2.line(img, pts[i], pts[i+1], (color, color, color), thickness, cv2.LINE_AA)

def generate_samples(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    random.seed(42)

    # 1. Healthy Pavement (No defects)
    healthy = create_base_asphalt(800, 600, base_tone=80)
    # Add subtle tire tread track wear
    tire_overlay = np.zeros_like(healthy, dtype=np.float32)
    cv2.rectangle(tire_overlay, (150, 0), (320, 600), (10, 10, 10), -1)
    cv2.rectangle(tire_overlay, (480, 0), (650, 600), (10, 10, 10), -1)
    tire_overlay = cv2.GaussianBlur(tire_overlay, (91, 91), 0)
    healthy = np.clip(healthy.astype(np.float32) - tire_overlay * 0.4, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "road_healthy.jpg"), healthy)

    # 2. Linear & Transverse Cracks
    cracked = create_base_asphalt(800, 600, base_tone=78)
    draw_crack_branch(cracked, (400, 580), length=160, angle=-np.pi/2, thickness=3)
    draw_crack_branch(cracked, (250, 450), length=110, angle=-0.4, thickness=2)
    draw_crack_branch(cracked, (550, 300), length=90, angle=-2.2, thickness=2)
    cracked = cv2.GaussianBlur(cracked, (3, 3), 0.3)
    cv2.imwrite(os.path.join(output_dir, "road_linear_crack.jpg"), cracked)

    # 3. Alligator / Fatigue Cracking (Interconnected polygonal web)
    alligator = create_base_asphalt(800, 600, base_tone=72)
    # Generate grid of seed points for Voronoi-like crack network
    grid_pts = []
    for gy in range(80, 520, 65):
        for gx in range(100, 700, 65):
            jx = gx + random.randint(-22, 22)
            jy = gy + random.randint(-22, 22)
            grid_pts.append((jx, jy))

    for pt in grid_pts:
        # Connect to closest neighbors with jagged lines
        for other in grid_pts:
            dist = np.hypot(pt[0] - other[0], pt[1] - other[1])
            if 30 < dist < 95 and random.random() < 0.45:
                # Jagged connector
                mid_x = (pt[0] + other[0]) / 2 + random.randint(-8, 8)
                mid_y = (pt[1] + other[1]) / 2 + random.randint(-8, 8)
                c_val = random.randint(18, 32)
                pts_arr = np.array([pt, (mid_x, mid_y), other], np.int32)
                cv2.polylines(alligator, [pts_arr], False, (c_val, c_val, c_val), 2, cv2.LINE_AA)

    alligator = cv2.GaussianBlur(alligator, (3, 3), 0.3)
    cv2.imwrite(os.path.join(output_dir, "road_alligator_crack.jpg"), alligator)

    # 4. Severe Pothole with Crater Depth & Jagged Edges
    pothole_img = create_base_asphalt(800, 600, base_tone=76)
    h, w = 600, 800
    center_x, center_y = 420, 310

    # Draw irregular pothole crater polygon
    angles = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    radii = [random.uniform(90, 140) for _ in range(28)]
    poly_pts = []
    for a, r in zip(angles, radii):
        px = int(center_x + r * np.cos(a))
        py = int(center_y + r * 0.75 * np.sin(a))  # flattened perspective
        poly_pts.append((px, py))
    poly_pts = np.array(poly_pts, np.int32)

    # Deep dark cavity mask
    cavity_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cavity_mask, [poly_pts], 255)

    # Distressed broken rim surrounding the pothole
    rim_mask = cv2.dilate(cavity_mask, np.ones((15, 15), np.uint8)) - cavity_mask

    # Pothole depth gradient (darker towards center)
    dist_map = cv2.distanceTransform(cavity_mask, cv2.DIST_L2, 5)
    max_d = dist_map.max() if dist_map.max() > 0 else 1.0
    depth_factor = (dist_map / max_d)[:, :, None]

    # Cavity interior texture: dark rocky bottom with loose asphalt aggregate
    cavity_interior = create_base_asphalt(800, 600, base_tone=28)
    cavity_interior = np.clip(cavity_interior * (1.0 - 0.45 * depth_factor), 15, 60).astype(np.uint8)

    # Blend cavity into road
    cavity_3ch = cavity_mask[:, :, None] / 255.0
    pothole_img = (pothole_img * (1 - cavity_3ch) + cavity_interior * cavity_3ch).astype(np.uint8)

    # Add cracked aggregate stress fractures radiating out from pothole rim
    for i in range(0, len(poly_pts), 3):
        pt = poly_pts[i]
        out_angle = np.arctan2(pt[1] - center_y, pt[0] - center_x)
        draw_crack_branch(pothole_img, (pt[0], pt[1]), length=random.randint(40, 85), angle=out_angle, thickness=2)

    # Edge shadow inside the rim
    shadow_mask = cv2.erode(cavity_mask, np.ones((7, 7), np.uint8))
    rim_shadow = cv2.subtract(cavity_mask, shadow_mask)
    pothole_img[rim_shadow > 0] = np.clip(pothole_img[rim_shadow > 0].astype(np.float32) * 0.5, 10, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, "road_pothole_severe.jpg"), pothole_img)

    print(f"Generated 4 realistic road pavement test samples in: {output_dir}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "static", "sample_images")
    generate_samples(out)

