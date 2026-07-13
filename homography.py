import numpy as np
import cv2
import json

INPUT_FILE = "homo_apr2.txt"
OUTPUT_JSON = "homography_23Jun.json"


def parse_points(line):
    """
    Parse a line like:
    1000,500 2000,520 2200,1000 900,950
    """
    pts = []
    tokens = line.strip().split()
    for token in tokens:
        x_str, y_str = token.split(",")
        pts.append((float(x_str), float(y_str)))
    return pts


def image_to_world(pt, H):
    """Apply homography to convert image point -> world meters"""
    px, py = pt
    p = np.array([px, py, 1.0], dtype=float)
    wp = H.dot(p)
    wp = wp / wp[2]
    return float(wp[0]), float(wp[1])


def main():

    with open(INPUT_FILE, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 3:
        raise ValueError("Input file must contain at least 3 lines.")

    # -----------------------------------
    # Line 1 → Homography image points
    # Line 2 → Homography world points
    # Line 3 → ROI1
    # Line 4 → (Optional) ROI2
    # Line 5 → (Optional) Conflict points
    # -----------------------------------

    image_points = parse_points(lines[0])
    world_points = parse_points(lines[1])
    roi1_points = parse_points(lines[2])

    roi2_points = []
    conflict_image_points = []

    if len(lines) >= 4:
        roi2_points = parse_points(lines[3])

    if len(lines) >= 5:
        conflict_image_points = parse_points(lines[4])

    if len(image_points) < 4:
        raise ValueError("At least 4 homography image points required.")

    if len(image_points) != len(world_points):
        raise ValueError("Image points and world points must match in count.")

    # Convert to numpy
    pts_img_np = np.array(image_points, dtype=np.float32)
    pts_world_np = np.array(world_points, dtype=np.float32)

    # Compute homography
    H, mask = cv2.findHomography(pts_img_np, pts_world_np, cv2.RANSAC)

    if H is None:
        raise RuntimeError("Homography computation failed.")

    print("Homography computed successfully.")
    print("H matrix:\n", H)

    # -----------------------------------
    # Convert conflict image points → world coordinates
    # -----------------------------------
    conflict_world_points = []

    for pt in conflict_image_points:
        world_pt = image_to_world(pt, H)
        conflict_world_points.append(world_pt)

    if conflict_world_points:
        print("\nConflict points (world meters):")
        for cp in conflict_world_points:
            print(cp)

    # Prepare output JSON
    output = {
        "H": H.tolist(),
        "image_points": image_points,
        "world_points": world_points,
        "roi1_image_points": roi1_points,
        "roi2_image_points": roi2_points,
        "conflict_world_points": conflict_world_points
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()