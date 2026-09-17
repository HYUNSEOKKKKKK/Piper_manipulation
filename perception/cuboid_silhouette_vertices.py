"""Step A only (per plan): mask -> stable 4/6-vertex silhouette polygon.
No pose computation here on purpose -- just verify the vertices land on
the real cuboid silhouette corners before building anything on top of it.

hull (cuboid is convex) -> approxPolyDP, sweeping epsilon from fine to
coarse and taking the first result that lands on 4 or 6 vertices.
"""
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def mask_to_polygon(mask, target_counts=(4, 6), n_steps=100, ratio_range=(0.002, 0.05)):
    """Returns (poly Nx2 int array, N) for N in target_counts, or (None, 0)
    if no epsilon in the swept range lands on exactly 4 or 6 vertices."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for ratio in np.linspace(*ratio_range, n_steps):
        poly = cv2.approxPolyDP(hull, ratio * peri, True)
        if len(poly) in target_counts:
            return poly.reshape(-1, 2), len(poly)
    return None, 0


def segment_yellow_box(bgr, valid_depth):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 60, 30]), np.array([40, 255, 255])) > 0
    return _largest_component(mask & valid_depth)


def segment_postit(bgr, valid_depth):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 150
    return _largest_component(mask & valid_depth)


def _largest_component(mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def draw_polygon(img, poly, color=(0, 0, 255)):
    vis = img.copy()
    if poly is None:
        cv2.putText(vis, "FAILED: no 4/6-vertex fit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return vis
    n = len(poly)
    for i in range(n):
        p0, p1 = tuple(poly[i]), tuple(poly[(i + 1) % n])
        cv2.line(vis, p0, p1, color, 2, cv2.LINE_AA)
    for i, p in enumerate(poly):
        cv2.circle(vis, tuple(p), 5, (0, 255, 255), -1)
        cv2.putText(vis, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"n={n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    out_dir = os.path.join(HERE, "cuboid_silhouette_out")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [
        ("box_frames", segment_yellow_box),
        ("postit_frames", segment_postit),
    ]

    counts = {4: 0, 6: 0, "other/fail": 0}
    for folder, segment_fn in datasets:
        frame_dirs = sorted(glob.glob(os.path.join(HERE, folder, "frame_*")))
        for fd in frame_dirs:
            name = f"{folder}_{os.path.basename(fd)}"
            rgb = cv2.imread(os.path.join(fd, "rgb.png"))
            depth = np.load(os.path.join(fd, "depth.npy"))
            valid = depth > 0.05

            mask = segment_fn(rgb, valid)
            if mask is None:
                print(f"{name}: segmentation failed")
                counts["other/fail"] += 1
                continue

            poly, n = mask_to_polygon(mask)
            if poly is None:
                print(f"{name}: no 4/6-vertex polygon found in sweep")
                counts["other/fail"] += 1
            else:
                print(f"{name}: {n}-vertex polygon found")
                counts[n] += 1

            vis = draw_polygon(rgb, poly)
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), vis)

    print(f"\nsummary: {counts}")
    print(f"overlays -> {out_dir}")


if __name__ == "__main__":
    main()
