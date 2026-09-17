"""Prototype: closed-form cuboid pose from KNOWN L,W,H via orthogonal
plane-fitting on the box's own visible faces -- no ICP, no reference
template, no per-frame shape estimation. Each frame is an independent
absolute measurement (like resectioning), not a relative update against
a prior frame, so there's no drift to manage in the first place.

Tests accuracy (visual overlay + face-agreement residual) and speed
against box_frames/ captures, with the real (ruler-measured) box dims.
Segmentation here is a simple, throwaway table-plane+height threshold
(not MobileSAMv2) since only the pose math is under test.
"""
import glob
import json
import os
import time
from itertools import permutations

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = np.array(sorted([0.077, 0.053, 0.030], reverse=True))  # L>=W>=H, meters


def pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def ransac_plane(pts, n_trials=200, thresh=0.004, rng=None):
    if len(pts) < 20:
        return None
    rng = rng or np.random.default_rng(0)
    idx_all = np.arange(len(pts))
    best = None
    for _ in range(n_trials):
        i = rng.choice(idx_all, 3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p0
        inliers = np.abs(pts @ n + d) < thresh
        cnt = int(inliers.sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    if best is None or best[0] < 20:
        return None
    _, n, d = best
    inliers = np.abs(pts @ n + d) < thresh
    P = pts[inliers]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    d = -n @ c
    inliers = np.abs(pts @ n + d) < thresh
    return n, d, inliers


def segment_box(pc, valid, bgr):
    """RGB color-threshold (bright yellow box vs near-black table) +
    largest-connected-component -- stands in for MobileSAMv2 so only the
    pose math below is under test. Deliberately NOT depth-height-threshold
    (journal #16 already found that class of approach too noise-fragile;
    the real pipeline finds objects from RGB and only reads depth *within*
    that mask, so this mirrors that division of labor).
    Returns (points, 2D bool mask) or (None, None)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 60, 30])  # V lowered 60->30: the box's own shadowed side
    upper = np.array([40, 255, 255])  # face dips to ~V=49-56, a stricter floor cut it out entirely
    color_mask = cv2.inRange(hsv, lower, upper) > 0
    obj_mask = (color_mask & valid).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    if n_labels <= 1:
        return None, None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = labels == biggest
    return pc[keep], keep


def render_box_silhouette(shape, center, R, dims, fx, fy, cx, cy):
    """Rasterized silhouette of the box (convex hull of its 8 projected
    corners -- exact for a convex solid like a cuboid, from any angle)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = center + (signs * (np.array(dims) / 2.0)) @ R.T
    proj = np.array([[fx * X / Z + cx, fy * Y / Z + cy] for X, Y, Z in corners], dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(proj)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    return mask.astype(bool)


def mask_iou(a, b):
    union = (a | b).sum()
    return (a & b).sum() / union if union > 0 else 0.0


def box_sdf_error(pts, center, R, dims):
    """Mean |signed distance to the candidate box's surface| over EVERY
    point, not just whichever ones RANSAC clustered into a face -- a
    scattered handful of correct points still pulls the score the right
    way, so this doesn't need a minimum-cluster-size the way per-face
    RANSAC does. Standard box SDF (as used in raymarching): 0 exactly on
    the surface, >0 outside, <0 inside."""
    local = (pts - center) @ R
    half = np.array(dims) / 2.0
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return np.mean(np.abs(outside + inside))


def fit_cuboid_known_dims(pts, dims, mask, fx, fy, cx, cy, prev_axis_map=None):
    """pts: Nx3 box points, camera frame, meters -- ALL valid depth points
    in the object's mask, not pre-clustered. dims: the 3 known side
    lengths, any order.

    Finds 1-3 orthogonal box faces via RANSAC to pin down the rotation's
    direction(s), then resolves (a) which known dimension belongs on which
    axis, and (b) with only 1 face, the remaining in-plane rotation too --
    by scoring every candidate on TWO signals together:
      - box_sdf_error() against ALL of `pts`, not just whatever RANSAC
        clustered -- survives sparse/patchy depth (glossy-surface dropout,
        see box_frames/ findings), but by itself is nearly blind to
        in-plane rotation/sizing: most points sit deep inside one dominant
        face where SDF~0 regardless of in-plane angle, so a handful of
        edge-region points get outvoted (checked empirically: SDF alone
        picked visually-misaligned wireframes with good-looking low error).
      - silhouette IoU against `mask` (RGB) -- the opposite profile: blind
        to depth/distance, but precisely sensitive to in-plane extent since
        the outer boundary IS what bounds it, and unaffected by depth
        dropout since it's a color/semantic mask, not raw depth.
    Depth pins distance-to-surface; RGB pins in-plane rotation/sizing --
    together they cover what either misses alone.

    prev_axis_map={dim_value: unit_axis_in_camera_frame} from a prior frame
    -> break ties toward whichever candidate is most CONTINUOUS with that
    prior labeling, instead of a fresh vote every frame (a physical
    dimension can't relabel itself frame to frame; a fresh vote under
    noise or a single-face view can occasionally flip which is which).

    Returns dict, or {"n_faces": 0} if no face was found at all."""
    remaining = pts.copy()
    MIN_FACE_PTS = 100
    ORTHOGONAL_TOL = 0.35  # |cos angle| vs every already-accepted face normal
    faces = []
    attempts = 0
    while len(faces) < 3 and attempts < 6 and len(remaining) >= MIN_FACE_PTS:
        attempts += 1
        res = ransac_plane(remaining, n_trials=200, thresh=0.003)
        if res is None:
            break
        n, d, inliers = res
        if inliers.sum() < MIN_FACE_PTS:
            break  # what's left is too sparse/noisy to be a real face
        face_pts = remaining[inliers]
        candidate_c = face_pts.mean(axis=0)
        # reject same-face fragments (rounded edges, a plane that RANSAC didn't
        # fully sweep up the first time) that aren't actually a distinct,
        # orthogonal box face -- discard the points either way so they aren't
        # re-fit into an infinite loop
        is_new_face = all(abs(n @ fn) < ORTHOGONAL_TOL for fn, fc, fp in faces)
        remaining = remaining[~inliers]
        if is_new_face:
            faces.append((n, candidate_c, face_pts))

    if not faces:
        return {"n_faces": 0}

    obj_c = pts.mean(axis=0)
    normals = [n if n @ (c - obj_c) > 0 else -n for n, c, fp in faces]
    n1 = normals[0]

    candidates = []  # each: (axis_dims, center, R)

    if len(faces) >= 2:
        # R's columns must be a proper (det=+1) rotation, but which of the
        # box's 8 corners is visible determines whether that corner's 3
        # TRUE outward normals form a right- or left-handed triple (e.g.
        # top+front+right is right-handed, bottom+front+right isn't) -- so
        # R is built via Gram-Schmidt + cross product (right-handed by
        # construction) from n1, n2's direction; a measured 3rd face is
        # used only for its own center-offset sign below, not for R itself.
        n2 = normals[1] - (normals[1] @ n1) * n1
        n2 /= np.linalg.norm(n2)
        n3 = np.cross(n1, n2)
        n3 /= np.linalg.norm(n3)
        axes = [n1, n2, n3]
        outward = [n1, n2, n3 if (len(faces) < 3 or n3 @ normals[2] > 0) else -n3]
        R = np.stack(axes, axis=1)
        for perm in permutations(range(3)):  # perm[axis_idx] -> index into `dims`
            axis_dims = [dims[perm[k]] for k in range(3)]
            estimates = [c - outward[i] * (axis_dims[i] / 2.0) for i, (n, c, fp) in enumerate(faces)]
            candidates.append((axis_dims, np.mean(estimates, axis=0), R))
    else:
        # only 1 face: n1 (thickness axis) is pinned, but the in-plane
        # rotation around it is completely unconstrained by RANSAC alone --
        # sweep it and let box_sdf_error over ALL points (including
        # whatever sparse/off-face survivors exist elsewhere) pick the angle
        c1 = faces[0][1]
        u0 = np.array([1.0, 0.0, 0.0]) - n1 * n1[0]
        if np.linalg.norm(u0) < 1e-3:
            u0 = np.array([0.0, 1.0, 0.0]) - n1 * n1[1]
        u0 /= np.linalg.norm(u0)
        v0 = np.cross(n1, u0)
        for perm in permutations(range(3)):
            thickness, da, db = dims[perm[0]], dims[perm[1]], dims[perm[2]]
            for deg in range(0, 180, 5):  # a rectangle repeats every 180 deg
                th = np.deg2rad(deg)
                u = np.cos(th) * u0 + np.sin(th) * v0
                v = np.cross(n1, u)
                R_cand = np.stack([u, v, n1], axis=1)
                center = c1 - n1 * (thickness / 2.0)
                candidates.append(([da, db, thickness], center, R_cand))

    if prev_axis_map is None:
        def score(cand):  # lower is better; (1-iou) scaled to ~mm so neither term dominates
            axis_dims, center, R_cand = cand
            sdf_mm = box_sdf_error(pts, center, R_cand, axis_dims) * 1000
            sil = render_box_silhouette(mask.shape, center, R_cand, axis_dims, fx, fy, cx, cy)
            iou = mask_iou(sil, mask)
            return sdf_mm + (1.0 - iou) * 10.0
        best = min(candidates, key=score)
    else:
        def score(cand):  # lower is better: total axis rotation vs the locked labeling
            axis_dims, center, R_cand = cand
            total = 0.0
            for dim_val, prev_dir in prev_axis_map.items():
                k = min(range(3), key=lambda i: abs(axis_dims[i] - dim_val))
                total += 1.0 - abs(R_cand[:, k] @ prev_dir)  # 0=same axis (sign-agnostic), ~1=perpendicular
            return total
        best = min(candidates, key=score)

    axis_dims, center, R = best
    # sign is arbitrary (an axis, not an oriented ray) -- both score() above
    # and the next frame's call compare via abs(dot), so any choice works
    axis_map = {d: R[:, i] for i, d in enumerate(axis_dims)}
    err = box_sdf_error(pts, center, R, axis_dims)

    return dict(center=center, R=R, dims_assigned=axis_dims, n_faces=len(faces),
                sdf_error_mm=err * 1000, axis_map=axis_map)


def draw_wireframe(img, center, R, dims, fx, fy, cx, cy, color=(0, 255, 255)):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = center + (signs * (np.array(dims) / 2.0)) @ R.T
    proj = []
    for X, Y, Z in corners:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        proj.append((int(round(u)), int(round(v))))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        cv2.line(img, proj[a], proj[b], color, 2, cv2.LINE_AA)
    return img


def main():
    out_dir = os.path.join(HERE, "cuboid_pose_prototype_out")
    os.makedirs(out_dir, exist_ok=True)
    frame_dirs = sorted(glob.glob(os.path.join(HERE, "box_frames", "frame_*")))
    print(f"dims (sorted desc, m): {DIMS}")
    print(f"{len(frame_dirs)} frames found\n")

    for fd in frame_dirs:
        name = os.path.basename(fd)
        rgb = cv2.imread(os.path.join(fd, "rgb.png"))
        depth = np.load(os.path.join(fd, "depth.npy"))
        meta = json.load(open(os.path.join(fd, "meta.json")))
        fx, fy, cx, cy = meta["fx"], meta["fy"], meta["cx"], meta["cy"]

        t0 = time.time()
        pc = pointcloud(depth, fx, fy, cx, cy)
        valid = depth > 0.05
        box_pts, obj_mask = segment_box(pc, valid, rgb)
        t_seg = time.time() - t0

        if box_pts is None or len(box_pts) < 30:
            print(f"{name}: segmentation failed ({0 if box_pts is None else len(box_pts)} pts)")
            continue

        t1 = time.time()
        fit = fit_cuboid_known_dims(box_pts, list(DIMS), obj_mask, fx, fy, cx, cy)
        t_fit = time.time() - t1

        if fit.get("n_faces", 0) == 0:
            print(f"{name}: no face found -- {len(box_pts)} box pts, seg={t_seg*1000:.1f}ms")
            continue

        print(f"{name}: {len(box_pts)} box pts | seg={t_seg*1000:.1f}ms fit={t_fit*1000:.2f}ms | "
              f"n_faces={fit['n_faces']} sdf_error={fit['sdf_error_mm']:.2f}mm "
              f"dims_assigned_cm={np.round(np.array(fit['dims_assigned'])*100, 1)} "
              f"center_cm={np.round(fit['center']*100, 1)}")

        vis = rgb.copy()
        draw_wireframe(vis, fit["center"], fit["R"], fit["dims_assigned"], fx, fy, cx, cy)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), vis)

    print(f"\noverlay renders -> {out_dir}")


if __name__ == "__main__":
    main()
