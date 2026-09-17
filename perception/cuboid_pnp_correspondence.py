"""Step B+C (per plan): 6 silhouette vertices -> which 3D cuboid corner is
each one? -> solvePnP per hypothesis -> rank by contour distance (RGB
only, no depth yet -- that's Step D/E, separate).

Key idea replacing the old angle sweep: a cuboid's 8 corners, indexed by
sign (sx,sy,sz), have a fixed edge graph (two corners adjacent iff they
differ in exactly one sign). A generic view showing 3 faces hides exactly
one antipodal corner pair (the nearest-to-camera and farthest-from-camera
corners); the remaining 6 corners are forced by that graph into a SINGLE
hexagonal cycle -- not an arbitrary ordering. So the search is: which of
the 4 antipodal pairs is hidden (4) x which detected vertex is the cycle's
start (6) x which winding direction (2) = 48 discrete hypotheses, each
directly solvable by PnP -- no continuous angle to sweep at all.
"""
import glob
import json
import os
from functools import lru_cache

import cv2
import numpy as np

from cuboid_silhouette_vertices import mask_to_polygon, segment_yellow_box, segment_postit
from cuboid_pose_prototype import box_sdf_error

HERE = os.path.dirname(os.path.abspath(__file__))

CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])  # 8x3


def cuboid_corners_3d(dims):
    return CORNER_SIGNS * (np.array(dims) / 2.0)  # 8x3, dims=(L,W,H) in the object's own frame


def _cube_edges():
    edges = {i: [] for i in range(8)}
    for i in range(8):
        for j in range(8):
            if i != j and np.sum(CORNER_SIGNS[i] != CORNER_SIGNS[j]) == 1:
                edges[i].append(j)
    return edges


_EDGES = _cube_edges()


def hexagon_corner_cycles():
    """The 4 possible 6-corner silhouette cycles (one per hidden antipodal
    pair), each a length-6 list of corner indices in cyclic (silhouette-
    boundary) order -- the ONLY orderings a real cuboid's hexagon can have,
    per the cube edge graph."""
    cycles = []
    seen = set()
    for i in range(8):
        j = 7 - i
        if (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        remaining = [k for k in range(8) if k != i and k != j]
        cycle = [remaining[0]]
        prev, cur = None, remaining[0]
        for _ in range(5):
            nxt = [n for n in _EDGES[cur] if n in remaining and n != prev][0]
            cycle.append(nxt)
            prev, cur = cur, nxt
        cycles.append(cycle)
    return cycles


def enumerate_hexagon_hypotheses(dims):
    """48 (4 hidden-pair x 6 rotation x 2 winding) object-point orderings,
    each 6x3, matched to however a detected 6-vertex polygon is ordered."""
    corners3d = cuboid_corners_3d(dims)
    hyps = []
    for cycle in hexagon_corner_cycles():
        for winding in (cycle, cycle[::-1]):
            for shift in range(6):
                order = winding[shift:] + winding[:shift]
                hyps.append(corners3d[order])
    return hyps


@lru_cache(maxsize=32)
def model_hexagon_hypotheses(dims, symmetry_key):
    """Cache exact 2D-3D correspondence classes for a metric model."""
    group=np.asarray(symmetry_key).reshape(-1,3,3)
    unique=[];seen=set()
    for h in enumerate_hexagon_hypotheses(dims):
        key=min(tuple((h@S).ravel()) for S in group)
        if key not in seen:
            seen.add(key);unique.append(h)
    return tuple(unique)


def solve_pnp_hexagon(poly, dims, camera_matrix, dist_coeffs=None, *, symmetry_reduced=False, model_symmetries=None):
    """poly: 6x2 detected silhouette vertices. Returns a list of dicts
    (R, t, obj_pts, reproj_err) for every hypothesis PnP could solve with
    the object in front of the camera."""
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)
    img_pts = poly.astype(np.float64)
    out = []
    hypotheses = enumerate_hexagon_hypotheses(dims)
    # Each of the four hidden-corner pairs is related to the first by one
    # of a rectangular cuboid's proper 180-degree symmetries. Twelve
    # correspondences cover the same physical geometries as all 48.
    if symmetry_reduced:
        hypotheses = (hypotheses[:12] if model_symmetries is None else
                      model_hexagon_hypotheses(tuple(dims),tuple(np.asarray(model_symmetries).ravel())))
    for obj_pts in hypotheses:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        t = tvec.flatten()
        if t[2] <= 0:
            continue
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
        out.append(dict(R=R, t=t, obj_pts=obj_pts, reproj_err=reproj_err))
    return out


def _boundary_distance_transform(shape, contour_px):
    """contour_px: Nx2 int pixel coords of a closed contour. Returns a dt
    image: distance (px) from every pixel to the nearest contour pixel."""
    boundary = np.full(shape, 255, dtype=np.uint8)
    cv2.polylines(boundary, [contour_px.astype(np.int32)], True, 0, 1)
    return cv2.distanceTransform(boundary, cv2.DIST_L2, 5)


def _mask_roi(mask, pad=20):
    """(x0,y0,x1,y1) bounding box of the mask, padded -- distance
    transforms only need to cover the object's own neighborhood, not the
    full 640x480 frame (profiled: this was ~80% of fit_pose_rgb_then_depth's
    time, since it ran a full-frame distanceTransform for EVERY one of
    ~15-40 surviving candidates PER OBJECT -- with several objects in
    frame this alone dropped live fps from ~11 to ~3.3, journal #54)."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
    return x0, y0, x1, y1


def contour_error(roi, offset, dt_obs, obs_contour, center, R, dims, camera_matrix):
    """Symmetric contour distance (px): rendered-box-contour points scored
    against the observed mask's own distance transform, and vice versa --
    gives an actual pixel-error number, unlike IoU. `roi`=(h,w) of the
    cropped region, `offset`=(x0,y0) of that crop in full-image
    coordinates -- `obs_contour` and every projected point are shifted
    into the SAME cropped-local frame before any pixel work."""
    x0, y0 = offset
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    corners = center + (CORNER_SIGNS * (np.array(dims) / 2.0)) @ R.T
    if np.any(corners[:, 2] <= 0):
        return 1e9
    proj = np.array([[fx * X / Z + cx - x0, fy * Y / Z + cy - y0] for X, Y, Z in corners], dtype=np.float32)
    hull = cv2.convexHull(proj).reshape(-1, 2)

    # dense pixels ALONG the hull's edges, not just its ~6-8 corner vertices --
    # scoring only the vertices badly under-samples a long edge (a wrong-shaped
    # box could have its few corners land in low-distance spots by coincidence
    # while the edge between them is nowhere near the real boundary)
    h, w = roi
    render_line_img = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(render_line_img, [hull.astype(np.int32)], True, 1, 1)
    ry, rx = np.where(render_line_img > 0)
    e_r_to_o = dt_obs[ry, rx].mean() if len(ry) else 1e9

    dt_render = _boundary_distance_transform(roi, hull)
    obs_clip = np.clip(obs_contour, [0, 0], [w - 1, h - 1])
    e_o_to_r = dt_render[obs_clip[:, 1], obs_clip[:, 0]].mean()
    return float((e_r_to_o + e_o_to_r) / 2.0)


def _candidates_from_mask(mask, dims, camera_matrix, reproj_slack):
    """Step B + reprojection pre-filter (see journal #53): drop any
    candidate whose reproj_err is more than `reproj_slack`x the best
    achieved among all 48 hypotheses -- a candidate whose R,t don't even
    reproject its OWN assumed 6 correspondences well can still have a
    box-shaped-enough silhouette to score deceptively low on contour
    distance alone (confirmed by overlaying the detected polygon against
    a "winning" wireframe that didn't touch it before this filter)."""
    poly, n = mask_to_polygon(mask, target_counts=(6,))
    if poly is None:
        return None
    candidates = solve_pnp_hexagon(poly, dims, camera_matrix)
    if not candidates:
        return None
    best_reproj = min(c["reproj_err"] for c in candidates)
    return [c for c in candidates if c["reproj_err"] <= best_reproj * reproj_slack + 1e-6]


def _rank_by_contour(mask, dims, camera_matrix, candidates):
    x0, y0, x1, y1 = _mask_roi(mask)
    roi = (y1 - y0, x1 - x0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    obs_contour = max(contours, key=cv2.contourArea).reshape(-1, 2) - [x0, y0]
    dt_obs = _boundary_distance_transform(roi, obs_contour)
    for c in candidates:
        c["contour_err"] = contour_error(roi, (x0, y0), dt_obs, obs_contour, c["t"], c["R"], dims, camera_matrix)
    return sorted(candidates, key=lambda c: c["contour_err"])


def fit_pose_rgb_only(mask, dims, camera_matrix, reproj_slack=1.5):
    """Step B+C only (RGB, no depth). Returns None if no 6-vertex polygon
    or no valid PnP hypothesis, else dict(R, t, contour_err_px,
    n_candidates)."""
    candidates = _candidates_from_mask(mask, dims, camera_matrix, reproj_slack)
    if not candidates:
        return None
    ranked = _rank_by_contour(mask, dims, camera_matrix, candidates)
    best = ranked[0]
    return dict(R=best["R"], t=best["t"], contour_err_px=best["contour_err"], n_candidates=len(candidates))


def rotation_distance(R1, R2):
    """Geodesic SO(3) angle (radians) between two rotations. NOT an Euler-
    angle difference -- that's axis-order-dependent and discontinuous;
    this is the actual angle of the rotation that takes R1 to R2."""
    R = R1.T @ R2
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_theta)


def fit_pose_rgb_then_depth(mask, pts, dims, camera_matrix, reproj_slack=1.5, top_k=3, lambda_depth=1.0,
                             prev_pose=None, lambda_temporal=1.0, rot_scale_deg=10.0, trans_scale_m=0.02,
                             reject_rot_deg=60.0, reject_trans_m=0.10):
    """Step D: RGB (Step B+C) narrows to the top `top_k` candidates by
    contour distance; ONLY those are re-scored with depth added in
    (box_sdf_error, journal #50's metric, over the mask's actual depth
    points), giving each candidate's image-only `combined` score. Depth
    here is a small tie-break among already-plausible RGB candidates, not
    a search from scratch -- the whole point of this redesign vs the old
    angle sweep.

    journal #57/#60 (user's design): if `prev_pose=(R_prev, t_prev)` is
    given, each candidate also gets a temporal term -- NOT used to
    generate candidates (all top_k are still tried every frame,
    independent of history), only to help pick among ones RGB+depth alone
    can't separate. `rot_delta_deg`/`trans_delta_m` are the raw SO(3)/
    Euclidean deltas vs. prev_pose (kept for logging, journal #57's V1).
    `e_temporal = rot_delta_deg/rot_scale_deg + trans_delta_m/trans_scale_m`
    (both terms unitless once scaled; scales are normalization constants,
    not physical maxima -- first-guess values per the user's spec, meant
    to be tuned from live logs, not treated as ground truth). Final
    selection score: `score_total = combined + lambda_temporal*e_temporal`.
    A candidate is also checked against a generous hard gate
    (`reject_rot_deg`=60, `reject_trans_m`=0.10, the user's own first-guess
    numbers) -- gate-passing candidates are preferred, but if NONE pass
    (e.g. genuine fast motion, or an object with no good fit at all) the
    gate is not an absolute veto: falls back to ranking all top_k by
    `score_total` rather than returning nothing every such frame. This
    fallback choice is mine, not explicitly specified -- flag if a hard
    reject (return None) was actually wanted instead.
    When prev_pose is None (new object, no history yet), `score_total ==
    combined` and the gate passes everything -- identical to the original
    RGB+depth-only behavior.

    Returns None, or dict(R, t, contour_err_px, sdf_err_mm, combined,
    score_total, winner_rank (1-based index into top_k),
    top_k=[(contour_err, sdf_err_mm, combined, rot_delta_deg,
    trans_delta_m, score_total), ...] for transparency; deltas are None
    when prev_pose is None)."""
    candidates = _candidates_from_mask(mask, dims, camera_matrix, reproj_slack)
    if not candidates:
        return None
    ranked = _rank_by_contour(mask, dims, camera_matrix, candidates)[:top_k]

    for c in ranked:
        c["sdf_err_mm"] = (box_sdf_error(pts, c["t"], c["R"], dims) * 1000) if len(pts) > 0 else 1e9
        c["combined"] = c["contour_err"] + lambda_depth * c["sdf_err_mm"]
        if prev_pose is not None:
            R_prev, t_prev = prev_pose
            c["rot_delta_deg"] = float(np.degrees(rotation_distance(R_prev, c["R"])))
            c["trans_delta_m"] = float(np.linalg.norm(c["t"] - t_prev))
            e_temporal = c["rot_delta_deg"] / rot_scale_deg + c["trans_delta_m"] / trans_scale_m
            c["score_total"] = c["combined"] + lambda_temporal * e_temporal
            c["gate_ok"] = c["rot_delta_deg"] <= reject_rot_deg and c["trans_delta_m"] <= reject_trans_m
        else:
            c["rot_delta_deg"] = c["trans_delta_m"] = None
            c["score_total"] = c["combined"]
            c["gate_ok"] = True

    survivors = [c for c in ranked if c["gate_ok"]] or ranked
    best = min(survivors, key=lambda c: c["score_total"])
    # journal #62: NOT ranked.index(best) -- these dicts hold numpy arrays (R, t,
    # obj_pts), and list.index() compares with `==`, which for arrays returns an
    # elementwise array instead of a bool and crashes ("truth value of an array
    # with more than one element is ambiguous"). Find the position by identity.
    winner_rank = next(i for i, c in enumerate(ranked) if c is best) + 1
    return dict(R=best["R"], t=best["t"], contour_err_px=best["contour_err"],
                sdf_err_mm=best["sdf_err_mm"], combined=best["combined"],
                score_total=best["score_total"], winner_rank=winner_rank,
                top_k=[(c["contour_err"], c["sdf_err_mm"], c["combined"],
                        c["rot_delta_deg"], c["trans_delta_m"], c["score_total"]) for c in ranked])


def draw_wireframe(img, center, R, dims, camera_matrix, color=(0, 255, 255)):
    # journal #64: was (0,0,255) red -- collided with drawFrameAxes' red X-axis,
    # making the two indistinguishable when both are drawn together. Yellow.
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    corners = center + (CORNER_SIGNS * (np.array(dims) / 2.0)) @ R.T
    proj = [(int(round(fx * X / Z + cx)), int(round(fy * Y / Z + cy))) for X, Y, Z in corners]
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        cv2.line(img, proj[a], proj[b], color, 2, cv2.LINE_AA)
    return img


def draw_axes(img, center, R, camera_matrix, length=0.04, dist_coeffs=None):
    """journal #64: draws the object frame's own x_O/y_O/z_O axes (OpenCV's
    standard red/green/blue) from `center`, using cv2.drawFrameAxes directly.
    Complements draw_wireframe -- the wireframe shows the box's shape/depth
    but all edges are one color, so a symmetry-flip (journal #61: identical
    silhouette, R swapped ~180deg about an axis) is hard to see live. Colored
    axes make a flip visually obvious (e.g. red suddenly pointing the other
    way) instead of needing to read rot_delta off the terminal log."""
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.asarray(center, dtype=np.float64).reshape(3, 1)
    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length, 3)
    return img


def main():
    out_dir = os.path.join(HERE, "cuboid_pnp_out")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [
        ("box_frames", segment_yellow_box, (0.077, 0.053, 0.030)),
        ("postit_frames", segment_postit, (0.078, 0.078, 0.008)),
    ]

    for folder, segment_fn, dims in datasets:
        frame_dirs = sorted(glob.glob(os.path.join(HERE, folder, "frame_*")))
        for fd in frame_dirs:
            name = f"{folder}_{os.path.basename(fd)}"
            rgb = cv2.imread(os.path.join(fd, "rgb.png"))
            depth = np.load(os.path.join(fd, "depth.npy"))
            meta = json.load(open(os.path.join(fd, "meta.json")))
            K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
            valid = depth > 0.05

            mask = segment_fn(rgb, valid)
            if mask is None:
                print(f"{name}: segmentation failed")
                continue

            fit = fit_pose_rgb_only(mask, dims, K)
            if fit is None:
                print(f"{name}: no 6-vertex polygon or no valid PnP hypothesis")
                continue

            print(f"{name}: contour_err={fit['contour_err_px']:.2f}px "
                  f"(of {fit['n_candidates']} valid PnP candidates) "
                  f"t_cm={np.round(fit['t']*100,1)}")

            vis = rgb.copy()
            draw_wireframe(vis, fit["t"], fit["R"], dims, K)
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), vis)

    print(f"\noverlays -> {out_dir}")


if __name__ == "__main__":
    main()
