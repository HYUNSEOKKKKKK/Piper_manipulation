#!/usr/bin/env python3
"""CPU-only cuboid pose example. Optional NPZ: mask, depth_m, K, dims_m."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
from cuboid_rgbd import fit_cuboid_rgbd, symmetry_angle
from cuboid_pnp_correspondence import draw_wireframe


def synthetic():
    """Independent ray/rectangle renderer, arbitrary orientation, no table prior."""
    dims = np.array([.078, .053, .030])
    K = np.array([[420., 0, 160.], [0, 420., 120.], [0, 0, 1.]])
    R = cv2.Rodrigues(np.radians([25., 35., 15.]))[0]
    t = np.array([.01, -.012, .44])
    yy, xx = np.indices((240, 320))
    rays = np.stack(((xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(xx)), axis=-1)
    depth = np.full(xx.shape, np.inf)
    for axis in range(3):
        for sign in (-1, 1):
            normal = sign * R[:, axis]
            center = t + normal * dims[axis] / 2
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (center @ normal) / (rays @ normal)
                local = (rays * z[..., None] - t) @ R
            valid = (z > 0) & np.isfinite(z)
            for a in range(3):
                if a != axis:
                    valid &= np.abs(local[..., a]) <= dims[a] / 2 + 1e-9
            depth = np.where(valid & (z < depth), z, depth)
    mask = np.isfinite(depth)
    depth[mask] += np.random.default_rng(1).normal(0, .0005, int(mask.sum()))
    depth[~mask] = 0
    return mask, depth.astype(np.float32), K, dims, (R, t)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/example'))
    args = p.parse_args()
    ground_truth = None
    if args.npz:
        with np.load(args.npz, allow_pickle=False) as data:
            mask, depth, K, dims = (np.asarray(data[k]) for k in ('mask', 'depth_m', 'K', 'dims_m'))
        if mask.shape != depth.shape or depth.ndim != 2 or K.shape != (3, 3) or dims.shape != (3,):
            raise SystemExit('Expected HxW mask/depth_m, 3x3 K and 3-vector dims_m')
        if not np.all(np.isfinite(K)) or np.any(dims <= 0) or not np.all(np.isfinite(dims)):
            raise SystemExit('Invalid intrinsics or metric box dimensions')
    else:
        mask, depth, K, dims, ground_truth = synthetic()
    started = time.perf_counter()
    fit = fit_cuboid_rgbd(mask.astype(bool), depth, dims, K)
    elapsed_ms = (time.perf_counter() - started) * 1000
    report = {'accepted': bool(fit['accepted']), 'reason': fit.get('reason'),
              'geometry_wall_ms': elapsed_ms, 'input': 'NPZ' if args.npz else 'synthetic',
              'frame': 'camera optical; x right, y down, z forward', 'dimensions_m': dims.tolist()}
    if fit['accepted']:
        report.update(R=fit['R'].tolist(), t_m=fit['t'].tolist())
        if ground_truth:
            report['translation_error_mm'] = float(np.linalg.norm(fit['t'] - ground_truth[1]) * 1000)
            report['symmetry_rotation_error_deg'] = symmetry_angle(ground_truth[0], fit['R'], dims)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'pose.json').write_text(json.dumps(report, indent=2)+'\n')
    vis = np.full((*mask.shape, 3), 245, np.uint8)
    vis[mask.astype(bool)] = [140, 190, 230]
    if fit['accepted']:
        draw_wireframe(vis, fit['t'], fit['R'], dims, K)
    cv2.imwrite(str(args.output / 'overlay.png'), vis)
    print(json.dumps(report, indent=2))
    if not fit['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
