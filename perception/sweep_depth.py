"""Sparse RGB-D height observation for the continuous drop mode (no inference).

Coverage is spatial, not merely a point count. Missing depth is never treated
as an empty table. A small median removes isolated stereo outliers before the
maximum height is taken; do not use a percentile that could erase a small box.
"""
import cv2
import numpy as np


def observe_workspace(depth, K, R_bc, t_bc, workspace, stride=4):
    filtered = cv2.medianBlur(np.asarray(depth, np.float32), 3)
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = filtered[::stride, ::stride]
    valid = np.isfinite(z) & (z > .15) & (z < 1.5)
    points = np.stack(((u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z), axis=-1)[valid]
    points = points @ R_bc.T + t_bc
    x,y,_ = workspace.places[0]
    dx,dy = workspace.drop_size

    def region(bounds):
        x0,x1,y0,y1 = bounds
        p = points[(points[:,0] >= x0) & (points[:,0] <= x1)
                   & (points[:,1] >= y0) & (points[:,1] <= y1)
                   & (points[:,2] >= workspace.table_z-.025)]
        if len(p) < 40:
            return dict(coverage=0., height_m=None)
        ix = np.clip(((p[:,0]-x0)/(x1-x0)*4).astype(int), 0, 3)
        iy = np.clip(((p[:,1]-y0)/(y1-y0)*4).astype(int), 0, 3)
        bins = np.bincount(ix+4*iy, minlength=16)
        return dict(coverage=float(np.mean(bins >= 3)),
                    height_m=float(max(workspace.table_z, np.max(p[:,2]))))

    return dict(workspace_digest=workspace.digest, frame_id='base_link',
                drop=region((x-dx/2,x+dx/2,y-dy/2,y+dy/2)),
                source=(None if getattr(workspace, 'feed_drop', False) else region(workspace.pick_bounds)))
