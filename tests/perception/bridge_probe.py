#!/usr/bin/env python3
"""Run the existing bridge processing method on saved RGB-D without ROS nodes.

Source ROS + piper_pnp and use the defm Python environment. This imports the
actual Segmenter, filters, target selection and pose adapter. Output publishers
are replaced with local collectors; no robot commands or ROS messages are sent.
World-up is unavailable in this offline check, so output is camera-frame only.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge

from ros_pose_bridge import CuboidPoseBridge, Segmenter, HERE


class Collector:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Probe:
    # Execute production methods directly, with local sinks in place of ROS.
    _process = CuboidPoseBridge._process
    _process_frame = CuboidPoseBridge._process_frame
    _draw_debug = CuboidPoseBridge._draw_debug
    _maybe_debug = CuboidPoseBridge._maybe_debug
    _reset_lock = CuboidPoseBridge._reset_lock
    _note_miss = CuboidPoseBridge._note_miss
    _pose_is_stale = CuboidPoseBridge._pose_is_stale

    def __init__(self, segmenter, dims, surface_offset, target_color='any'):
        self.seg = segmenter
        self.pose_solver = 'legacy'  # existing regression probes; CLI can select rgbd
        self.bridge = CvBridge()
        self.dims = dims
        self.object_catalog = None
        self.target_pub = Collector()
        self.model_markers_pub = Collector()
        self.surface_offset = surface_offset
        self.min_points = 30
        self.min_depth, self.max_depth = 0.15, 0.90
        self.border_px, self.max_border_frac = 6, 0.12
        self.excl_corner_h, self.excl_corner_w = 0.32, 0.24
        self.max_combined, self.max_sdf_mm = 40.0, 60.0
        self.pick_lock, self.pick_select = True, 'best_fit'
        self.target_color = target_color
        self.target_color_min_fraction = 0.6
        self.lock_radius, self.lock_timeout = 0.08, 15
        self.lock_timeout_s = 2.0
        self.max_pose_age_s = 0.0
        self.use_tf_up = False
        self.base_frame = 'base_link'
        self._lock_seen = self._lock_frame = None
        self._lock_pos = self._prev_z = self._prev_x = None
        self._lock_miss = 0
        self.pick_id = 0
        self.pub, self.dbg_pub = Collector(), Collector()
        self.logs = []

    def get_logger(self):
        return self

    def info(self, message, **kwargs):
        self.logs.append(message)
        print(message, flush=True)

    warn = info

    def _base_from_camera(self, frame, stamp):
        return None

