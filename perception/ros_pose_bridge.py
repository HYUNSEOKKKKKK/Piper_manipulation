#!/usr/bin/env python3
"""ROS RGB-D cuboid pose bridge: YOLO box prompts, MobileSAMv2, geometric SE(3).

Runs in the perception Python environment with the ROS workspace sourced.
Model files are downloaded separately by scripts/fetch_weights.py.
The object-center estimate and grasp-frame adaptation are distinct outputs.
"""

import os
import json
import sys
import math
import threading
import time

import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
MOBILE_SAM_ROOT = os.environ.get("MOBILE_SAM_ROOT", os.path.join(ROOT, "third_party", "MobileSAM", "MobileSAMv2"))
sys.path.insert(0, MOBILE_SAM_ROOT)
WEIGHTS = os.environ.get("PIPER_WEIGHTS_DIR", os.path.join(ROOT, "weights"))

# --- RGB-D refinement and the legacy PnP comparison backend ------------------
from cuboid_silhouette_vertices import mask_to_polygon
from cuboid_pnp_correspondence import fit_pose_rgb_then_depth, draw_wireframe, draw_axes
from cuboid_pose_prototype import pointcloud
from perception_timing import FrameTiming, TimingWindow
from cuboid_rgbd import FitConfig, fit_cuboid_rgbd, fit_cuboid_models, align_symmetric_rotation
from cuboid_native import LIB as CUBOID_NATIVE

# --- ROS --------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, TransformException

try:
    from ros2_aruco_interfaces.msg import ArucoMarkers
except ImportError:  # pragma: no cover
    ArucoMarkers = None


# ===========================================================================
# MobileSAMv2 object detection + segmentation
#   Copied from cuboid_pose_live.py (lines ~123-178). If that file changes,
#   update here too. Kept as a class so weights load once.
# ===========================================================================
class Segmenter:
    def __init__(self, mobile_sam_ckpt, object_aware_ckpt, decoder_ckpt,
                 conf=0.4, iou=0.9, imgsz=640, fp16=True, log=print):
        from mobilesamv2 import sam_model_registry, SamPredictor
        from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.fp16 = bool(fp16) and self.device == "cuda"

        encoder = sam_model_registry["tiny_vit"]()
        sd = torch.load(mobile_sam_ckpt, map_location="cpu")
        enc_sd = {k[len("image_encoder."):]: v for k, v in sd.items()
                  if k.startswith("image_encoder.")}
        encoder.load_state_dict(enc_sd, strict=True)

        sam = sam_model_registry["vit_h"]()
        decoder = sam_model_registry["PromptGuidedDecoder"](decoder_ckpt)
        sam.image_encoder = encoder
        sam.prompt_encoder = decoder["PromtEncoder"]
        sam.mask_decoder = decoder["MaskDecoder"]
        sam.to(device=self.device).eval()

        self.sam = sam
        self.predictor = SamPredictor(sam)
        self.obj_model = ObjectAwareModel(object_aware_ckpt)
        log(f"[bridge] MobileSAMv2 loaded on {self.device}")

    def _autocast(self):
        import contextlib
        if self.fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @staticmethod
    def _suppress_contained(boxes, confs, thresh):
        order = np.argsort(confs)[::-1]
        keep = []
        for i in order:
            bi = boxes[i]
            area_i = max(bi[2] - bi[0], 0) * max(bi[3] - bi[1], 0)
            suppressed = False
            for j in keep:
                bj = boxes[j]
                x0, y0 = max(bi[0], bj[0]), max(bi[1], bj[1])
                x1, y1 = min(bi[2], bj[2]), min(bi[3], bj[3])
                inter = max(x1 - x0, 0) * max(y1 - y0, 0)
                area_j = max(bj[2] - bj[0], 0) * max(bj[3] - bj[1], 0)
                if (inter / max(area_i, 1e-6) > thresh
                        or inter / max(area_j, 1e-6) > thresh):
                    suppressed = True
                    break
            if not suppressed:
                keep.append(i)
        return np.array(sorted(keep), dtype=int)

    def detect_and_segment(self, bgr, contain_thresh=0.7):
        self.last_timing_ms = {}
        started = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.obj_model(rgb, device=self.device, retina_masks=True,
                             imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                             verbose=False, half=self.fp16)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()
        self.last_timing_ms["seg_detector_wall"] = (time.perf_counter()-started)*1000.
        if len(boxes) == 0:
            return [], []
        keep = self._suppress_contained(boxes, confs, contain_thresh)
        boxes, confs = boxes[keep], confs[keep]
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if self.device == 'cuda' else None
        with torch.no_grad(), self._autocast():
            started = time.perf_counter()
            if events: events[0].record()
            self.predictor.set_image(rgb)
            if events: events[1].record()
            self.last_timing_ms['seg_encoder_host'] = (time.perf_counter()-started)*1000.
            started = time.perf_counter()
            if events: events[2].record()
            boxes_t = torch.from_numpy(
                self.predictor.transform.apply_boxes(
                    boxes, self.predictor.original_size)).to(self.device)
            image_embedding = torch.repeat_interleave(
                self.predictor.features, boxes_t.shape[0], dim=0)
            prompt_embedding = torch.repeat_interleave(
                self.sam.prompt_encoder.get_dense_pe(), boxes_t.shape[0], dim=0)
            sparse_emb, dense_emb = self.sam.prompt_encoder(
                points=None, boxes=boxes_t, masks=None)
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embedding, image_pe=prompt_embedding,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False, simple_type=True)
            masks = self.predictor.model.postprocess_masks(
                low_res_masks, self.predictor.input_size,
                self.predictor.original_size)
            masks = (masks > self.sam.mask_threshold).squeeze(1)
            if events: events[3].record()
            masks = masks.cpu().numpy()  # existing transfer waits for the CUDA work above
            self.last_timing_ms['seg_decoder_transfer_wall'] = (time.perf_counter()-started)*1000.
            # No extra synchronize: publish event durations only when already complete.
            if events and events[3].query():
                self.last_timing_ms['seg_encoder_cuda_event'] = events[0].elapsed_time(events[1])
                self.last_timing_ms['seg_decoder_cuda_event'] = events[2].elapsed_time(events[3])
        return masks, confs


# ===========================================================================
# pose convention adapter
# ===========================================================================
def mat_to_quat(R):
    """3x3 rotation -> (x, y, z, w). Shepperd's method."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-12)


def quat_from_yaw(yaw_rad):
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def to_grasp_pose(R, t, dims, up_cam=None, prev_z=None, prev_x=None, hyst=0.35,
                  surface_offset=0.0):
    """
    R : object->camera rotation (columns = the box's L,W,H axes in the
        camera OPTICAL frame).  t : box CENTRE, camera optical frame [m].
    dims : (L, W, H) [m].
    up_cam : world "up" (base_link +Z) in the camera optical frame, from TF.
        If None, fall back to camera image-up (-Y).
    prev_z, prev_x : last published Z_out / X_out (camera frame). Give history
        continuity a vote so the PnP silhouette-symmetry flip (README #61) and
        the near-tie between two "up-ish" faces don't chatter the output --
        this is what piper_pnp's 15 deg / 20 mm stability gate needs.
    hyst : how much weight the previous normal gets in the face vote.

    Reduce the 6DoF cuboid pose to what piper_pnp needs for a top-down pick,
    WITHOUT assuming which face the box rests on:

      * +Z_out (approach normal) = box face-normal (of +-L,+-W,+-H) best
        matching world-up (+ a hysteresis bonus for staying on last frame's
        face) == the physical TOP face, any placement.
      * origin = centre of that face.
      * +X_out = the longer in-plane box axis; sign locked to prev_x (else a
        fixed representative) so a 180 deg flip -> the SAME published yaw.

    Returns (pos[3], quat_xyzw[4], R_out[3x3]) in the camera optical frame.
    Box tilt is fine -- piper_pnp sweeps tilt candidates around this normal.
    """
    dims = np.asarray(dims, dtype=np.float64)
    axes = [R[:, i] / (np.linalg.norm(R[:, i]) + 1e-12) for i in range(3)]

    u = np.asarray(up_cam if up_cam is not None else (0.0, -1.0, 0.0), dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    pz = None if prev_z is None else np.asarray(prev_z, dtype=np.float64)

    def face_score(n):
        s = float(np.dot(n, u))
        if pz is not None:
            s += hyst * float(np.dot(n, pz))     # stick to last frame's face
        return s

    cands = [(i, s, s * axes[i]) for i in range(3) for s in (1.0, -1.0)]
    k, _s, z_out = max(cands, key=lambda c: face_score(c[2]))

    rest = [i for i in range(3) if i != k]
    a, b = (rest if dims[rest[0]] >= dims[rest[1]] else rest[::-1])  # a = longer
    x_out = axes[a] - np.dot(axes[a], z_out) * z_out
    if np.linalg.norm(x_out) < 1e-6:                    # near-degenerate: use b
        x_out = axes[b] - np.dot(axes[b], z_out) * z_out
    x_out = x_out / (np.linalg.norm(x_out) + 1e-12)

    # collapse the 180 deg flip. Anchor the sign to the previous X_out if we
    # have one (kills the frame-to-frame yaw flip outright); otherwise pick a
    # fixed representative by the dominant image-plane component.
    if prev_x is not None:
        if float(np.dot(x_out, np.asarray(prev_x, dtype=np.float64))) < 0.0:
            x_out = -x_out
    else:
        ref = x_out[0] if abs(x_out[0]) >= abs(x_out[1]) else x_out[1]
        if ref < 0.0:
            x_out = -x_out
    y_out = np.cross(z_out, x_out)
    R_out = np.column_stack([x_out, y_out, z_out])

    # origin: centre of the top face, + a calibration nudge along the outward
    # normal (surface_offset: +out / -into the box) for systematic depth/mount error.
    pos = t + z_out * (dims[k] / 2.0 + surface_offset)
    return pos, mat_to_quat(R_out), R_out, k


def _pose(pos, quat_xyzw):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in pos)
    (p.orientation.x, p.orientation.y,
     p.orientation.z, p.orientation.w) = (float(v) for v in quat_xyzw)
    return p


# ===========================================================================
# node
# ===========================================================================
class CuboidPoseBridge(Node):
    def __init__(self):
        super().__init__("cuboid_pose_bridge")

        g = self.declare_parameter
        g("dims", [0.077, 0.035, 0.030])          # current measured box, metres
        g("object_models_file", "")
        g("color_topic", "/camera/camera/color/image_raw")
        g("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        g("info_topic", "/camera/camera/color/camera_info")
        g("markers_topic", "/aruco_markers")
        g("pick_marker_id", 0)
        g("place_marker_id", 1)
        g("pick_select", "best_fit")              # best_fit | nearest | largest | highest
        g("target_color", "any")                 # any | yellow; identity hint for this demo
        g("batch_config_file", "")
        g("target_color_min_fraction", 0.6)      # fraction of each SAM mask matching color
        g("min_points", 30)
        # reject the eye-in-hand gripper fingertips (fixed image corners, run
        # off the frame edge, very close to the lens) and far background.
        g("min_depth_m", 0.15)     # detection median depth below this -> gripper
        g("max_depth_m", 0.90)     # above this -> background
        g("max_border_frac", 0.12) # >this fraction of the mask on the frame edge -> clipped
        g("border_px", 6)          # width of the "frame edge" band [px]
        g("excl_corner_h", 0.32)   # bottom fraction of the image height that is "corner"
        g("excl_corner_w", 0.24)   # left/right fraction of the width that is "corner"
        # noise gate: drop a detection whose fit is this bad (see live log --
        # real boxes ~10, junk >150). Both must pass.
        g("max_combined_err", 40.0)
        g("max_sdf_err_mm", 60.0)
        # object lock: once a target is chosen, keep following the detection
        # nearest its (smoothed) last position, so frame-to-frame flicker
        # between several equally-good boxes can't move the published target.
        g("pick_lock", True)
        g("lock_radius_m", 0.08)
        g("lock_timeout_frames", 15)
        g("lock_timeout_s", 2.0)
        # which box face is "up": align a face normal with world-up (base_frame
        # +Z) looked up at image time. Missing TF suppresses output. Offline
        # diagnostics may explicitly use use_tf_up=false for a camera-only pose.
        g("base_frame", "base_link")
        g("use_tf_up", True)
        g("max_pose_age_s", 0.5)  # capture-to-publication deadline; 0 for offline replay
        g("require_cuda", False) # robot launcher enables this; CPU is allowed for diagnostics
        # calibration: shift the published grasp point along the top-face
        # outward normal. + = away from the box (up), - = into it. Use it to
        # dial out a systematic depth / camera-mount error.
        g("surface_offset_m", 0.0)
        g("max_rate_hz", 30.0)
        g("sync_slop_s", 0.05)
        # fixed place target (state machine needs BOTH ids before it starts).
        g("publish_place", True)
        g("place_frame", "base_link")
        g("place_xyz", [0.30, -0.15, 0.02])       # base_link [m]; z ~ half object height
        g("place_yaw_deg", 0.0)                   # about base_link +Z (up)
        g("mobile_sam_checkpoint", os.path.join(WEIGHTS, "mobile_sam.pt"))
        g("object_aware_checkpoint",
          os.path.join(WEIGHTS, "ObjectAwareModel.pt"))
        g("decoder_checkpoint",
          os.path.join(WEIGHTS, "Prompt_guided_Mask_Decoder.pt"))
        g("conf", 0.4)
        g("iou", 0.9)
        g("imgsz", 640)
        g("fp16", True)
        g("publish_debug_image", True)
        g("profile_period_s", 5.0)
        g("pose_solver", "rgbd")              # rgbd | legacy (comparison/recovery)
        g("pose_max_points", 192)
        g("pose_contour_points", 64)
        g("pose_refine_candidates", 4)
        g("pose_refine_iterations", 5)
        g("pose_lock_prefilter", True)

        p = self.get_parameter
        self.dims = [float(v) for v in p("dims").value]
        self.object_catalog = None
        if p('object_models_file').value:
            from piper_pnp.object_models import ObjectCatalog
            self.object_catalog = ObjectCatalog.load(p('object_models_file').value)
        self.markers_topic = p("markers_topic").value
        self.pick_id = int(p("pick_marker_id").value)
        self.place_id = int(p("place_marker_id").value)
        self.pick_select = str(p("pick_select").value)
        self.target_color = str(p("target_color").value).lower()
        self.target_color_min_fraction = float(p("target_color_min_fraction").value)
        self.min_points = int(p("min_points").value)
        self.min_depth = float(p("min_depth_m").value)
        self.max_depth = float(p("max_depth_m").value)
        self.max_border_frac = float(p("max_border_frac").value)
        self.border_px = int(p("border_px").value)
        self.excl_corner_h = float(p("excl_corner_h").value)
        self.excl_corner_w = float(p("excl_corner_w").value)
        self.max_combined = float(p("max_combined_err").value)
        self.max_sdf_mm = float(p("max_sdf_err_mm").value)
        self.pick_lock = bool(p("pick_lock").value)
        self.lock_radius = float(p("lock_radius_m").value)
        self.lock_timeout = int(p("lock_timeout_frames").value)
        self.lock_timeout_s = float(p("lock_timeout_s").value)
        self.max_pose_age_s = float(p("max_pose_age_s").value)
        self.base_frame = str(p("base_frame").value)
        self.use_tf_up = bool(p("use_tf_up").value)
        self.surface_offset = float(p("surface_offset_m").value)
        self.min_period = 1.0 / max(float(p("max_rate_hz").value), 1e-3)
        self.publish_place = bool(p("publish_place").value)
        self.place_frame = str(p("place_frame").value)
        self.place_xyz = [float(v) for v in p("place_xyz").value]
        self.place_yaw = math.radians(float(p("place_yaw_deg").value))
        self.publish_debug = bool(p("publish_debug_image").value)
        self.profile_period_s = float(p("profile_period_s").value)
        if not math.isfinite(self.profile_period_s) or self.profile_period_s < 0:
            raise ValueError("profile_period_s must be finite and >= 0")
        self._timing_window = TimingWindow()
        self._last_profile = time.monotonic()
        self.pose_solver = str(p("pose_solver").value)
        if self.pose_solver not in ('legacy', 'rgbd'):
            raise ValueError('pose_solver must be legacy or rgbd')
        if self.object_catalog is not None and self.pose_solver != 'rgbd':
            raise ValueError('Multiple metric models require the rgbd solver')
        self.pose_config = FitConfig(
            max_points=int(p('pose_max_points').value),
            contour_points=int(p('pose_contour_points').value),
            refine_candidates=int(p('pose_refine_candidates').value),
            iterations=int(p('pose_refine_iterations').value))
        self.pose_lock_prefilter = bool(p('pose_lock_prefilter').value)
        if self.pose_solver == 'rgbd' and CUBOID_NATIVE is None:
            self.get_logger().warn('RGB-D geometry uses the slower NumPy fallback. '
                                   'Run bash ~/manipulator/build_cuboid_native.sh for CPU acceleration.')
        batch_path = p("batch_config_file").value
        self._batch_workspace = None
        if batch_path:
            from piper_pnp.batch_workspace import BatchWorkspace
            self._batch_workspace = BatchWorkspace.load(batch_path)
            if self._batch_workspace.drop and self.object_catalog is None:
                raise ValueError('sweep requires atomic metric object observations')
            if (self.target_color != "any" or not self.use_tf_up
                    or self.base_frame != "base_link"
                    or tuple(self.dims) != self._batch_workspace.dims):
                raise ValueError("Batch requires any color, capture-time base_link TF and matching dimensions")
            self.publish_place = False  # controller chooses the next unoccupied slot
        from rcl_interfaces.msg import ParameterDescriptor
        self.declare_parameter(
            "batch_workspace_digest", self._batch_workspace.digest if batch_path else "",
            ParameterDescriptor(read_only=True))
        self.declare_parameter('object_catalog_digest',
                              self.object_catalog.digest if self.object_catalog else '',
                              ParameterDescriptor(read_only=True))

        if len(self.dims) != 3 or not all(math.isfinite(v) and v > 0 for v in self.dims):
            raise ValueError("dims must contain three positive finite lengths in metres")
        if self.target_color not in ("any", "yellow"):
            raise ValueError("target_color must be any or yellow")
        if not 0 < self.target_color_min_fraction <= 1:
            raise ValueError("target_color_min_fraction must be in (0, 1]")
        if self.lock_timeout < 0 or self.lock_timeout_s <= 0 or self.max_pose_age_s < 0:
            raise ValueError("Invalid lock timeout or pose age limit")
        if bool(p("require_cuda").value) and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Check nvidia-smi / driver versions before robot inference.")

        if ArucoMarkers is None:
            raise RuntimeError(
                "ros2_aruco_interfaces not importable -- "
                "`source ~/piper_pnp/install/setup.bash` before running this node.")

        self.bridge = CvBridge()
        self.K = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
        self._lock_pos = None       # base-frame position when TF is enabled
        self._lock_miss = 0         # consecutive frames with no detection near the lock
        self._lock_seen = None
        self._lock_frame = None
        self._prev_z = None         # last published axes in the same frame as _lock_pos
        self._prev_x = None         # orientation hysteresis in to_grasp_pose
        self._prev_box_R = None
        self._lock_model_id = None
        self._track_id = time.time_ns()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.seg = Segmenter(
            p("mobile_sam_checkpoint").value, p("object_aware_checkpoint").value,
            p("decoder_checkpoint").value, conf=float(p("conf").value),
            iou=float(p("iou").value), imgsz=int(p("imgsz").value),
            fp16=bool(p("fp16").value), log=self.get_logger().info)

        self.pub = self.create_publisher(ArucoMarkers, self.markers_topic, 10)
        self.target_pub = self.create_publisher(String, '~/target', 1) if self.object_catalog else None
        self.model_markers_pub = self.create_publisher(ArucoMarkers, '~/markers', 1) if self.object_catalog else None
        self._performance_pub = self.create_publisher(String, "~/performance", 1)
        self.dbg_pub = (self.create_publisher(Image, "~/debug_image", 1)
                        if self.publish_debug else None)

        self.create_subscription(CameraInfo, p("info_topic").value, self._on_info, 1)
        cs = Subscriber(self, Image, p("color_topic").value)
        ds = Subscriber(self, Image, p("depth_topic").value)
        self.sync = ApproximateTimeSynchronizer(
            [cs, ds], queue_size=5, slop=float(p("sync_slop_s").value))
        self.sync.registerCallback(self._on_rgbd)

        if self.publish_place:
            # 10 Hz: the controller confirms a target only after 8 samples in a
            # 3 s window, so 2 Hz never accumulates enough for the fixed place.
            self.create_timer(0.1, self._publish_place)

        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"cuboid_pose_bridge up. dims(LWH)={self.dims} m, "
            f"models={list(self.object_catalog.by_id) if self.object_catalog else 'single'}, "
            f"select={self.pick_select}{' +lock' if self.pick_lock else ''}, "
            f"pose_solver={self.pose_solver}, "
            f"geometry={'native' if self.pose_solver == 'rgbd' and CUBOID_NATIVE is not None else 'numpy'}, "
            f"surface_offset={self.surface_offset*1000:.1f}mm, "
            f"color={self.target_color}, "
            f"gate combined<={self.max_combined:.0f} sdf<={self.max_sdf_mm:.0f}mm, "
            f"pick output={'~/target + ~/markers' if self.object_catalog else self.markers_topic} (id {self.pick_id} = object"
            + (f", id {self.place_id} = fixed place @ {self.place_xyz} {self.place_frame}"
               if self.publish_place else "") + ")")

    # ------------------------------------------------------------------
    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f"camera_info: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} "
                f"cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}")

    def _on_rgbd(self, color_msg, depth_msg):
        with self._lock:
            self._timing_window.input(replaced=self._latest is not None)
            self._latest = (color_msg, depth_msg)

    # ------------------------------------------------------------------
    def _work_loop(self):
        last = 0.0
        while not self._stop:
            now = time.monotonic()
            if now - last < self.min_period:
                time.sleep(0.005)
                continue
            with self._lock:
                item = self._latest
                self._latest = None
            if item is None:
                time.sleep(0.005)
                continue
            last = now
            try:
                self._process(*item)
            except Exception as exc:  # keep the node alive on a bad frame
                self.get_logger().warn(f"frame dropped: {exc}",
                                       throttle_duration_sec=5.0)

    def _process(self, color_msg, depth_msg):
        timing = FrameTiming()
        self._frame_timing = timing
        try:
            self._process_frame(color_msg, depth_msg)
        except Exception:
            timing.outcome = 'error'
            raise
        finally:
            timing.finish()
            self._last_timing = timing
            if hasattr(self, '_timing_window'):
                with self._lock:
                    self._timing_window.add(timing)
                period = self.profile_period_s
                if period > 0 and time.monotonic()-self._last_profile >= period:
                    self._last_profile = time.monotonic()
                    with self._lock:
                        report = self._timing_window.summary()
                    report['pose_solver'] = self.pose_solver
                    report['geometry_backend'] = 'native' if self.pose_solver == 'rgbd' and CUBOID_NATIVE is not None else 'numpy'
                    self._performance_pub.publish(String(data=json.dumps(report)))
                    stages = report['stage_ms']
                    self.get_logger().info(
                        f"perception: input={report['rgbd_input_hz']:.1f}Hz "
                        f"process={report['processed_hz']:.1f}Hz pick={report['published_pick_hz']:.1f}Hz "
                        f"total p50/p95={stages['total']['p50']:.1f}/{stages['total']['p95']:.1f}ms "
                        f"seg={stages.get('segmentation', {}).get('p50', 0.):.1f}ms "
                        f"pose={stages.get('pose', {}).get('p50', 0.):.1f}ms")

    def _process_frame(self, color_msg, depth_msg):
        timing = self._frame_timing
        preprocess_start = time.perf_counter()
        self._last_fit_details = []
        self._last_selected_fit = None
        if self.K is None:
            timing.outcome = 'no_intrinsics'
            self.get_logger().warn("waiting for camera_info…",
                                   throttle_duration_sec=5.0)
            return
        bgr = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self._debug_header = color_msg.header
        if self._pose_is_stale(color_msg.header.stamp):
            timing.outcome = 'stale_input'
            self._note_miss()
            self._maybe_debug(bgr, None)
            return
        cam_frame = color_msg.header.frame_id or "camera_color_optical_frame"
        transform = self._base_from_camera(cam_frame, color_msg.header.stamp)
        if self.use_tf_up and transform is None:
            timing.outcome = 'missing_tf'
            self._note_miss()
            self._maybe_debug(bgr, None)
            return
        R_bc, t_bc = transform if transform is not None else (np.eye(3), np.zeros(3))
        workspace = getattr(self, '_batch_workspace', None)
        if workspace is not None and transform is None:
            self._note_miss()
            self._maybe_debug(bgr, None)
            return
        lock_frame = self.base_frame if transform is not None else cam_frame
        if (self._lock_frame != lock_frame or
                (self._lock_seen is not None and
                 time.monotonic() - self._lock_seen > self.lock_timeout_s)):
            self._reset_lock()
        self._lock_frame = lock_frame
        if (workspace is not None and self._lock_pos is not None
                and not workspace.allows_pick(self._lock_pos)):
            self._reset_lock()
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        if depth_msg.encoding == "16UC1":
            depth_m = depth_raw.astype(np.float32) * 1e-3
        elif depth_msg.encoding == "32FC1":
            depth_m = depth_raw.astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth encoding: {depth_msg.encoding}")
        if bgr.shape[:2] != depth_m.shape:
            raise ValueError("Color and aligned depth dimensions differ")

        sweep_scene = None
        if workspace is not None and workspace.drop:
            from sweep_depth import observe_workspace
            with timing.stage('workspace_depth'):
                sweep_scene = observe_workspace(depth_m, self.K, R_bc, t_bc, workspace)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        timing.ms['preprocess'] = (time.perf_counter()-preprocess_start)*1000.
        with timing.stage("segmentation"):
            masks, confs = self.seg.detect_and_segment(bgr)
        timing.ms.update(getattr(self.seg, "last_timing_ms", {}))
        timing.counts["segmented_instances"] = len(masks)
        if len(masks) == 0:
            self._note_miss()
            self._maybe_debug(bgr, None)
            return
        solver = getattr(self, 'pose_solver', 'rgbd')
        if solver == 'legacy':
            with timing.stage("pointcloud"):
                pc = pointcloud(depth_m, fx, fy, cx, cy)
        valid = np.isfinite(depth_m) & (depth_m > 0.05)
        h, w = depth_m.shape
        bpx = max(1, int(round(self.border_px)))
        border = np.zeros((h, w), bool)
        border[:bpx, :] = border[-bpx:, :] = border[:, :bpx] = border[:, -bpx:] = True
        # eye-in-hand gripper fingertips sit rigidly in the two BOTTOM CORNERS
        # of the frame, whatever the arm pose. Exclude those corner rectangles.
        y_lo = int((1.0 - self.excl_corner_h) * h)
        x_l = int(self.excl_corner_w * w)
        x_r = int((1.0 - self.excl_corner_w) * w)

        dets = []
        catalog = getattr(self, 'object_catalog', None)
        max_diagonal = max(np.linalg.norm(m['dims']) for m in catalog.models) if catalog else np.linalg.norm(self.dims)
        # Filter whole SAM instances before geometry. A good cuboid fit alone
        # can also describe an unrelated object.
        target_pixels = None
        if self.target_color == "yellow":
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            target_pixels = cv2.inRange(hsv, (20, 80, 50), (38, 255, 255)) > 0
        color_rejected = 0
        filter_started = time.perf_counter()
        for m in masks:
            ys, xs = np.where(m)
            if len(ys) == 0:
                continue
            cy_m, cx_m = ys.mean(), xs.mean()
            if cy_m > y_lo and (cx_m < x_l or cx_m > x_r):
                continue                                   # gripper fingertip corner
            area = int(m.sum())
            if (target_pixels is not None and
                    np.count_nonzero(m & target_pixels) / area < self.target_color_min_fraction):
                color_rejected += 1
                continue
            if solver == 'legacy' and float((m & border).sum()) / area > self.max_border_frac:
                continue                                   # clipped by frame edge
            good = valid[ys, xs]
            if np.count_nonzero(good) < self.min_points:
                continue
            zmed = float(np.median(depth_m[ys[good], xs[good]]))
            if zmed < self.min_depth or zmed > self.max_depth:
                continue                                   # gripper (near) / background (far)
            if (solver == 'rgbd' and getattr(self, 'pose_lock_prefilter', True)
                    and self.pick_lock and self._lock_pos is not None):
                # A visible surface point can be up to half a box diagonal from
                # its center. Keep that margin: this is only a coarse lock filter.
                indices = np.flatnonzero(good)
                indices = indices[np.linspace(0,len(indices)-1,min(64,len(indices))).astype(int)]
                z = depth_m[ys[indices], xs[indices]]
                xyz = np.column_stack(((xs[indices]-cx)*z/fx, (ys[indices]-cy)*z/fy, z))
                observed_center = R_bc @ np.median(xyz,axis=0) + t_bc
                if np.linalg.norm(observed_center-self._lock_pos) > self.lock_radius + max_diagonal/2 + .005:
                    timing.counts['lock_prefilter_skips'] += 1
                    continue
            with timing.stage("pose"):
                if solver == 'legacy':
                    poly, nv = mask_to_polygon(m, target_counts=(6,))
                    fit = (fit_pose_rgb_then_depth(m, pc[m & valid], self.dims, self.K)
                           if poly is not None else None)
                elif catalog is not None:
                    fit = fit_cuboid_models(m, depth_m, catalog.models, self.K,
                                            config=getattr(self, 'pose_config', None))
                else:
                    fit = fit_cuboid_rgbd(m, depth_m, self.dims, self.K,
                                          config=getattr(self, 'pose_config', None))
            timing.counts["pose_attempts"] += 1
            if fit is None:
                continue
            if solver == 'rgbd':
                self._last_fit_details.append(fit)
                for key, value in fit['timing_ms'].items():
                    timing.ms['pose_'+key] = timing.ms.get('pose_'+key,0.) + value
                if 'vertex_count' in fit:
                    timing.counts['vertices_'+str(fit['vertex_count'])] += 1
                timing.counts['pose_'+fit['reason']] += 1
                if not fit['accepted']:
                    continue
                if catalog:
                    timing.counts['model_'+fit['model_id']] += 1
            if not (np.isfinite(fit["t"]).all() and np.isfinite(fit["R"]).all()
                    and math.isfinite(fit["combined"]) and math.isfinite(fit["sdf_err_mm"])):
                continue
            if (fit["combined"] > self.max_combined
                    or fit["sdf_err_mm"] > self.max_sdf_mm):
                continue                                   # noise / bad fit
            if workspace is not None and not workspace.allows_pick(R_bc @ fit['t'] + t_bc):
                continue
            dets.append((fit, m))

        timing.ms['instance_filter'] = max(0., (time.perf_counter()-filter_started)*1000. - timing.ms.get('pose',0.))

        if not dets:
            if color_rejected:
                self.get_logger().info(
                    f"No {self.target_color} cuboid passed gates; "
                    f"color rejected {color_rejected} instance(s)", throttle_duration_sec=5.0)
            self._note_miss()
            self._maybe_debug(bgr, None)
            return

        # --- choose the target ------------------------------------------------
        locked = None
        if self.pick_lock and self._lock_pos is not None:
            distance = lambda d: float(np.linalg.norm(R_bc @ d[0]["t"] + t_bc - self._lock_pos))
            near = min(dets, key=distance)
            if distance(near) <= self.lock_radius:
                locked = near
            else:
                # An unrelated survivor must not steal a temporarily lost target.
                self._note_miss()
                self._maybe_debug(bgr, None)
                return
        if locked is not None:
            fit, m = locked
        elif self.pick_select == "nearest":
            fit, m = min(dets, key=lambda d: float(d[0]["t"][2]))
        elif self.pick_select == "largest":
            fit, m = max(dets, key=lambda d: int(d[1].sum()))
        elif self.pick_select == "highest":
            # World height of the cuboid's highest corner, not camera depth.
            def top(d):
                f = d[0]
                r = R_bc @ f['R']
                return float((R_bc @ f['t'] + t_bc)[2] +
                             np.dot(np.abs(r[2]), f.get('dims', self.dims))/2)
            fit, m = max(dets, key=top)
        else:                                              # best_fit
            fit, m = min(dets, key=lambda d: float(d[0]["combined"]))

        # GPU work can make a frame stale even if it was fresh on entry.
        if self._pose_is_stale(color_msg.header.stamp):
            timing.outcome = 'stale_after_processing'
            self._note_miss()
            self._maybe_debug(bgr, None)
            return
        tc = R_bc @ np.asarray(fit["t"], dtype=np.float64) + t_bc
        selected_dims = fit.get('dims', self.dims)
        selected_id = fit.get('model_id', 'single')
        same_model = selected_id == getattr(self, '_lock_model_id', None)
        if locked is None or not same_model:
            self._prev_box_R = self._prev_z = self._prev_x = None
            self._track_id = getattr(self, '_track_id', 0) + 1
        if self._prev_box_R is not None:
            fit = dict(fit, R=align_symmetric_rotation(fit['R'], R_bc.T @ self._prev_box_R, selected_dims))
        self._prev_box_R = R_bc @ fit['R']
        self._lock_model_id = selected_id
        self._last_selected_fit = fit
        self._lock_pos = tc if self._lock_pos is None else 0.6 * self._lock_pos + 0.4 * tc
        self._lock_miss = 0
        self._lock_seen = time.monotonic()

        up_cam = R_bc.T @ np.array([0., 0., 1.]) if transform is not None else None
        pos, quat, R_out, up_axis = to_grasp_pose(
            fit["R"], fit["t"], selected_dims, up_cam=up_cam,
            prev_z=None if self._prev_z is None else R_bc.T @ self._prev_z,
            prev_x=None if self._prev_x is None else R_bc.T @ self._prev_x,
            surface_offset=self.surface_offset)
        self._prev_z, self._prev_x = R_bc @ R_out[:, 2], R_bc @ R_out[:, 0]

        msg = ArucoMarkers()
        msg.header.stamp = color_msg.header.stamp        # controller TF-looks up at this time
        msg.header.frame_id = color_msg.header.frame_id or "camera_color_optical_frame"
        msg.marker_ids = [self.pick_id]
        msg.poses = [_pose(pos, quat)]
        # Keep typed observations away from old controllers that interpret all
        # /aruco_markers picks as the original fixed-size box. Place-only legacy
        # output remains available for non-batch mode.
        if catalog is None:
            self.pub.publish(msg)
        else:
            self.model_markers_pub.publish(msg)
        if catalog is not None and getattr(self, 'target_pub', None) is not None:
            packet = dict(version=1, catalog_digest=catalog.digest, model_id=selected_id,
                          dims_m=list(selected_dims), track_id=self._track_id,
                          stamp_ns=Time.from_msg(msg.header.stamp).nanoseconds,
                          frame_id=msg.header.frame_id, position=np.asarray(pos).tolist(),
                          orientation=np.asarray(quat).tolist(),
                          box_in_grasp_position=(R_out.T @ (fit['t']-pos)).tolist(),
                          box_in_grasp_orientation=mat_to_quat(R_out.T @ fit['R']).tolist())
            if sweep_scene is not None:
                packet['workspace_observation'] = sweep_scene
            self.target_pub.publish(String(data=json.dumps(packet, separators=(',',':'), allow_nan=False)))
        timing.outcome = "published"
        timing.counts["accepted_instances"] = len(dets)
        if self.max_pose_age_s > 0:
            timing.ms["capture_to_publish"] = (self.get_clock().now().nanoseconds-Time.from_msg(color_msg.header.stamp).nanoseconds)/1e6
        self.get_logger().info(
            f"pick id{self.pick_id}: t_cam=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f})m "
            f"model={selected_id} dims_mm={[round(v*1000,1) for v in selected_dims]} "
            f"up-face={'LWH'[up_axis]}({selected_dims[up_axis]*1000:.1f}mm) "
            f"contour={fit['contour_err_px']:.1f}px depth_fit={fit['sdf_err_mm']:.1f}mm "
            f"{'[locked] ' if locked is not None else ''}"
            f"({len(dets)} cuboid(s) pass gate)", throttle_duration_sec=1.0)
        self._maybe_debug(bgr, (fit, pos, R_out))

    def _reset_lock(self):
        self._lock_pos = self._prev_z = self._prev_x = None
        self._prev_box_R = self._lock_model_id = None
        self._lock_seen = None
        self._lock_miss = 0

    def _note_miss(self):
        self._lock_miss += 1
        if (self._lock_miss > self.lock_timeout or
                (self._lock_seen is not None and
                 time.monotonic() - self._lock_seen > self.lock_timeout_s)):
            self._reset_lock()

    def _pose_is_stale(self, stamp):
        if self.max_pose_age_s == 0:
            return False
        age = (self.get_clock().now().nanoseconds - Time.from_msg(stamp).nanoseconds) / 1e9
        if age < -0.05 or age > self.max_pose_age_s:
            self.get_logger().warn(f"image age {age:.3f}s outside pose deadline",
                                   throttle_duration_sec=5.0)
            return True
        return False

    def _base_from_camera(self, cam_frame, stamp):
        """Use the acquisition-time transform only. Missing TF suppresses robot output."""
        if not self.use_tf_up:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame, cam_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=0.05))
        except TransformException as exc:
            self.get_logger().warn(f"No image-time TF {cam_frame}->{self.base_frame}: {exc}",
                                   throttle_duration_sec=5.0)
            return None
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        t = tf.transform.translation
        return R, np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------
    def _publish_place(self):
        msg = ArucoMarkers()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.place_frame
        msg.marker_ids = [self.place_id]
        msg.poses = [_pose(self.place_xyz, quat_from_yaw(self.place_yaw))]
        self.pub.publish(msg)

    def _maybe_debug(self, bgr, chosen):
        timing = getattr(self, '_frame_timing', None)
        if timing is None:
            return self._draw_debug(bgr, chosen)
        with timing.stage('debug'):
            return self._draw_debug(bgr, chosen)

    def _draw_debug(self, bgr, chosen):
        if self.dbg_pub is None:
            return
        vis = bgr.copy()
        # draw the excluded gripper-corner rectangles (detections whose centroid
        # lands here are dropped)
        h, w = vis.shape[:2]
        y_lo = int((1.0 - self.excl_corner_h) * h)
        x_l = int(self.excl_corner_w * w)
        x_r = int((1.0 - self.excl_corner_w) * w)
        for (a, b) in [((0, y_lo), (x_l, h)), ((x_r, y_lo), (w, h))]:
            cv2.rectangle(vis, a, b, (0, 0, 255), 2)
        if chosen is not None and self.K is not None:
            fit, pos, R_out = chosen
            # yellow: the raw cuboid fit (shape/depth). RGB axes: the PUBLISHED
            # grasp frame -- blue must sit on the box's top face pointing up.
            dims = fit.get('dims', self.dims)
            draw_wireframe(vis, fit["t"], fit["R"], dims, self.K)
            draw_axes(vis, np.asarray(pos), R_out, self.K, length=max(dims) * 0.8)
            label = f"{fit.get('model_id','single')}  " + ' x '.join(f'{v*1000:g}' for v in dims) + ' mm'
            cv2.putText(vis, label, (12,24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,220,255), 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        out.header = self._debug_header
        self.dbg_pub.publish(out)

    def destroy_node(self):
        self._stop = True
        try:
            self._worker.join(timeout=1.0)
        except RuntimeError:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = CuboidPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
