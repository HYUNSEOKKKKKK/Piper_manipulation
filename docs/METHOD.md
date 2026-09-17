# Perception and control

The detector is the YOLO-based `ObjectAwareModel` provided with MobileSAMv2.
Its boxes prompt the TinyViT encoder and prompt-guided mask decoder. The model
loads once, the predictor/backend is reused, and every processed image gets new
inference. No detector outputs are reused as if they were new observations.

For each surviving mask, the estimator builds a silhouette and sparse, eroded
interior depth observations. Invalid depth and discontinuities near boundaries
are excluded. Hypotheses use known cuboid dimensions: six-corner edge-graph PnP,
planar face PnP where available, and depth-supported geometry seeds. Four-, five-
and six-vertex observations do not all go through one six-point correspondence
case; the RGB-D backend also supports other silhouette counts. PnP itself uses
2D–3D correspondences and camera intrinsics, not depth.

Candidate poses are ranked by silhouette/depth consistency and refined under a
bounded budget (up to 192 depth samples and four refined candidates in the current
default configuration). Geometry observations are shared across size models.
An optional C++ residual accelerates the same objective, with a NumPy fallback.
Exact cuboid symmetries are handled per model: unequal axes cannot be arbitrarily
exchanged. Similar poses or model identities with insufficient evidence are
rejected rather than assigned a confident target.

`R,t` describe the box centre in the camera optical frame; axes have their normal
ROS optical convention. A separate adapter selects an upward-facing grasp frame,
using timestamped TF and gravity for manipulation. The default marker origin is
5 mm outside that face. Controller grasp penetration is a separate parameter;
the physical wrappers currently use `grasp_offset=-0.030` m. These values must be
calibrated with the jaw geometry. They do not impose a flat tabletop prior on the
cuboid pose estimate. Arbitrary pose estimation does not imply arbitrary grasp
reachability or reliable inference under complete occlusion.

The ROS bridge publishes `/cuboid_pose_bridge/target` (JSON model/track/pose and
geometry metadata), compatible marker messages, `/cuboid_pose_bridge/debug_image`
and `/cuboid_pose_bridge/performance`. RViz's wrist image display can use the debug
image topic to show the fitted cuboid and frame on the RGB view. TF transforms the
observation into `base_link`; the controller requires fresh, consistent observations
before freezing a target for motion.

MoveIt 2 plans with OMPL RRTConnect and `pick_ik`. The feed approach and retreat
use complete Cartesian paths; longer transfers use collision-checked joint
trajectories. Time parameterization respects the configured joint limits and
speed/acceleration scaling. The joint trajectory controller feeds the guarded
forwarder; AgileX's driver/SDK sends CAN commands to the PiPER. The precise servo
law inside the firmware is not implemented by this repository.

## Timing and evidence

`process` / `process_hz` measures processed image frames per second over the
bridge timing window. It is different from camera arrival rate, accepted target
rate, geometric solver milliseconds, and pick-and-place cycles per second.
The node uses a latest-frame buffer, so slow inference can skip input frames.
`max_rate_hz=30` is a ceiling, not a promise of 30 FPS. The detailed timing message
separates detector, SAM, pose fitting and publication costs; processing many masks
can change throughput.

The development archive contains a 23-minute two-model live run: median 7.7 Hz,
5th–95th percentile 6.3–10.3 Hz over 274 overlapping log windows, on an RTX 2070
SUPER (8 GiB) and i9-9900K. These are historical deployment measurements with
scene-dependent workload, not independent repetitions or a hardware benchmark
shipped in this repository. Raw logs/videos and the manuscript remain separate.
No ground-truth real-camera pose accuracy or universal grasp-success claim is
made from those logs.

The included CPU example has independent synthetic geometry ground truth. Tests
cover noisy depth, boundary corruption, partial silhouettes, exact symmetries,
model ambiguity, command gating, temporal rejection and interrupted-cycle state.
Mocked controller tests validate software decisions, not physical collision safety.
