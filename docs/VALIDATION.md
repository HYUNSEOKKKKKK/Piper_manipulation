# Validation records

## Workstation ROS and GPU readiness — 2026-10-18

After the system ROS installation, the following checks ran on this workstation
with no physical arm, CAN adapter or RealSense connected. ROS node checks used
localhost and a separate ROS domain. Physical driver classes were imported but
never instantiated.

| Check | Result |
|---|---|
| System / workspace | Ubuntu 22.04, ROS 2 Humble; 6 project/vendor packages built with the project robot Python |
| Robot environment | SDK and driver/controller imports passed; setuptools 79.0.1 satisfies colcon-core's `<80` requirement; `pip check` passed |
| Robot model | URDF and SRDF generated; camera/TCP frames present, 15 URDF links |
| Perception environment | ROS messages and cv_bridge RGB/depth round trips passed in the dedicated Python 3.10 Conda environment |
| Actual perception node | Started and shut down successfully with both object models, native geometry and CUDA/FP16 enabled on RTX 3070 |
| Actual detector / SAM inference | 38 masks on the pinned upstream sample using CUDA/FP16 |
| Default suite with ROS sourced | 129 passed with pytest plugin auto-discovery disabled |
| Complete mocked ROS/controller/perception suite | 393 passed |
| Virtual hardware stack | `GenericSystem`, MoveIt and all three controllers active; joint plan and mock trajectory execution succeeded; Cartesian plan completed fully; mock gripper action succeeded |
| Preserved real RGB-D replay | 12 frames processed through the actual GPU segmenter and RGB-D solver; 2 frames emitted model-tagged targets, 10 were rejected by fitting gates |
| Camera software preparation | RealSense launch arguments loaded and executable shared-library dependencies resolved |

The RGB-D replay used camera-frame diagnostics with TF and frame-age gates
explicitly disabled, without changing production defaults. It had no external
pose ground truth; these 12 historical frames are not a pose-accuracy or task
success-rate benchmark. The replay, ROS node startup, mocked suite and mock
hardware exercise different parts of the system; they do not demonstrate a
live camera-to-physical-arm experiment.

CAN communication, camera streaming/USB permissions, capture-time TF, arm
firmware/mode, measured mount/TCP calibration and physical picking still require
the actual equipment. The original controller/perception algorithms and physical
calibration were retained. See [connection and operation](ROBOT.md).

## Workspace consolidation — 2026-10-18

The dedicated Python 3.10 Conda environment `piper_manipulation` was created
without borrowing an existing DeFM environment. Pinned source checkouts, the
robot SDK environment and all three model files were prepared under the project
root. No camera, ROS node, CAN configuration or robot command was started.

| Check | Result |
|---|---|
| Default CPU suite | 128 passed; 1 ROS-only test skipped, including 3 environment/path selection cases |
| Native residual build and synthetic pose example | Passed; generated case accepted with about 0.168 mm centre error and 0.451° symmetry-aware rotation error |
| Dedicated Conda and robot venv dependency checks | Both `pip check` commands passed |
| Three active checkpoints | Exact manifest sizes and SHA-256 matched |
| Pinned detector + SAM CPU inference | Actual `Segmenter` methods loaded all three checkpoints and produced 38 masks on the upstream bus sample |
| Earlier experiment preservation | All 901 files, including Git metadata, had matching SHA-256 before and after relocation; transfer bundle verified |
| ROS workspace build / full mocked ROS suite | Initially deferred for system installation; completed in the later workstation check above |

The model smoke check extracted the unchanged `Segmenter` class for offline
execution; it did not start the ROS bridge or validate live camera topics.
The synthetic geometry check measures generated input, not real-camera accuracy.
The September results below describe the previous release validation. The
workstation checks above record the completed local system setup and its limits.
See [installation](INSTALL.md) and [workspace layout](WORKSPACE.md).

## Release validation — 2026-09-17

Validation performed on the separate release checkout, without sending motion,
gripper, enable, stop or CAN commands to the physical robot:

| Check | Result |
|---|---|
| CPU geometry/configuration/release-tool suite | 125 passed; 1 ROS-only test skipped |
| Full mocked ROS/controller/perception suite | 393 passed |
| Native residual vs NumPy regression | Included and passed with the native library built |
| Synthetic tilted-box example | Accepted; about 0.168 mm centre error and 0.451° symmetry-aware rotation error in this generated case |
| Three checkpoint downloads | All downloaded from manifest URLs and matched exact size/SHA-256 |
| Pinned MobileSAM deployment patch | Applied successfully; repeated fetch recognizes the applied patch |
| Full detector + SAM CPU smoke | Real checkpoint loading and mask inference passed (38 masks on the upstream bus sample) |
| Separate colcon workspace | 6 packages built, including vendor interfaces, description, driver and `piper_pnp` |
| New workspace URDF/MoveIt configuration | Generated successfully with camera and TCP links |
| New workspace driver/SDK import | Passed |
| Shell syntax and tracked release audit | Passed |

The full perception tests used the existing Python 3.10 ML environment; ROS
packages were also built separately in a new robot venv. Vendor ROS/SDK source
checkouts for the build were copied through local Git clones at the manifest
revisions; the MobileSAM source and all checkpoints were obtained over the network.
This is not a from-scratch operating-system installation or a new physical robot
trial. The original operating workspace was not modified. GitHub CI separately
runs the core install/tests on an Ubuntu runner after publication.

The synthetic example is a functionality check, not a real-camera pose-accuracy
claim. Timing from a CPU smoke test is not the deployed GPU throughput. Original
third-party deprecation warnings remain; they did not cause the checks to fail.
