# Validation records

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
| ROS workspace build / full mocked ROS suite | Pending: system ROS 2 Humble is not installed on this workstation; installation requires administrator authentication |

The model smoke check extracted the unchanged `Segmenter` class for offline
execution; it did not start the ROS bridge or validate live camera topics.
The synthetic geometry check measures generated input, not real-camera accuracy.
The September ROS results below describe a previous release validation and do
not establish ROS readiness on this workstation. See [installation](INSTALL.md)
and [workspace layout](WORKSPACE.md) for the remaining system setup.

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
