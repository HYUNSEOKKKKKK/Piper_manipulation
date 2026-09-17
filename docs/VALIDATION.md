# Release validation — 2026-09-17

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
