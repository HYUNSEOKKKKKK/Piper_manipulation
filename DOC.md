# Documentation

Start with the **[README demos and quick start](README.md)** or the
**[한국어 안내](docs/README.ko.md)**.

| I want to… | Start here |
|---|---|
| Try pose estimation without hardware | [CPU quick start](README.md#try-perception-first--cpu-only) |
| Install ROS, ML dependencies and models | [Installation](docs/INSTALL.md) |
| Configure my camera mount, TCP and boxes | [Hardware configuration](docs/ROBOT.md#configure-the-actual-setup) |
| Run the three-terminal robot experiment | [Robot operation](docs/ROBOT.md#three-terminals) |
| Recover from CAN, teaching-mode or trajectory errors | [Recovery](docs/ROBOT.md#recovery-and-common-failures) |
| Understand SE(3), refinement and timing | [Method and performance](docs/METHOD.md) |
| Check what was validated | [Release validation](docs/VALIDATION.md) |
| Inspect or reproduce the demo previews | [Media provenance](assets/README.md) |
| Extend the project | [Contributing](CONTRIBUTING.md) and [component licenses](NOTICE.md) |

## Source entry points

| Component | Code |
|---|---|
| RGB-D hypotheses, refinement and model selection | [`perception/cuboid_rgbd.py`](perception/cuboid_rgbd.py) |
| Cuboid correspondences and PnP baseline | [`perception/cuboid_pnp_correspondence.py`](perception/cuboid_pnp_correspondence.py) |
| Native geometry residual | [`perception/native/cuboid_residual.cpp`](perception/native/cuboid_residual.cpp) |
| YOLO/SAM inference, ROS input, target publishing and overlays | [`perception/ros_pose_bridge.py`](perception/ros_pose_bridge.py) |
| Observation acceptance and main FSM | [`piper_pnp_controller.py`](ros2/piper_pnp/piper_pnp/piper_pnp_controller.py) |
| Trajectory planning and execution interface | [`moveit_client.py`](ros2/piper_pnp/piper_pnp/moveit_client.py) |
| Fixed-position air drop | [`feed_drop.py`](ros2/piper_pnp/piper_pnp/feed_drop.py) |
| Real command guard and forwarding | [`control_guard.py`](ros2/piper_pnp/piper_pnp/control_guard.py), [`real_control.py`](ros2/piper_pnp/piper_pnp/real_control.py) |
| Box identities and dimensions | [`config/object_models.json`](config/object_models.json) |
| Default continuous-feed workspace | [`config/feed_auto_workspace.json`](config/feed_auto_workspace.json) |

Use `python -m pytest -q` for the CPU suite after core installation. The full
mocked ROS suite requires the environments described in [INSTALL.md](docs/INSTALL.md#4-validation).
