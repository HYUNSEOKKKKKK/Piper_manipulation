# Installation

Commands below assume Ubuntu 22.04, ROS 2 Humble and `/usr/bin/python3` 3.10.
Work from the repository root. Use the dedicated `piper_manipulation` Conda
environment for perception and the system Python's project-local `.venv-robot`
for ROS control. `.workspace`, `.venv-*` and `.local` are ignored by Git and do
not modify another robot workspace. The two environments belong to this project;
the system ROS installation remains under `/opt/ros/humble`.

## 1. ROS and system packages

Install [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
first, then:

```bash
sudo apt update
sudo apt install build-essential git python3-venv python3-pip python3-colcon-common-extensions \
  python3-rosdep python3-pytest python3-opencv python3-scipy can-utils \
  ros-humble-moveit ros-humble-pick-ik ros-humble-ros2-control \
  ros-humble-ros2-controllers ros-humble-realsense2-camera \
  ros-humble-cv-bridge ros-humble-message-filters ros-humble-xacro
python3 scripts/fetch_dependencies.py
```

The fetcher initializes pinned submodules (including AgileX URDF/meshes),
prints exact revisions, does not start hardware and refuses to switch
an existing checkout to a different revision. It applies the small MobileSAMv2
patch idempotently. Use `--group perception`, `--group robot` or `--dry-run` as
needed. Do not replace the bundled Ultralytics fork with a recent pip release.

If using rosdep for additional OS dependencies, initialize it once if needed,
then update it and resolve this workspace:

```bash
rosdep update
rosdep install --from-paths .workspace/src --ignore-src -r -y --rosdistro humble
```

The dependency manifest pins Git source revisions. Ubuntu/ROS binary packages
still come from your Humble repositories; it is not a container or a bit-for-bit
lock of every system library.

## 2. Robot Python environment and build

```bash
/usr/bin/python3 -m venv --system-site-packages .venv-robot
.venv-robot/bin/python -m pip install --upgrade pip 'setuptools<81' wheel 'packaging>=24'
.venv-robot/bin/python -m pip install --no-build-isolation .workspace/vendor/pyAgxArm
bash scripts/build_workspace.sh
```

The SDK is installed in the robot venv; the colcon build uses that Python for
Python package entry points. `agx_arm_ctrl`, descriptions, ArUco message
interfaces and this project's `piper_pnp` are built together. `build_workspace.sh`
only builds: it never enables CAN, starts a driver or moves the arm.

## 3. Perception environment

Use the dedicated Python 3.10 Conda environment. If you created it for the CPU
example, activate it and continue with the pip commands:

```bash
conda create -n piper_manipulation python=3.10 pip -y
conda activate piper_manipulation
python -m pip install 'setuptools<81' wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-perception.txt
python3 scripts/fetch_weights.py
bash scripts/build_native.sh
```

Choose a GPU driver compatible with the selected PyTorch CUDA build. NumPy is
pinned below 2 because Humble's `cv_bridge` uses the NumPy 1.x ABI. Do not upgrade
it independently. The tested perception package versions are listed in
`requirements-perception.txt`; versions there describe the September 2026
release environment.

MobileSAM's encoder is downloaded from its official repository. The v2 detector
and decoder URLs point to an explicitly identified third-party mirror; exact
hashes are verified. The official Google Drive archive is an alternative:
[weights instructions](../weights/README.md). No YOLO/SAM fine-tuning is required.

## 4. Validation

```bash
conda activate piper_manipulation
python -m pytest -q
python examples/estimate_pose.py
python3 scripts/fetch_weights.py --verify
```

For the complete mocked suite, including ROS message types and the real bridge
methods, source the newly built workspace first:

```bash
source /opt/ros/humble/setup.bash
source .workspace/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  ros2/piper_pnp/test tests/perception
```

These tests replace publishers, service/action clients and motion execution with
local test objects. They do not execute a physical robot experiment. The CPU CI
runs without vendor checkouts, weights, ROS, CUDA or hardware.

The launch scripts select the active Conda environment when its name is
`piper_manipulation`. `run_bridge.sh` sources ROS and the project's built
workspace before starting that Python, making the Humble message modules
available without installing DeFM or borrowing another project's environment.
You can also use a system-site-packages `.venv-perception` as before, or choose
an interpreter explicitly with `PIPER_PERCEPTION_PYTHON`. Recreate local
environments and rebuild `.workspace` after moving to another machine.

## Optional paths

| Variable | Default |
|---|---|
| `PIPER_WS` | `<repo>/.workspace` for launch/build scripts |
| `PIPER_PERCEPTION_PYTHON` | Active `piper_manipulation` Conda Python; otherwise `<repo>/.venv-perception/bin/python` |
| `PIPER_ROBOT_PYTHON` | `<repo>/.venv-robot/bin/python` for build |
| `PIPER_CAN` | `can0` |
| `PIPER_OBJECT_MODELS` | `<repo>/config/object_models.json` |
| `PIPER_BATCH_CONFIG` | Matching `config/*_workspace.json` for the selected mode |
| `PIPER_WEIGHTS_DIR` | `<repo>/weights` |
| `MOBILE_SAM_ROOT` | `<repo>/third_party/MobileSAM/MobileSAMv2` |

The dependency fetcher intentionally creates the standard `.workspace`; to use a
custom workspace, place the pinned dependencies and this ROS package there and
set `PIPER_WS` for build/launch. An existing conda Python 3.10 environment can be
selected via `PIPER_PERCEPTION_PYTHON`; it must expose the same ROS/NumPy ABI.
