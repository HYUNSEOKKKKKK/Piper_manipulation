# Installation

Commands below assume Ubuntu 22.04, ROS 2 Humble and `/usr/bin/python3` 3.10.
Work from the repository root. Use the dedicated `piper_manipulation` Conda
environment for perception and the system Python's project-local `.venv-robot`
for ROS control. `.workspace`, `.venv-*` and `.local` are ignored by Git and do
not modify another robot workspace. The two environments belong to this project;
the system ROS installation remains under `/opt/ros/humble`.

## Start from a fresh clone

Have Ubuntu 22.04, Git, a Conda installation such as Miniforge, and administrator
access for the system packages below. For the documented live perception command,
also have an NVIDIA driver compatible with the CUDA 12.4 PyTorch build.

```bash
git clone https://github.com/HYUNSEOKKKKKK/Piper_manipulation.git
cd Piper_manipulation
```

Complete sections 1–4 in order, then follow [physical setup and operation](ROBOT.md).
A clone contains the perception/controller source, configuration, source pins and
model manifest. The commands below obtain the vendor sources and weights and
build them locally. A developer's private `.local/` folder is not needed.

## Where each dependency lives

| Component | Installation location | How it is selected |
|---|---|---|
| Perception Python, PyTorch and NumPy | Conda environment `piper_manipulation` | `conda activate piper_manipulation` |
| ROS 2 Humble, MoveIt, camera and ROS control packages | System `/opt/ros/humble` through apt | Launch/build scripts source ROS |
| Robot SDK and system-compatible Python | Project `.venv-robot/` | Build script selects this Python; no separate venv activation |
| ROS source, generated messages and build output | Project `.workspace/` | `bash scripts/build_workspace.sh` |

The project has one Conda environment. ROS installation is a system setup step;
the project build stays in this checkout. Activating Conda does not install ROS
or build the workspace. The launch scripts select their interpreters, so normal
operation does not require switching between Python environments by hand.

ROS can also be packaged entirely inside Conda with
[RoboStack](https://robostack.github.io/conda.html). That is a different
installation route from the Ubuntu ROS binaries used by this project; the
current launch/build scripts target `/opt/ros/humble`, and a RoboStack port has
not been validated. ROS's [Python package guidance](https://docs.ros.org/en/humble/How-To-Guides/Using-Python-Packages.html)
explains the interpreter compatibility requirement. Keep the control/build
Python based on `/usr/bin/python3`; the Conda perception bridge must pass the
ROS import and mocked checks below on the target machine.

## 1. ROS and system packages

Install [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
using the official Ubuntu 22.04 apt instructions, including the ROS apt repository
and `ros-humble-desktop`. Then install this project's additional packages:

```bash
sudo apt update
sudo apt install --no-remove build-essential git python3-venv python3-pip python3-colcon-common-extensions \
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

If using rosdep for additional OS dependencies, run `sudo rosdep init` once on a
new system if `/etc/ros/rosdep/sources.list.d/20-default.list` does not exist.
Then update it and resolve this workspace:

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
.venv-robot/bin/python -c 'import rclpy, pyAgxArm; print("Robot Python imports OK")'
python -c 'import rclpy, cv_bridge; from agx_arm_msgs.msg import AgxArmStatus; print("Perception ROS imports OK")'
ros2 pkg prefix piper_pnp
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  ros2/piper_pnp/test tests/perception
```

These tests replace publishers, service/action clients and motion execution with
local test objects. They do not execute a physical robot experiment. The CPU CI
runs without vendor checkouts, weights, ROS, CUDA or hardware.

Check the GPU used by the default live bridge before connecting the experiment:

```bash
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required by the default live bridge"; print(torch.cuda.get_device_name(0))'
```

An import failure here is an environment issue to resolve before launch. The
Python 3.10 and NumPy 1.x pins are necessary for the documented combination;
matching the version numbers alone does not prove ROS binary compatibility.
The current workstation's system ROS installation and these ROS-dependent checks
are still pending; [validation records](VALIDATION.md) distinguish that status
from the earlier release's successful ROS build and mocked suite.

After these checks, configure the camera mount, TCP, object models, workspace
and CAN connection in [ROBOT.md](ROBOT.md). Those values belong to the physical
setup and cannot be supplied by cloning the repository alone.

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
