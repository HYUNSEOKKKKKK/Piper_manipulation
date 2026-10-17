# Project workspace

Use `Piper_manipulation/` as the single source checkout for both perception and
robot control. The program paths are derived from this checkout, so its location
does not depend on another research workspace.

```text
Piper_manipulation/
├── perception/                    # RGB-D SE(3), segmentation and ROS pose bridge
├── ros2/piper_pnp/                # MoveIt, arm/gripper control and launch files
├── config/                       # Object models, grasp/workspace configuration
├── weights/                      # Three verified model files, downloaded locally
├── third_party/MobileSAM/         # Pinned and patched model implementation
├── .workspace/                   # Pinned ROS vendor sources and colcon build
├── .venv-robot/                  # System-Python environment for ROS control
├── outputs/, captures/           # Generated results and experiment recordings
└── .local/                      # Optional local archive, inventories and logs
```

The dedicated Conda environment is named **`piper_manipulation`**. Activate it
with `conda activate piper_manipulation` for perception and CPU examples. Conda
stores the environment in its normal environment directory; it belongs to this
project and does not require DeFM. The robot Python environment is inside the
checkout because it uses Ubuntu's ROS Python packages. See [installation](INSTALL.md).

## Data flow

The live pipeline is YOLO box detection, MobileSAMv2 segmentation, metric RGB-D
cuboid SE(3) fitting, ROS target publication, frame conversion/pose acceptance,
MoveIt planning, and guarded PiPER arm/gripper commands. The object catalog in
`config/object_models.json` is shared by perception and control.

`examples/estimate_pose.py` exercises the SE(3) geometry without ROS, camera,
weights or an arm. The live ROS bridge and controller require the system ROS
installation and the built `.workspace`. Preparing code or models does not
mean that hardware operation has been tested.

## Preserve earlier research

When consolidating previous experiments, keep their source history, original
frames, model files and transfer bundles in `.local/archive/`. Preserve the
original directory layout and verify file hashes. Use the current public source
for execution, retaining older experiments as provenance rather than mixing
their implementations into the current pipeline.

`.local/`, downloaded weights/vendor sources and runtime output are excluded
from Git. Back them up separately; a Git clone does not restore research data.
On an existing workstation, `.local/README.md` describes actual archive paths,
environment records, migration links and checks performed. Recreate environments
and rebuild native/ROS outputs after transferring to a different machine.
