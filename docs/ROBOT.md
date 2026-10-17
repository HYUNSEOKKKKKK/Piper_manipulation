# Physical robot operation

Start with the CPU example and complete [installation](INSTALL.md). Robot
operation requires an unobstructed workspace, functional stop controls and
hardware-specific calibration. Keep hands out of the motion volume while a
sequence or reset is active.

## Equipment for the documented pipeline

The complete vision-to-arm example uses:

- An AgileX PiPER arm with the parallel gripper, its power supply and a
  Linux SocketCAN-compatible connection between the PC and arm.
- A wrist-mounted Intel RealSense D435, its USB connection and a measured mount
  transform. The example consumes RGB, aligned depth and camera intrinsics.
- An Ubuntu 22.04 PC with the installed ROS 2 Humble workspace and the documented
  CUDA-capable perception environment.
- Boxes with known dimensions matching the shared object catalog, plus measured
  source/drop regions and a usable robot stop control.

The robot arm alone does not supply the images, calibration or work-area
measurements required by this perception algorithm. The supplied dimensions and
mount/workspace values describe the development setup; adapt them below.

## Check the ROS stack without hardware

After installation, this launch uses only `mock_components/GenericSystem`.
It starts MoveIt and the virtual controllers, with automatic sequence start and
fake camera targets disabled. It does not start an arm or camera driver.

```bash
cd ~/Piper_manipulation
source /opt/ros/humble/setup.bash
source .workspace/install/setup.bash
export ROS_DOMAIN_ID=163 ROS_LOCALHOST_ONLY=1
ros2 launch piper_pnp pnp_demo.launch.py use_rviz:=false auto_start:=false use_fake_aruco:=false
```

In another terminal, source the same ROS/workspace, use the same two environment
variables, and run `ros2 control list_controllers`. `arm_controller`,
`gripper_controller` and `joint_state_broadcaster` should be active. Ctrl-C ends
the virtual stack. Use fresh terminals for the physical session below so this
diagnostic ROS domain does not accidentally differ between physical nodes.

## Camera USB permissions

Install the RealSense USB permission rules once per PC, before attaching the
camera. The ROS camera packages on the verified workstation supplied the SDK
libraries but did not install these rules. Reuse an existing RealSense rules
installation when present; the commands below are for a PC without one.

The following file is the official SDK 2.58.4 rule set, matching the validated
ROS SDK version. Download it into the project and verify its hash before installation:

```bash
mkdir -p .local/system
curl -fL https://raw.githubusercontent.com/realsenseai/librealsense/v2.58.4/config/99-realsense-libusb.rules \
  -o .local/system/99-realsense-libusb.rules
printf '%s\n' '34c287000862710b60732aea942bda6f51faf7436b5b53f66fbfd021bc8c22dd  .local/system/99-realsense-libusb.rules' | sha256sum --check
sudo install -m 0644 .local/system/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
```

The rules apply when the camera is next connected. See the
[official SDK installation instructions](https://github.com/realsenseai/librealsense/blob/v2.58.4/doc/installation.md).
Actual USB enumeration and RGB/aligned-depth streaming still need the camera.

## Configure the actual setup

- Measure camera extrinsics in `ros2/piper_pnp/urdf/custom_piper_d435.xacro`.
  The supplied camera-to-link6 origin `(-.080, -.015, .035)` m, RPY `(0, -1.2, 0)`
  and TCP offset `0.1425` m describe the development mount. Rebuild after edits.
- Check `config/object_models.json` against your boxes. All geometry uses metres.
  Add a model by extending this catalog; both terminals must use the same file.
- Copy a workspace to `config/local_workspace.json` and edit source bounds,
  destination, table height and obstacle envelopes. Export
  `PIPER_BATCH_CONFIG="$PWD/config/local_workspace.json"` in **both** terminals.
  Do not casually alter a workspace that has occupied destinations or an
  interrupted transfer: its adjacent `.state.json` protects progress across
  restarts and deliberately detects incompatible configuration changes.
- Check camera-ready/home joint poses, collision geometry, jaw aperture and
  offsets in `ros2/piper_pnp/config/pnp_params.yaml` on the actual arm. The camera
  mount is visual-only in the supplied model; its collision volume is not modeled.
- Verify CAN at 1 Mbit/s and PC control mode. The wrappers reject warning/passive
  CAN states and retained teaching mode. They do not toggle hardware modes on launch.

## First CAN connection

With the arm powered and the supported CAN adapter connected, identify the
interface exposed by Linux:

```bash
ip -brief link
```

Use the actual CAN interface name. With all robot programs stopped, configure
it at 1 Mbit/s; the following example uses `can0`:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
ip -details -statistics link show can0
```

The interface must be UP and ERROR-ACTIVE for physical execution. If Linux has
not created a CAN interface, install/configure the adapter's driver first.
For an interface named differently, export `PIPER_CAN=<actual-name>` in each
execution terminal. Check the arm's PC control mode before launch.

## Three terminals

After installation validation and physical configuration, open three terminals.
Use the path where you cloned the repository; these examples assume
`~/Piper_manipulation`.

```bash
# Terminal 1: robot driver, MoveIt, camera and controller
cd ~/Piper_manipulation
bash scripts/run_robot.sh feed-auto
```

```bash
# Terminal 2: perception
cd ~/Piper_manipulation
conda activate piper_manipulation
bash scripts/run_bridge.sh feed-auto
```

```bash
# Terminal 3: operator console
cd ~/Piper_manipulation
bash scripts/run_console.sh
```

The scripts source ROS and this project's built workspace. Terminal 1 uses the
robot Python entry points built with `.venv-robot`; terminal 2 selects the named
Conda Python. No manual activation of `.venv-robot` is needed.

Click terminal 3, clear the workspace, then press **c once, without Enter**.
The controller observes a stable source target, picks it, retreats, moves to the
fixed destination, opens the gripper and returns to camera-ready. It repeats
without home or another approval between boxes, up to `max_transfers` (default
10). With no accepted source target it waits; a fresh valid observation may
resume the loop. **Waiting for perception is not a safe state for adding boxes.**
Stop before entering the workspace.

The default fixed destination is `(0.410, -0.180, 0.150)` m in `base_link`.
Its orientation is unconstrained during planning; there is no placement descent.
This height denotes the TCP target, not a guaranteed gap between a box and the
pile. Held-box geometry, gripper clearance, depth coverage and the growing goal
region are checked. Insufficient clearance can stop a run before 10 transfers.
The system does not guarantee removal of every box in arbitrary clutter.

| Mode | Behavior |
|---|---|
| `preview` | Hardware-connected planning preview; physical command forwarding stays guarded. Not a standalone simulator. |
| `hover` | Physical approach without gripping. |
| `grasp` | Original sequence with step approval and real gripper. |
| `feed` | One approved original feed cycle per `c`; precise placement slots. |
| `feed-auto` | One `c`, repeated feed picking, one fixed air-drop goal, no home between objects. |
| `batch`, `sweep` | Retained experimental variants; `feed-auto` is the documented continuous mode. |

`feed` / `feed-auto` use velocity scaling `0.50` and acceleration scaling `0.25`;
the launch defaults remain `0.1`. These are fractions of configured joint
limits, not 50% of all possible hardware motion. Cartesian completeness and
collision checks remain active. Extra launch arguments can lower speed, e.g.
`bash scripts/run_robot.sh feed-auto velocity_scaling:=0.1 acceleration_scaling:=0.1`.

## Console

| Key | Action |
|---|---|
| `c` / `ㅊ` | Approve the next step/cycle or start continuous processing |
| `h`, then `y` | Confirmed home/reset after clearing held and goal boxes; successful reset clears transfer count |
| `r` / `ㄱ` | Request stop release where supported; a stopped continuous run needs reset/restart |
| `q` / `ㅂ` | Exit this console; do not use it as the emergency stop key |
| Other keys, including Space and Enter | Stop request |

The stop closes the external control gate, requests hold, and cancels the active
controller sequence. Console keys require focus. Software stop is not a substitute
for the robot's physical stop. Do not press `h` → `y` until the gripper, goal and
home path are clear; that confirmation can move the robot.

## Recovery and common failures

**CAN `ERROR-WARNING` / `ERROR-PASSIVE`:** check power, connector, wiring and bit
rate. After all robot programs are stopped, the interface can be reset manually:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
ip -details -statistics link show can0
```

Use your actual interface name. Repeated errors indicate a communication issue;
down/up is not a repair for wiring or power. `recover_control.sh` does **not** reset CAN.

**Teaching recording ended but control mode remains teaching:** with healthy CAN,
a running driver, no active motion and hands clear, use
`bash scripts/recover_control.sh`. It checks feedback, attempts PC control while
holding the measured posture, and leaves the experiment waiting. It is a robot
command, not a general-purpose reset or motion start.

**No controller / settings unavailable:** terminal 1 must still contain the running
controller. Read its first error. Relaunch it if it exited, then reopen terminal 3.
Do not spawn a second driver while an old one is running.

**Cartesian path incomplete:** this is a planning failure along the interpolated
approach/retreat. It may result from collision, wrist/joint limits, the seed or a
bad target; it is not proof of slow joint servoing. Partial paths are not executed.
Check target overlays, actual box dimensions, TCP/camera calibration and logs.

**Unstable target:** four distinct timestamped valid observations are required
within a 3 s window by the current configuration; their position and symmetry-aware
orientation must meet the configured tolerance. Duplicate messages are not extra
observations. The position limit is 20 mm and angular limit 15°, not whichever
measured deviation happens to appear in a warning. Bad depth, ambiguity and loss
of tracking can delay acceptance.
