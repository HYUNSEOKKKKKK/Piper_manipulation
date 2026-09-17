# Physical robot operation

Start with the CPU example and complete [installation](INSTALL.md). Robot
operation requires an unobstructed workspace, functional stop controls and
hardware-specific calibration. Keep hands out of the motion volume while a
sequence or reset is active.

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

## Three terminals

Run at the repository root after configuration:

```bash
# 1
bash scripts/run_robot.sh feed-auto
# 2
bash scripts/run_bridge.sh feed-auto
# 3
bash scripts/run_console.sh
```

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
