# Piper Manipulation workspace

Read [RESUME.md](RESUME.md) before maintenance and `.local/README.md` when present.

- Both halves belong to this checkout: `perception/` implements RGB-D SE(3), segmentation and the ROS pose bridge; `ros2/piper_pnp/` implements MoveIt planning, control and launch configuration.
- Perception uses the dedicated Python 3.10 Conda environment `piper_manipulation`. Robot control/build uses the system Python in `.venv-robot` with ROS 2 Humble. Do not borrow the DeFM/locomotion environment.
- The launch scripts select the active named Conda environment or the project-local `.venv-perception` fallback. Explicit `PIPER_PERCEPTION_PYTHON` overrides remain supported.
- Vendor revisions and model hashes are pinned in `third_party/`. Keep the deployment patch and checksum verification; do not replace the vendored Ultralytics fork with a pip package.
- Preserve camera/TCP/workspace/gripper calibration and the current perception/controller behavior during organization work.
- `.local/archive/` can contain the original experimental Git repository, captured data, models and transfer bundles. These are research records despite being ignored. Do not use blanket `git clean -fdx`.
- Run the default CPU suite, native build, checkpoint verification and release audit for tooling changes. ROS-dependent checks require the installed system ROS packages and a built `.workspace`; report skipped checks accurately. Cleanup validation uses offline data and mocked execution, without starting hardware.
