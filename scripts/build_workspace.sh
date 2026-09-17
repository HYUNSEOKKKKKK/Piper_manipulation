#!/usr/bin/env bash
# Builds packages only; never launches a node or changes CAN configuration.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source_ros
[[ -d "$PIPER_WS/src/piper_pnp" ]] || { echo 'Run scripts/fetch_dependencies.py --group robot first.' >&2; exit 2; }
robot_python="${PIPER_ROBOT_PYTHON:-$PIPER_ROOT/.venv-robot/bin/python}"
[[ -x "$robot_python" ]] || { echo 'Create .venv-robot as described in docs/INSTALL.md.' >&2; exit 2; }
export PATH="$(dirname -- "$robot_python"):$PATH"
cd "$PIPER_WS"
"$robot_python" -m colcon build --symlink-install \
  --packages-up-to piper_pnp --cmake-args "-DPython3_EXECUTABLE=$robot_python" "-DPYTHON_EXECUTABLE=$robot_python" "-DPython_EXECUTABLE=$robot_python" "$@"
