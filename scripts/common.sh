#!/usr/bin/env bash
# Shared path configuration. Source this file; it does not contact hardware.
PIPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_WS="${PIPER_WS:-$PIPER_ROOT/.workspace}"
PIPER_CAN="${PIPER_CAN:-can0}"
if [[ -z "${PIPER_PERCEPTION_PYTHON:-}" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" == piper_manipulation && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    PIPER_PERCEPTION_PYTHON="$CONDA_PREFIX/bin/python"
  else
    PIPER_PERCEPTION_PYTHON="$PIPER_ROOT/.venv-perception/bin/python"
  fi
fi
source_ros() {
  local setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  [[ -f "$setup" ]] || { echo "ROS setup missing: $setup" >&2; exit 2; }
  source "$setup"
}
source_workspace() {
  [[ -f "$PIPER_WS/install/setup.bash" ]] || { echo "Build the workspace first; see docs/INSTALL.md: $PIPER_WS" >&2; exit 2; }
  source "$PIPER_WS/install/setup.bash"
}
