#!/usr/bin/env bash
# Pure geometry acceleration; compilation only, no ROS/hardware processes.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$task_root/../perception/native/build"
g++ -O3 -std=c++17 -fPIC -shared -Wall -Wextra -Werror \
  "$task_root/../perception/native/cuboid_residual.cpp" -o "$task_root/../perception/native/build/libcuboid_residual.so"
sha256sum "$task_root/../perception/native/cuboid_residual.cpp" | cut -d ' ' -f 1 > "$task_root/../perception/native/build/source.sha256"
