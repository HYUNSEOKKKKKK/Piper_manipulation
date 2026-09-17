#!/usr/bin/env bash
# 이 PC의 MoveIt/드라이버 실행. 기본값은 preview.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:-preview}"
requested_mode="$mode"
continuous_feed=false
if [ "$mode" = feed-auto ]; then
  mode=feed
  continuous_feed=true
fi
MODELS_FILE="${PIPER_OBJECT_MODELS:-$PIPER_ROOT/config/object_models.json}"
if [ "$#" -gt 0 ]; then shift; fi
confirm=true
loop=false
batch_args=()
case "$mode" in
  preview) preview=true; gripper=false; offset=0.02 ;;
  real|hover) preview=false; gripper=false; offset=0.02 ;;
  grasp) preview=false; gripper=true; offset=-0.030 ;;
  batch|feed|sweep)
    preview=false; gripper=true; offset=-0.030; confirm=false; loop=true
    workspace_mode="$mode"
    if [ "$continuous_feed" = true ]; then workspace_mode=feed_auto; fi
    batch_config="${PIPER_BATCH_CONFIG:-$PIPER_ROOT/config/${workspace_mode}_workspace.json}"
    if [ ! -f "$batch_config" ]; then
      echo "연속 이송의 놓기 위치 설정이 필요합니다: $batch_config"
      echo '설정 예시와 실행 안내: docs/ROBOT.md'
      exit 2
    fi
    /usr/bin/python3 - "$batch_config" "$mode" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    feed = json.load(stream).get('operator_feed', False)
if feed is not (sys.argv[2] == 'feed'):
    raise SystemExit('feed/batch 실행 모드와 operator_feed 설정이 다릅니다. 시작하지 않습니다.')
PY
    batch_args=("batch_config_file:=$batch_config")
    if [[ "$mode" = sweep || "$mode" = feed ]]; then
      batch_args+=(velocity_scaling:=0.50 acceleration_scaling:=0.25)
    fi
    if [ "$mode" = sweep ]; then
      batch_args+=(gripper_settle_time:=0.35)
    fi
    ;;
  *) echo '사용: bash scripts/run_robot.sh [preview|hover|grasp|batch|feed|feed-auto|sweep] [use_rviz:=false]'; exit 2 ;;
esac
exec 9>/tmp/manipulator-robot.lock
flock -n 9 || { echo '이미 run_robot.sh가 실행 중입니다.'; exit 1; }
if pgrep -af '/agx_arm_ctrl/[a]gx_arm_ctrl_single'; then
  echo '기존 팔 드라이버가 남아 있습니다. 해당 실행 터미널에서 정상 종료 후 다시 실행하세요.'
  exit 1
fi
can_status=$(ip -details link show "$PIPER_CAN") || exit 1
if [[ "$can_status" != *"state UP"* ]]; then
  echo 'can0이 꺼져 있습니다. 먼저 CAN 연결을 활성화하세요.' >&2
  exit 1
fi
if [[ "$preview" == false && "$can_status" != *"can state ERROR-ACTIVE"* ]]; then
  echo 'CAN 상태가 정상 범위가 아니어서 실기 실행을 시작하지 않습니다.' >&2
  echo "$can_status" >&2
  echo '모든 로봇 프로그램 종료 후 can0을 down/up하고 상태를 다시 확인하세요.' >&2
  exit 1
fi
source_ros
source_workspace
echo "모드=$requested_mode / preview=$preview / 그리퍼=$gripper / 파지 오프셋=${offset}m"
exec ros2 launch piper_pnp pnp_real_cuboid.launch.py \
  object_models_file:="$MODELS_FILE" \
  can_port:="$PIPER_CAN" auto_start:=false step_confirm:="$confirm" loop_mode:="$loop" \
  continuous_feed:="$continuous_feed" \
  preview_only:="$preview" use_real_gripper:="$gripper" grasp_offset:="$offset" \
  "${batch_args[@]}" "$@"
