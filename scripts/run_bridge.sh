#!/usr/bin/env bash
# 같은 PC의 defm 환경에서 인식. 실기에서는 CUDA와 촬영 시각 TF가 필요하다.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
DIMS="[0.078, 0.035, 0.030]"
MODELS_FILE="${PIPER_OBJECT_MODELS:-$PIPER_ROOT/config/object_models.json}"
PLACE_XYZ="[0.35, -0.12, 0.02]"
SURFACE_OFFSET="0.005"  # 2026-09-14 사용자 요청: 추정 면 바깥 법선 방향 +5mm.
TARGET_COLOR="yellow" # 사용자가 지정한 노란 박스. 다른 색은 any로 변경한다.
batch_args=()
PICK_SELECT=best_fit
if [[ "${1:-}" = batch || "${1:-}" = feed || "${1:-}" = feed-auto || "${1:-}" = sweep ]]; then
  batch_mode="$1"
  workspace_mode="$batch_mode"
  if [ "$batch_mode" = feed-auto ]; then workspace_mode=feed_auto; fi
  if [ "$batch_mode" = feed-auto ]; then batch_mode=feed; fi
  shift
  TARGET_COLOR=any
  batch_config="${PIPER_BATCH_CONFIG:-$PIPER_ROOT/config/${workspace_mode}_workspace.json}"
  if [ ! -f "$batch_config" ]; then
    echo "연속 이송의 놓기 위치 설정이 필요합니다: $batch_config"
    echo '설정 예시와 실행 안내: docs/ROBOT.md'
    exit 2
  fi
  /usr/bin/python3 - "$batch_config" "$batch_mode" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    feed = json.load(stream).get('operator_feed', False)
if feed is not (sys.argv[2] == 'feed'):
    raise SystemExit('feed/batch 실행 모드와 operator_feed 설정이 다릅니다. 시작하지 않습니다.')
PY
  batch_args=(-p "batch_config_file:=$batch_config")
  if [ "$batch_mode" = sweep ]; then PICK_SELECT=highest; fi
fi
exec 9>/tmp/manipulator-bridge.lock
flock -n 9 || { echo '이미 run_bridge.sh가 실행 중입니다.'; exit 1; }
source_ros
source_workspace
/usr/bin/python3 "$PIPER_ROOT/scripts/fetch_weights.py" --verify --weights-dir "${PIPER_WEIGHTS_DIR:-$PIPER_ROOT/weights}"
export YOLO_AUTOINSTALL=False
cd "$PIPER_ROOT/perception"
exec "$PIPER_PERCEPTION_PYTHON" ros_pose_bridge.py --ros-args \
  -p object_models_file:="$MODELS_FILE" \
  -p dims:="$DIMS" -p place_xyz:="$PLACE_XYZ" \
  -p surface_offset_m:="$SURFACE_OFFSET" -p pick_select:="$PICK_SELECT" \
  -p target_color:="$TARGET_COLOR" \
  -p min_depth_m:=0.15 -p require_cuda:=true -p use_tf_up:=true \
  -p max_pose_age_s:=0.5 "${batch_args[@]}" "$@"
