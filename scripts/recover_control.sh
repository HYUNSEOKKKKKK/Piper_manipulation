#!/usr/bin/env bash
# feed 시작 전, 교시 기록 종료 후 PC 제어 모드 복구.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec 9>/tmp/manipulator-control-recovery.lock
flock -n 9 || { echo '이미 제어 모드 복구가 실행 중입니다.' >&2; exit 1; }
can_status=$(ip -details link show "$PIPER_CAN") || exit 1
if [[ "$can_status" != *"state UP"* || "$can_status" != *"can state ERROR-ACTIVE"* ]]; then
  echo 'CAN 통신을 먼저 복구하세요. 제어 모드를 변경하지 않습니다.' >&2
  echo "$can_status" >&2
  exit 1
fi
source_ros
source_workspace
echo 'feed 시작 전 복구: 팔에서 손을 뗀 상태로 사용하세요.'
exec /usr/bin/python3 "$PIPER_ROOT/scripts/recover_control.py"
