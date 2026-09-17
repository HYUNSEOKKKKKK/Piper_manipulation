#!/usr/bin/env bash
# 전용 터미널에서 실행. 시작 요청 뒤 Step 1 승인 대기 중 정지 콘솔을 연다.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [ ! -t 0 ]; then
  echo '키보드를 읽을 수 있는 별도 터미널에서 실행하세요.' >&2
  exit 1
fi
can_status=$(ip -details link show "$PIPER_CAN") || exit 1
if [[ "$can_status" != *"state UP"* || "$can_status" != *"can state ERROR-ACTIVE"* ]]; then
  echo 'CAN 상태를 먼저 확인해야 합니다. 실험 시작을 요청하지 않습니다.' >&2
  echo "$can_status" >&2
  exit 1
fi
source_ros
source_workspace
# 단일 단계 승인 또는 최초 c 승인이 필수인 batch 실기에만 시작을 요청한다.
/usr/bin/python3 - <<'PY'
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import GetParameters
from agx_arm_msgs.msg import AgxArmStatus
rclpy.init()
node = Node('check_step_confirmation')
try:
    arm_status = []
    subscription = node.create_subscription(
        AgxArmStatus, '/feedback/arm_status', arm_status.append,
        qos_profile_sensor_data)
    deadline = time.monotonic() + 5.0
    while not arm_status and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not arm_status:
        raise SystemExit('팔 상태 피드백이 없습니다. 시작하지 않습니다.')
    status = arm_status[-1]
    if status.ctrl_mode == 2:
        if status.teach_status == 2:
            raise SystemExit(
                '티칭 기록은 종료됐지만 제어 모드가 티칭으로 남아 있습니다. '
                '현재 자세 유지 방식으로 PC 제어 모드 복구가 필요합니다. 실험은 시작하지 않습니다.')
        raise SystemExit(
            '팔이 티칭 모드입니다. 티칭 버튼 표시등과 모드를 확인하세요. '
            '자동으로 모드를 바꾸거나 실험을 시작하지 않습니다.')
    if status.ctrl_mode not in (0, 1) or status.arm_status != 0 or status.err_status != 0:
        raise SystemExit(
            f'팔 상태 확인이 필요합니다: ctrl_mode={status.ctrl_mode}, '
            f'arm_status={status.arm_status}, err_status={status.err_status}. 시작하지 않습니다.')
    client = node.create_client(GetParameters, '/piper_pnp_controller/get_parameters')
    if not client.wait_for_service(timeout_sec=10):
        raise SystemExit('제어기가 없습니다. 터미널 1에서 run_robot.sh feed-auto / feed / sweep / grasp를 실행하세요.')
    names = ['step_confirm', 'preview_only', 'auto_start', 'guard_real_commands',
             'batch_config_file', 'loop_mode', 'use_real_gripper', 'batch_operator_feed']
    future = client.call_async(GetParameters.Request(names=names))
    rclpy.spin_until_future_complete(node, future, timeout_sec=5)
    if not future.done() or future.result() is None:
        raise SystemExit('제어기 설정을 확인하지 못했습니다. 시작하지 않습니다.')
    values = future.result().values
    if (len(values) != 8 or any(values[i].type != 1 for i in (0,1,2,3,5,6,7))
            or values[4].type != 4
            or [v.bool_value for v in values[1:4]] != [False, False, True]):
        raise SystemExit('실기 시작 승인과 명령 보호 설정을 확인하지 못했습니다. 시작하지 않습니다.')
    batch = bool(values[4].string_value)
    if batch:
        if values[0].bool_value or not values[5].bool_value or not values[6].bool_value:
            raise SystemExit('batch 연속 이송 설정이 일치하지 않습니다. 시작하지 않습니다.')
        if values[7].bool_value:
            print('feed: 새 박스를 놓고 손을 뺀 뒤 c 한 번 → 9단계 자동 진행 → 홈에서 다음 투입 대기.')
        else:
            print('연속 이송: c 한 번으로 자동 반복. 박스 관측 대기 중에도 다시 보이면 자동 진행합니다. 박스를 추가하려면 먼저 정지하세요.')
    elif not values[0].bool_value:
        raise SystemExit('단일 실험은 step_confirm=true가 필요합니다. 시작하지 않습니다.')
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
echo '시작을 요청합니다. 아래 콘솔에서 c를 누르기 전까지 움직이지 않습니다.'
ros2 service call /piper_pnp_controller/start std_srvs/srv/Trigger '{}'
echo '정지/단계 승인 콘솔을 엽니다. 이 창을 클릭한 상태로 사용하세요.'
exec ros2 run piper_pnp piper_pnp_estop
