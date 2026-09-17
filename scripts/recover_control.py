#!/usr/bin/env python3
"""Known PiPER setup: hold measured joints after teaching has ended; never start PnP."""
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import time

import rclpy
from action_msgs.msg import GoalStatusArray
from agx_arm_msgs.msg import AgxArmStatus
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, SetBool


class ControlRecovery(Node):
    def __init__(self):
        super().__init__('piper_manual_control_recovery')
        self.states = deque(maxlen=600)
        self.joints = deque(maxlen=600)
        self.actions = {}
        self.report = {}
        self.create_subscription(AgxArmStatus, '/feedback/arm_status',
                                 self.on_status, qos_profile_sensor_data)
        self.create_subscription(JointState, '/feedback/joint_states',
                                 self.on_joints, qos_profile_sensor_data)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action in ('/move_action', '/execute_trajectory',
                       '/arm_controller/follow_joint_trajectory',
                       '/gripper_controller/follow_joint_trajectory'):
            self.create_subscription(
                GoalStatusArray, action + '/_action/status',
                lambda msg, name=action: self.actions.update(
                    {name: [s.status for s in msg.status_list]}), qos)

    def on_status(self, msg):
        self.states.append((time.monotonic(), msg))

    def on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        names = [f'joint{i}' for i in range(1, 7)]
        if all(name in values for name in names):
            self.joints.append((time.monotonic(), [values[name] for name in names]))

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def call(self, kind, name, request):
        client = self.create_client(kind, name)
        try:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f'{name} 서비스가 없습니다. 터미널 1의 feed 실행을 확인하세요.')
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            if not future.done() or future.result() is None:
                raise RuntimeError(f'{name} 응답을 확인하지 못했습니다.')
            return future.result()
        finally:
            self.destroy_client(client)

    def snapshot(self):
        now = time.monotonic()
        if (not self.states or not self.joints or now - self.states[-1][0] >= 0.25
                or now - self.joints[-1][0] >= 0.25):
            raise RuntimeError('최신 팔 피드백이 없습니다. 복구를 중단합니다.')
        window = [q for t, q in self.joints if now - t < 0.5]
        if len(window) < 10 or not all(math.isfinite(v) for q in window for v in q):
            raise RuntimeError('관절 피드백이 부족하거나 유효하지 않습니다.')
        span = max(max(q[i] for q in window) - min(q[i] for q in window) for i in range(6))
        s = self.states[-1][1]
        return dict(ctrl_mode=s.ctrl_mode, teach_status=s.teach_status,
                    arm_status=s.arm_status, err_status=s.err_status,
                    joints_rad=self.joints[-1][1], joint_span_rad=span)

    def require_ended_and_idle(self, status):
        if tuple(status[k] for k in ('ctrl_mode', 'teach_status', 'arm_status', 'err_status')) != (2, 2, 0, 0):
            raise RuntimeError('교시 기록 종료·팔 무오류 상태가 아닙니다. 버튼과 팔 상태를 확인하세요.')
        if status['joint_span_rad'] >= 0.005:
            raise RuntimeError('팔이 움직이고 있습니다. 팔에서 손을 떼고 정지한 뒤 실행하세요.')
        if any(s in (1, 2, 3) for states in self.actions.values() for s in states):
            raise RuntimeError('실행 중인 동작이 있습니다. 시작 전 대기 상태에서만 복구하세요.')

    def recover(self):
        params = self.call(GetParameters, '/piper_pnp_controller/get_parameters',
                           GetParameters.Request(names=[
                               'auto_start', 'guard_real_commands', 'loop_mode',
                               'preview_only']))
        if (len(params.values) != 4 or any(v.type != 1 for v in params.values)
                or [v.bool_value for v in params.values] != [False, True, True, False]):
            raise RuntimeError('실기 반복 이송 시작 대기 설정이 아닙니다. 터미널 1의 feed/sweep 실행을 확인하세요.')
        self.spin(1.0)
        before = self.snapshot()
        self.report['before'] = before
        if before['ctrl_mode'] == 1 and before['arm_status'] == 0 and before['err_status'] == 0:
            return '이미 PC 제어 모드입니다. 복구 명령을 보내지 않았습니다.'
        self.require_ended_and_idle(before)
        gate = self.call(SetBool, '/control_enable', SetBool.Request(data=False))
        if not gate.success:
            raise RuntimeError('제어 게이트를 닫지 못했습니다: ' + gate.message)
        self.report['gate_closed'] = True
        self.spin(0.55)
        before = self.snapshot()
        self.require_ended_and_idle(before)
        # This workstation's S-V1.8-7 driver sends move_j(measured_joints).
        self.call(Empty, '/emergency_stop', Empty.Request())
        self.report['hold_requested'] = True
        self.spin(1.5)
        after = self.snapshot()
        change = max(abs(a - b) for a, b in zip(before['joints_rad'], after['joints_rad']))
        self.report.update(after=after, max_joint_change_rad=change)
        if after['ctrl_mode'] != 1 or after['arm_status'] != 0 or after['err_status'] != 0:
            raise RuntimeError('PC 제어 모드 복구를 확인하지 못했습니다. 실험을 시작하지 마세요.')
        if change >= 0.01 or after['joint_span_rad'] >= 0.005:
            raise RuntimeError('자세 유지 범위를 벗어났습니다. 실험 시작 전 팔 상태 확인이 필요합니다.')
        return 'PC 제어 모드 복구 완료. 현재 자세 유지, 실험 시작 대기 상태입니다.'


def main():
    rclpy.init()
    node = ControlRecovery()
    code = 0
    try:
        message = node.recover()
        node.report['result'] = message
        print(message)
        print('다음: bash ~/run_console.sh')
    except (RuntimeError, KeyboardInterrupt) as exc:
        code = 1
        node.report['error'] = str(exc) or '사용자가 복구를 중단했습니다.'
        print(node.report['error'])
    finally:
        stamp = datetime.now()
        path = (Path(__file__).resolve().parent / 'captures' /
                stamp.strftime('assessment_%Y%m%d') /
                stamp.strftime('control_recovery_%H%M%S_%f.json'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(node.report, ensure_ascii=False, indent=2) + '\n')
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
