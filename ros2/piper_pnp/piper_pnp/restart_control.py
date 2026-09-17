"""Operator-confirmed empty-workspace home/reset, executed by the cycle worker."""
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_srvs.srv import SetBool
from std_msgs.msg import String

from piper_pnp.moveit_client import make_start_state


class RestartControlMixin:
    def _init_restart_control(self):
        self._restart_lock = threading.RLock()
        self._restart_event = threading.Event()
        self._restart_waiting = False
        self._restart_pending = False
        self._restart_status = None
        self._restart_actions = {}
        self._restart_supported = self.guard_real_commands and not self.preview_only
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self._restart_state_pub = self.create_publisher(String, '~/experiment_state', qos)
        self.create_service(SetBool, '~/reset_experiment', self._on_reset_request,
                            callback_group=self._cb_group)
        if not self._restart_supported:
            return
        from agx_arm_msgs.msg import AgxArmStatus
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(AgxArmStatus, '/feedback/arm_status',
                                 self._on_restart_arm_status, qos_profile_sensor_data,
                                 callback_group=self._cb_group)
        actions = ['/move_action', '/execute_trajectory', self.arm_controller_action]
        if self.use_real_gripper:
            actions.append(self.gripper_action)
        for name in set(actions):
            state = {'statuses': {}}
            state['client'] = self.create_client(
                CancelGoal, name + '/_action/cancel_goal', callback_group=self._cb_group)
            self._restart_actions[name] = state
            self.create_subscription(
                GoalStatusArray, name + '/_action/status',
                lambda msg, state=state: self._on_restart_action_status(state, msg),
                qos, callback_group=self._cb_group)

    def _on_restart_arm_status(self, msg):
        self._restart_status = time.monotonic(), msg

    def _show_restart_state(self, message):
        self.get_logger().info(message)
        self._restart_state_pub.publish(String(data=message))

    @staticmethod
    def _on_restart_action_status(state, msg):
        state['statuses'] = {bytes(s.goal_info.goal_id.uuid): s.status for s in msg.status_list}

    def _cancel_restart_request(self):
        with self._restart_lock:
            self._restart_pending = False
            self._abort.set()
            self._restart_event.set()

    def _on_reset_request(self, request, response):
        with self._restart_lock:
            if not request.data:
                response.message = '빈 그리퍼·goal 전체 비움·홈 경로 확인이 필요합니다. h → y.'
            elif not self._restart_supported:
                response.message = '이 런치는 실기 초기화를 지원하지 않습니다.'
            elif self._terminate.is_set() or not self._restart_waiting:
                response.message = '동작이 아직 정리 중입니다. 정지 후 재시작 대기 안내가 나오면 h → y.'
            elif self._restart_pending:
                response.message = '홈 초기화 요청을 이미 받았습니다.'
            else:
                self._restart_pending = True
                self._restart_event.set()
                response.success = True
                response.message = '전체 비움 확인 — 홈 복귀 후 이송 횟수를 0으로 초기화합니다.'
        return response

    def run_restartable(self):
        """Keep services alive on failure; only a confirmed reset may clear abort."""
        if not self._restart_supported:
            return self.run()
        needs_reset = bool(getattr(self, '_batch_progress', None) and
                           self._batch_progress.in_progress)
        while rclpy.ok() and not self._terminate.is_set():
            if not needs_reset:
                try:
                    self.run()
                except Exception as exc:
                    self.get_logger().error(f'실험 실패: {exc}')
            if self._terminate.is_set():
                return False
            self.emergency_stop('실험 정지 — 콘솔에서 h → y로 처음부터 준비할 수 있습니다.',
                                terminate=False)
            self.print_report()
            self._waiting_confirmation = False
            with self._restart_lock:
                self._restart_waiting = True
                self._restart_pending = False
                self._restart_event.clear()
            self._show_restart_state(
                '재시작 대기: 그리퍼와 goal의 박스를 모두 치우고 홈 경로에서 손을 뺀 뒤 '
                '콘솔 h → y. 성공하면 이송 횟수 0, 다음 c부터 새 실험입니다.')
            needs_reset = True
            while rclpy.ok() and not self._terminate.is_set():
                self._restart_event.wait(.1)
                with self._restart_lock:
                    self._restart_event.clear()
                    if not self._restart_pending or self._terminate.is_set():
                        continue
                    self._restart_pending = False
                    self._restart_waiting = False
                    self._abort.clear()
                    self._stop_sent = False
                    self._continue_event.clear()
                    self._start_requested.clear()
                try:
                    ok = self._home_for_restart()
                except Exception as exc:
                    self.get_logger().error(f'홈 초기화 실패: {exc}')
                    ok = False
                if ok:
                    needs_reset = False
                    self._start_requested.set()
                    break
                self.emergency_stop('홈 초기화 실패 — 현재 자세 유지, 기록 초기화 안 함.',
                                    terminate=False)
                with self._restart_lock:
                    self._restart_waiting = True
                self._show_restart_state('재시작 대기: 원인을 확인한 뒤 h → y로 다시 요청하세요.')
        return False

    def _restart_arm_ready(self):
        value = self._restart_status
        if value is None or time.monotonic() - value[0] > self.feedback_max_age:
            self.get_logger().error('최신 팔 상태 피드백이 없습니다.')
            return False
        msg = value[1]
        if msg.ctrl_mode not in (0, 1) or msg.arm_status != 0 or msg.err_status != 0:
            self.get_logger().error('티칭 모드/팔 오류를 먼저 해제해야 홈 초기화할 수 있습니다.')
            return False
        return True

    def _cancel_previous_motion(self):
        """Cancel each motion server and wait for terminal status before reseeding."""
        required = []
        for name, state in self._restart_actions.items():
            client = state['client']
            if self.aborted or not client.wait_for_service(timeout_sec=2.):
                return False
            response = self._await_control_future(client.call_async(CancelGoal.Request()), 3.)
            if response is None or response.return_code != CancelGoal.Response.ERROR_NONE:
                self.get_logger().error(f'이전 동작 취소 확인 실패: {name}')
                return False
            required.append((state, {bytes(g.goal_id.uuid) for g in response.goals_canceling}))
        active = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                  GoalStatus.STATUS_CANCELING)
        terminal = (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                    GoalStatus.STATUS_ABORTED)
        deadline = time.monotonic() + 5.
        while not self.aborted and time.monotonic() < deadline:
            if (self.moveit.motion_requests_settled() and all(
                    not any(s in active for s in state['statuses'].values()) and
                    all(state['statuses'].get(g) in terminal for g in goals)
                    for state, goals in required)):
                return True
            time.sleep(.02)
        self.get_logger().error('이전 동작의 종료 응답을 확인하지 못했습니다. 홈으로 움직이지 않습니다.')
        return False

    def _home_for_restart(self):
        self._show_restart_state('홈 초기화 중 — 정지 키는 그대로 사용할 수 있습니다.')
        self._command_guard.close()
        if not self._restart_arm_ready() or not self._cancel_previous_motion():
            return False
        if self.aborted or not self.moveit.wait_for_servers(timeout=3.):
            return False
        if not self._setup_planning_scene():
            return False
        workspace = self._batch_workspace
        if workspace is not None:
            ids = [f'batch_placed_{workspace.digest[:12]}_{i}'
                   for i in range(len(workspace.places))]
            if workspace.drop:
                from piper_pnp.sweep_control import HELD_BOX_ID
                if not self.moveit.clear_attached_box(HELD_BOX_ID,self.ee_link):
                    return False
                ids.append(f'batch_placed_{workspace.digest[:12]}_source')
                if workspace.feed_drop:
                    ids.append(f'batch_placed_{workspace.digest[:12]}_0_landing')
            if self.aborted or not self.moveit.clear_collision_boxes(ids):
                self.get_logger().error('비운 goal의 충돌 모델 정리를 확인하지 못했습니다.')
                return False
        if self.aborted or not self._restart_arm_ready():
            return False
        guard = self._command_guard
        with guard.lock:
            if guard.feedback_at is None or time.monotonic()-guard.feedback_at > guard.max_age:
                return False
            state = make_start_state(guard.feedback.keys(), guard.feedback.values())
        if not self.moveit.check_state_validity(state):
            return False
        self._reset_cycle_state()
        if self.aborted or not self._prepare_real_control():
            return False
        # No gripper opening or hardware move_home command: plan from actual pose,
        # retain gripper width, and use the same measured-arrival check as a cycle.
        if not self._goto_joints('초기화: Zero Pose', self.zero_pose):
            return False
        guard.close()
        if not self._set_driver_gate(self._control_enable_client, False):
            return False
        with self._restart_lock:
            if self.aborted or self._terminate.is_set():
                return False
            if workspace is not None:
                self._batch_progress.reset_after_home()
                self._batch_completed = 0
            self._reset_cycle_state()
        self._show_restart_state('홈 초기화 완료 — 이송 횟수 0. 새 박스를 놓고 c로 새 실험을 시작하세요.')
        return True
