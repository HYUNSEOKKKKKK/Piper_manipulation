"""Cuboid 실기 런치용 명령 중계: 실제 자세로 mock 시드 후 명시적으로 허용."""
import math
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_pnp.control_guard import CommandGuard


class RealControlMixin:
    def _init_real_control(self):
        names = list(self.arm_joints)
        if self.use_real_gripper:
            names.append(self.gripper_joint)
        self._command_guard = CommandGuard(names, self.feedback_max_age,
                                            self.feedback_joint_tolerance)
        if not self.guard_real_commands or self.preview_only:
            return
        self._hardware_pub = self.create_publisher(JointState, '/control/joint_states', 1)
        self.create_subscription(JointState, '/feedback/joint_states', self._on_real_feedback,
                                 qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(JointState, '/pnp/control/joint_states', self._on_mock_command,
                                 1, callback_group=self._cb_group)
        self._arm_seed_client = ActionClient(self, FollowJointTrajectory,
                                             self.arm_controller_action,
                                             callback_group=self._cb_group)
        self._arm_enable_client = self.create_client(SetBool, '/enable_agx_arm',
                                                     callback_group=self._cb_group)
        self.create_timer(0.05, self._command_watchdog, callback_group=self._cb_group)

    def _on_real_feedback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        age = self.get_clock().now().nanoseconds / 1e9 - stamp
        if not 0 <= age <= self.feedback_max_age:
            return
        self._command_guard.record('feedback', msg.name, msg.position, time.monotonic())

    def _on_mock_command(self, msg):
        guard = self._command_guard
        reason = None
        with guard.lock:
            valid = guard.record('command', msg.name, msg.position, time.monotonic())
            if not guard.enabled:
                return
            reason = guard.reason(time.monotonic()) if valid else 'malformed command'
            if reason or self._abort.is_set():
                guard.close()
            else:
                out = JointState()
                out.header = msg.header
                # 더미 그리퍼는 하위 토픽에서도 제거한다.
                out.name = list(guard.joints)
                out.position = [guard.command[name] for name in out.name]
                self._hardware_pub.publish(out)
        if reason:
            self.emergency_stop(f'명령 중계 중단: {reason}')

    def _command_watchdog(self):
        guard = self._command_guard
        with guard.lock:
            reason = guard.reason(time.monotonic()) if guard.enabled else None
            if reason:
                guard.close()
        if reason:
            self.emergency_stop(f'실기 피드백/명령 시간 초과: {reason}')

    def _await_control_future(self, future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline and not self.aborted:
            time.sleep(0.02)
        if not future.done() or self.aborted:
            return None
        return future.result()

    def _seed_mock_controller(self, client, names, positions):
        if not client.wait_for_server(timeout_sec=5.0):
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=1)
        goal.trajectory.points = [point]
        result = self.moveit._send_goal(
            client, goal, 5.0, 'mock seed', return_wrapper=True)
        return (not self.aborted and result is not None and
                result.status == GoalStatus.STATUS_SUCCEEDED and
                result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)

    def _set_driver_gate(self, client, value):
        if not client.wait_for_service(timeout_sec=3.0):
            return False
        request = SetBool.Request()
        request.data = value
        response = self._await_control_future(client.call_async(request), 10.0)
        return response is not None and response.success

    def _prepare_real_control(self):
        if not self.guard_real_commands or self.preview_only:
            return True
        guard = self._command_guard
        guard.close()
        # 외부 중복 publisher가 있으면 로컬 게이트만으로 명령을 차단할 수 없다.
        if (self.count_publishers('/control/joint_states') != 1 or
                self.count_publishers('/feedback/joint_states') != 1 or
                self.count_publishers('/pnp/control/joint_states') != 1):
            self.get_logger().error('실기/피드백/명령 토픽은 각각 publisher 1개여야 합니다.')
            return False
        if not self._set_driver_gate(self._control_enable_client, False):
            return False
        with guard.lock:
            if (guard.feedback_at is None or
                    time.monotonic()-guard.feedback_at > guard.max_age):
                self.get_logger().error('실제 관절 피드백이 없어 제어기를 동기화할 수 없습니다.')
                return False
            current = dict(guard.feedback)
        self.get_logger().info('명령 차단 상태에서 mock 제어기를 실제 관절값으로 동기화합니다.')
        if not self._seed_mock_controller(self._arm_seed_client, self.arm_joints,
                                          [current[j] for j in self.arm_joints]):
            return False
        if self.use_real_gripper and not self._seed_mock_controller(
                self._gripper_client, [self.gripper_joint], [current[self.gripper_joint]]):
            return False
        deadline = time.monotonic()+2.0
        while guard.reason(time.monotonic(), require_alignment=True) and time.monotonic() < deadline:
            if self.aborted:
                return False
            time.sleep(0.02)
        if guard.reason(time.monotonic(), require_alignment=True):
            self.get_logger().error('mock 명령과 실제 관절값이 일치하지 않습니다.')
            return False
        if not self._set_driver_gate(self._arm_enable_client, True):
            return False
        if not self._set_driver_gate(self._control_enable_client, True):
            return False
        with guard.lock:
            if self.aborted:
                return False
            try:
                guard.open(time.monotonic())
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return False
        self.get_logger().info('실기 명령 허용 — 실제 관절 피드백을 감시합니다.')
        return True

    def _verify_real_arrival(self, trajectory, joint_target=None):
        if not self.guard_real_commands or self.preview_only:
            return True
        if trajectory is None:
            return False
        jt = trajectory.joint_trajectory
        if jt.points:
            target = dict(zip(jt.joint_names, jt.points[-1].positions))
        elif joint_target is not None:
            # MoveIt은 관절 목표가 이미 만족되면 SUCCESS + 빈 궤적을 반환한다.
            # 성공 응답만으로 통과하지 않고 요청한 관절 목표를 실측과 비교한다.
            target = joint_target
            self.get_logger().info('이동 경로 없음 — 실제 관절이 요청한 목표에 도달했는지 확인합니다.')
        else:
            self.get_logger().error('빈 이동 경로에 비교할 관절 목표가 없어 도달을 확인할 수 없습니다.')
            return False
        if not all(j in target and math.isfinite(target[j]) for j in self.arm_joints):
            return False
        guard = self._command_guard
        deadline, stable_since = time.monotonic()+self.feedback_arrival_timeout, None
        while time.monotonic() < deadline and not self.aborted:
            now = time.monotonic()
            with guard.lock:
                if guard.reason(now):
                    return False
                error = max(abs(target[j]-guard.feedback[j]) for j in self.arm_joints)
            if error <= self.feedback_joint_tolerance:
                stable_since = now if stable_since is None else stable_since
                if now-stable_since >= 0.15:
                    return True
            else:
                stable_since = None
            time.sleep(0.02)
        self.get_logger().error('실제 관절이 계획 끝점에 도달하지 못했습니다.')
        return False
