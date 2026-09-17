"""놓기 전, 그리퍼 개방 중간 폭과 개방 후 직선 복귀까지 계획만으로 검사."""
import math
import time

from piper_pnp.moveit_client import make_start_state, trajectory_end_state


def with_joint(state, name, value):
    values = dict(zip(state.joint_state.name, state.joint_state.position))
    values[name] = float(value)
    return make_start_state(values.keys(), values.values())


class ReleaseChecksMixin:
    def _release_current_state(self):
        if self.preview_only:
            return self._preview_state
        guard = self._command_guard
        with guard.lock:
            if (guard.feedback_at is None
                    or not 0 <= time.monotonic()-guard.feedback_at <= guard.max_age
                    or guard.feedback is None
                    or self.gripper_joint not in guard.feedback):
                self.get_logger().error('새 관절/그리퍼 피드백이 없어 놓기 검사를 중단합니다.')
                return None
            return make_start_state(guard.feedback.keys(), guard.feedback.values())

    def _gripper_sweep_clear(self, state):
        values = dict(zip(state.joint_state.name, state.joint_state.position))
        current = values.get(self.gripper_joint)
        target = self.gripper_open_position
        if current is None or not all(math.isfinite(v) and 0 <= v <= .1
                                      for v in (current, target)):
            return False
        # 끝 상태를 먼저 검사해 명백히 불가능한 개방을 빠르게 거부한다.
        if not self.moveit.check_state_validity(with_joint(state, self.gripper_joint, target)):
            return False
        # 총 개방 폭 2 mm 간격 = 각 손가락 최대 1 mm 간격의 이산 검사.
        steps = max(1, math.ceil(abs(target-current)/.002))
        for i in range(steps):
            if self.aborted or not self.moveit.check_state_validity(with_joint(
                    state, self.gripper_joint, current+(target-current)*i/steps)):
                return False
        return True

    def _plan_release_clearance(self, state, retreat_pose, contact_pose=None, check_camera=False):
        if state is None or self.aborted:
            return False
        if contact_pose is not None:
            ok, descent = self.moveit.move_cartesian(
                [contact_pose], start_state=state, plan_only=True,
                max_step=self.cartesian_max_step, min_fraction=1.0)
            if not ok or descent is None:
                return False
            state = trajectory_end_state(descent, state)
            if state is None:
                return False
        if not self._gripper_sweep_clear(state):
            return False
        opened = with_joint(state, self.gripper_joint, self.gripper_open_position)
        ok, retreat = self.moveit.move_cartesian(
            [retreat_pose], start_state=opened, plan_only=True,
            max_step=self.cartesian_max_step, min_fraction=1.0)
        if not ok or self.aborted:
            return False
        if check_camera:
            if retreat is None:
                return False
            raised = trajectory_end_state(retreat, opened)
            if raised is None:
                return False
            ok, _ = self.moveit.move_to_joints(
                self.arm_joints, self.camera_ready_pose, start_state=raised, plan_only=True)
        return ok and not self.aborted

    def _check_place_release(self, retreat_pose, contact_pose=None):
        if not self.guard_real_commands or self.preview_only or not self.use_real_gripper:
            return True
        if self._plan_release_clearance(
                self._release_current_state(), retreat_pose, contact_pose):
            self.get_logger().info('놓기 사전 검사 통과: 개방 중간 폭·완전 개방·직선 복귀.')
            return True
        self._report.append(('놓기 전 개방·복귀 경로 검사', False))
        self.emergency_stop('놓기 전 개방/복귀 경로가 막혀 있습니다. 그리퍼를 열지 않습니다.')
        return False

    def _place_open_ik_clear(self, pre, contact):
        if not self.guard_real_commands or self.preview_only or not self.use_real_gripper:
            return True
        state = self._release_current_state()
        if state is None:
            return False
        opened = with_joint(state, self.gripper_joint, self.gripper_open_position)
        return all(self.moveit.compute_ik(
            pose, start_state=opened, avoid_collisions=True,
            ik_timeout=self.grasp_ik_check_timeout) for pose in (pre, contact))
