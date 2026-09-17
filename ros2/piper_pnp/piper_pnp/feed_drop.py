"""Legacy feed pick, followed by one position-only move and an airborne release."""
import math
from itertools import product

from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from piper_pnp.geometry import quat_inverse, quat_multiply, quat_rotate_vector
from piper_pnp.moveit_client import trajectory_end_state
from piper_pnp.object_manipulation import held_transform, span
from piper_pnp.release_checks import with_joint
from piper_pnp.sweep_control import HELD_BOX_ID, pose_at


class FeedDropMixin:
    def _prepare_feed_drop(self):
        """Refresh the shared goal obstacle before approaching the next box."""
        workspace = self._batch_workspace
        pick = self._targets['pick']
        top = pick['sweep_scene']['drop_top']
        # Coarse pre-pick bound. Actual held-box bottom clearance is checked at
        # each planned release pose; a diagonal bound blocks even an empty goal
        # at the requested 100 mm TCP height.
        if top + workspace.drop_clearance > workspace.place(0)[2]:
            self.get_logger().error(
                f'goal 높이 {top*1000:.0f}mm: 고정 낙하 높이의 여유가 부족합니다. '
                '새 박스를 집지 않습니다.')
            return False
        return (self.moveit.clear_collision_boxes([self._sweep_obstacle_id(False) + '_landing'])
                and self._sweep_obstacle(False, top))

    def _drop_box_extent(self, pose):
        """World bottom and vertical span for the actual box/tool transform."""
        pick = self._targets['pick']
        offset, relative = held_transform(
            pick, self._pick_grasp_q, self.grasp_offset, self._pick_contact_position)
        q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        height = span(pick['dims'], quat_multiply(q, relative), (0., 0., 1.))
        bottom = pose.position.z + quat_rotate_vector(q, offset)[2] - height/2
        return bottom, height

    def _drop_landing_geometry(self, pose):
        """Gravity projection of the held box onto the observed goal height."""
        pick = self._targets['pick']
        offset, relative = held_transform(
            pick, self._pick_grasp_q, self.grasp_offset, self._pick_contact_position)
        q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        box_q = quat_multiply(q, relative)
        delta = quat_rotate_vector(q, offset)
        height = span(pick['dims'], box_q, (0., 0., 1.))
        center = (pose.position.x + delta[0], pose.position.y + delta[1],
                  pick['sweep_scene']['drop_top'] + height/2)
        return center, box_q

    def _drop_landing_fits(self, pose, landing):
        """Measured box corners must fit inside the checked landing envelope."""
        center, q = self._drop_landing_geometry(pose)
        planned = landing.primitive_poses[0]
        origin = (planned.position.x, planned.position.y, planned.position.z)
        inverse = quat_inverse((planned.orientation.x, planned.orientation.y,
                                planned.orientation.z, planned.orientation.w))
        bounds = landing.primitives[0].dimensions
        for signs in product((-1., 1.), repeat=3):
            corner = quat_rotate_vector(q, tuple(s*d/2 for s,d in zip(signs, self._targets['pick']['dims'])))
            local = quat_rotate_vector(inverse, tuple(c+v-o for c,v,o in zip(center, corner, origin)))
            if not all(abs(v) <= d/2 + 1e-9 for v,d in zip(local, bounds)):
                return False
        return True

    def _drop_return_scene(self, pose):
        """Hypothetical post-release scene, only for the return planning request.

        The real planning scene retains the held box until measured opening.
        Keep the observed pile and add one local landing envelope. Inflating
        the entire 260 mm goal to the next box's height invents obstacles in
        empty space, particularly around the gripper base at a 100 mm release.
        """
        center, q = self._drop_landing_geometry(pose)
        obj = CollisionObject(id=self._sweep_obstacle_id(False) + '_landing', operation=CollisionObject.ADD)
        obj.header.frame_id = self.base_frame
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                        dimensions=[d+.010 for d in self._targets['pick']['dims']])]
        # Preserve the box orientation. A world-axis bounding box invents
        # corners between the fingers when the cuboid is tilted/yawed.
        # This is a predicted landing, refreshed by depth before the next pick.
        obj.primitive_poses = [pose_at(center, q)]
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [AttachedCollisionObject(
            link_name=self.ee_link, object=CollisionObject(id=HELD_BOX_ID, operation=CollisionObject.REMOVE))]
        # Detachment would otherwise leave a duplicate world object at the
        # carry pose in this hypothetical scene.
        scene.world.collision_objects = [CollisionObject(id=HELD_BOX_ID, operation=CollisionObject.REMOVE), obj]
        return scene

    def _plan_feed_drop(self, state):
        w = self._batch_workspace
        top = self._targets['pick']['sweep_scene']['drop_top']
        rejected = dict(height=0, opening=0, return_path=0)
        for attempt in range(1, 4):
            if self.aborted:
                return None
            ok, path = self.moveit.move_to_pose(
                pose_at(w.place(self._batch_completed), (0., 0., 0., 1.)),
                start_state=state, plan_only=True, position_only=True)
            if not ok or path is None or self.aborted:
                if not self.aborted:
                    self.get_logger().error('중앙 낙하 위치로 가는 이동 경로를 계산하지 못했습니다.')
                return None
            end = trajectory_end_state(path, state)
            if end is None:
                return None
            pose = self.moveit.compute_fk(end.joint_state.name, end.joint_state.position)
            if pose is None:
                return None
            bottom, _ = self._drop_box_extent(pose)
            gap = bottom - top
            if not bottom >= top + w.drop_clearance:
                rejected['height'] += 1
                self.get_logger().warn(
                    f'낙하 후보 {attempt}/3: 이동 경로 계산 성공, 높이 여유 검사 탈락 — '
                    f'TCP z={pose.position.z*1000:.1f}mm, 박스 아랫면={bottom*1000:.1f}mm, '
                    f'goal 관측 상한={top*1000:.1f}mm (깊이 여유 15mm 포함), '
                    f'간격={gap*1000:.1f}mm < 필요 {w.drop_clearance*1000:.1f}mm. '
                    '개방/복귀 계획 전 기각.')
                continue
            if not self._gripper_sweep_clear(end):
                rejected['opening'] += 1
                self.get_logger().warn(
                    f'낙하 후보 {attempt}/3: 이동 경로·높이 검사 통과, '
                    '그리퍼 개방 상태/충돌 검사 탈락. 복귀 계획 전 기각.')
                continue
            opened = with_joint(end, self.gripper_joint, self.gripper_open_position)
            scene = self._drop_return_scene(pose)
            ok, back = self.moveit.move_to_joints(
                self.arm_joints, self.camera_ready_pose, start_state=opened, plan_only=True,
                scene_diff=scene)
            if ok and back is not None and not self.aborted:
                return path, back, scene.world.collision_objects[-1]
            if self.aborted:
                return None
            rejected['return_path'] += 1
            self.get_logger().warn(
                f'낙하 후보 {attempt}/3: 이동 경로·높이·개방 검사 통과, 촬영 복귀 경로 계산 실패.')
        self.get_logger().error(
            '낙하 후보 3개가 사전 검사에서 탈락했습니다. '
            f'높이 여유={rejected["height"]}, 개방 검사={rejected["opening"]}, '
            f'복귀 계획={rejected["return_path"]}. goal 이동/개방은 실행하지 않았습니다.')
        return None

    def _run_feed_drop(self):
        """No place orientation constraint, Cartesian descent, or place retreat.

        Validate opening/return, execute that same accepted plan, verify XYZ,
        open at the achieved orientation, then return directly to Camera-Ready.
        """
        if not self._begin_step(6, '중앙 낙하 위치로 이동 (XYZ만)'):
            return False
        workspace = self._batch_workspace
        xyz = workspace.place(self._batch_completed)
        pick = self._targets['pick']
        # Outbound checks the held box against the observed pile. Return uses
        # a separate post-release scene, without a duplicate still-held box.
        if not self._sweep_obstacle(False, pick['sweep_scene']['drop_top']):
            return False
        state = self._release_current_state()
        if state is None or self.aborted:
            return False
        planned = self._plan_feed_drop(state)
        if planned is None or self.aborted:
            self._report.append(('중앙 낙하 사전 검사', False))
            return False
        path, back, landing = planned
        ok = self.moveit._execute_trajectory(path, 60.)
        self._last_pose_trajectory = path if ok else None
        if not self._after_motion('중앙 낙하 위치 이동', ok, path) or self.aborted:
            return False
        actual = self._achieved_ee_pose()
        if actual is None or not math.dist(
                (actual.position.x, actual.position.y, actual.position.z), xyz) <= .012:
            self.get_logger().error('실측 XYZ가 낙하 목표에 도달하지 않았습니다. 그리퍼를 열지 않습니다.')
            return False
        # Deliberately no orientation/alignment test: release as arrived.
        bottom, _ = self._drop_box_extent(actual)
        top = pick['sweep_scene']['drop_top']
        if not (bottom >= top + workspace.drop_clearance and
                self._drop_landing_fits(actual, landing)):
            self.get_logger().error('실측 박스 자세의 낙하/복귀 여유가 부족합니다. 개방하지 않습니다.')
            return False
        current = self._release_current_state()
        if current is None or not self._gripper_sweep_clear(current):
            return False
        if not self._begin_step(7, '공중에서 그리퍼 개방 (하강 없음)'):
            return False
        if not self.open_gripper() or self.aborted or not self._sweep_verify_open():
            return False
        if not self.moveit.clear_attached_box(HELD_BOX_ID, self.ee_link):
            return False
        if not self.moveit.add_collision_box(
                landing.id, self.base_frame, landing.primitive_poses[0], landing.primitives[0].dimensions):
            return False
        if not self._begin_step(8, '촬영 자세로 직접 복귀 (놓기 상승 없음)'):
            return False
        ok = self.moveit._execute_trajectory(back, 60.)
        return self._after_motion(
            '낙하 후 Camera-Ready Pose', ok, back,
            joint_target=dict(zip(self.arm_joints, self.camera_ready_pose)))
