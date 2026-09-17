"""Continuous pick / elevated drop with measured workspace height.

No PD changes or open-loop trajectories. All movements use MoveIt and measured
arrival. Fresh depth and pose observations are frozen together before a pick.
"""
from collections import deque
from contextlib import contextmanager
import json
import math
import time

from geometry_msgs.msg import Pose
from std_msgs.msg import String
from piper_pnp.geometry import grasp_candidates, quat_multiply, quat_rotate_vector
from piper_pnp.object_manipulation import held_transform, span
from piper_pnp.moveit_client import trajectory_end_state
from piper_pnp.release_checks import with_joint

HELD_BOX_ID = 'sweep_held_box'


def validate_scene(scene, workspace):
    if (not isinstance(scene, dict) or scene.get('workspace_digest') != workspace.digest
            or scene.get('frame_id') != 'base_link'):
        raise ValueError('goal/source depth observation is missing or mismatched')
    heights = []
    names = ('drop',) if getattr(workspace, 'feed_drop', False) else ('drop', 'source')
    for name in names:
        r = scene.get(name, {})
        coverage, height = r.get('coverage'), r.get('height_m')
        if (type(coverage) not in (float, int) or not .75 <= coverage <= 1.
                or type(height) not in (float, int) or not math.isfinite(height)
                or not workspace.table_z-.025 <= height <= workspace.table_z+workspace.drop_max_height):
            raise ValueError(f'{name}: insufficient depth coverage or pile exceeds height limit')
        heights.append(max(workspace.table_z, height))
    return (heights[0], workspace.table_z) if len(heights) == 1 else tuple(heights)


def pose_at(position, q):
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
    return p


def drop_geometry(pick, pick_q, tool_q, workspace, drop_top, source_top,
                  grasp_offset, contact=None):
    offset, relative = held_transform(pick, pick_q, grasp_offset, contact)
    box_q = quat_multiply(tool_q, relative)
    offset_world = quat_rotate_vector(tool_q, offset)
    diagonal = math.sqrt(sum(d*d for d in pick['dims']))
    radius = math.sqrt(sum(x*x for x in offset)) + diagonal/2
    # Choose release height above observed clutter and the possible new pile.
    # Intermediate transfer clearance is checked with the attached collision box.
    floor = max(workspace.table_z+.22, source_top+radius+.035,
                drop_top+radius+.035, drop_top+diagonal+.04)
    x,y,_ = workspace.places[0]
    z = max(floor+.01, drop_top+workspace.drop_clearance+
            span(pick['dims'], box_q, (0.,0.,1.))/2-offset_world[2])
    release = pose_at((x-offset_world[0],y-offset_world[1],z), tool_q)
    for i, direction in enumerate(((1.,0.,0.), (0.,1.,0.))):
        if span(pick['dims'],box_q,direction)+.04 > workspace.drop_size[i]:
            raise ValueError('Held box does not fit inside observed drop zone')
    return release, floor


class SweepControlMixin:
    def _sweep_active(self):
        return bool(getattr(getattr(self, '_batch_workspace', None), 'sweep', False))

    def _accept_sweep_scene(self, metadata):
        """Called inside target lock, once per distinct acquisition timestamp."""
        samples = getattr(self, '_sweep_scene_samples', None)
        if samples is None:
            samples = self._sweep_scene_samples = deque(maxlen=self.marker_stable_samples)
        try:
            heights = validate_scene((metadata or {}).get('workspace_observation'), self._batch_workspace)
        except ValueError as exc:
            samples.clear()
            self.get_logger().warn(f'sweep 깊이 관측 대기: {exc}', throttle_duration_sec=3.)
            return None
        samples.append(heights)
        if len(samples) < self.marker_stable_samples:
            return False
        # Depth extrema fluctuate at occlusion boundaries. Use their maximum;
        # requiring stable extrema also repeatedly erased otherwise stable poses.
        return dict(drop_top=max(s[0] for s in samples)+.015,
                    source_top=max(s[1] for s in samples)+.015)

    @contextmanager
    def _sweep_speed(self, contact=False):
        before = self.moveit.velocity_scaling, self.moveit.acceleration_scaling
        self.moveit.velocity_scaling = self.velocity_scaling
        self.moveit.acceleration_scaling = self.acceleration_scaling
        try:
            yield
        finally:
            self.moveit.velocity_scaling, self.moveit.acceleration_scaling = before

    def _sweep_timed(self, label, function, *args, **kwargs):
        start = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.monotonic()-start
            self._sweep_times[label] = self._sweep_times.get(label, 0.) + elapsed
            self.get_logger().info(f'sweep 시간: {label}={elapsed:.2f}s')

    def _sweep_obstacle(self, source, top):
        w = self._batch_workspace
        if source:
            x0,x1,y0,y1 = w.pick_bounds
            x,y,dx,dy = (x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0
        else:
            x,y,_ = w.places[0]
            dx,dy = w.drop_size
        bottom = w.table_z-.005
        pose = pose_at((x,y,(top+bottom)/2), (0.,0.,0.,1.))
        return self.moveit.add_collision_box(
            self._sweep_obstacle_id(source), self.base_frame, pose,
            (dx,dy,max(.005,top-bottom)))

    def _sweep_obstacle_id(self, source=False):
        return f'batch_placed_{self._batch_workspace.digest[:12]}_' + ('source' if source else '0')

    def _choose_sweep_drop(self, pick_q, contact=None):
        pick = self._targets['pick']
        scene = pick['sweep_scene']
        # Drop orientation is independent of the grasp. Reuse a successful
        # orientation first, then try downward orientations including modest tilt.
        candidates = grasp_candidates((0.,0.,0.,1.), [0.,math.pi,math.pi/2,-math.pi/2],
                                      [0.,math.pi/12,math.pi/6,math.pi/4], [0.,math.pi,math.pi/2,-math.pi/2])
        cached = getattr(self, '_sweep_drop_q', None)
        if cached is not None:
            candidates.insert(0,cached)
        candidates.append(pick_q)
        state = self._release_current_state()
        if state is None:
            return None
        for q in candidates:
            if quat_rotate_vector(q,(0.,0.,1.))[2] > -.7:
                continue
            pose,floor = drop_geometry(pick,pick_q,q,self._batch_workspace,
                                      scene['drop_top'],scene['source_top'],self.grasp_offset,contact)
            if pose.position.z+.04 > .65:
                continue
            if self.moveit.compute_ik(pose, start_state=state, ik_timeout=self.grasp_ik_check_timeout):
                return pose,floor
        return None

    def _plan_sweep_pick(self, q, pre, grasp):
        """Check approach and short descent, reusing the accepted approach plan.

        IK at two endpoints alone is insufficient near this arm's limits.
        Reuse the accepted approach plan rather than ask OMPL for a new branch.
        """
        state = self._release_current_state()
        if state is None or self.aborted:
            return False
        ok,path = self.moveit.move_to_pose(pre,start_state=state,plan_only=True)
        if not ok or path is None or self.aborted:
            return False
        ready = trajectory_end_state(path,state)
        if ready is None:
            return False
        fk = self.moveit.compute_fk(ready.joint_state.name,ready.joint_state.position)
        if fk is None:
            return False
        _,contact = self._align_grasp_to_achieved(pre,grasp,anchor_contact=True,achieved_pose=fk)
        if contact is None:
            return False
        ok,_ = self.moveit.move_cartesian([contact],start_state=ready,plan_only=True,
                                        max_step=self.cartesian_max_step,min_fraction=1.)
        if not ok or self.aborted:
            return False
        self._sweep_pick_path = path
        return True

    def _sweep_attach_pick(self):
        pick = self._targets['pick']
        offset,q = held_transform(pick,self._pick_grasp_q,self.grasp_offset,
                                  self._pick_contact_position)
        return self.moveit.set_attached_box(
            HELD_BOX_ID,self.ee_link,pose_at(offset,q),
            tuple(d+.004 for d in pick['dims']),
            [self.ee_link,'gripper_base','gripper_link1','gripper_link2','flange_link'])

    def _run_sweep(self):
        if not self._check_batch_bridge():
            return False
        self.get_logger().info(
            'sweep: c 한 번으로 최대 10개 연속 이송. 같은 goal 위에서 개방합니다. '
            '박스가 없거나 goal 깊이가 안 보이면 촬영 자세에서 대기. Space=정지.')
        if not self._wait_for_confirmation('연속 이송 전체 시작 / 손 뺌 확인', timeout=None):
            return False
        if not self._prepare_real_control() or not self.open_gripper():
            return False
        if not self._goto_joints('sweep 촬영 자세', self.camera_ready_pose):
            return False
        self._sweep_timing_pub = self.create_publisher(String, '~/cycle_timing', 1)
        while not self.aborted and not self._terminate.is_set():
            if self._batch_completed >= len(self._batch_workspace.places):
                self.get_logger().info('sweep 최대 이송 횟수 완료. 촬영 자세에서 대기; goal을 비운 뒤 정지 → h → y.')
                while not self.aborted and not self._terminate.wait(.2):
                    pass
                return True
            self._reset_cycle_state()
            self._sweep_times = {}
            self._sweep_details = {}
            started = time.monotonic()
            ok = self._run_sweep_cycle()
            report = dict(mode='sweep', cycle=self._batch_completed+1, success=bool(ok),
                          total_s=time.monotonic()-started, stage_s=self._sweep_times,
                          detail_s=self._sweep_details)
            self._sweep_timing_pub.publish(String(data=json.dumps(report)))
            self.get_logger().info('sweep 사이클 시간: '+json.dumps(report, ensure_ascii=False))
            self.print_report()
            if not ok or self.aborted:
                self.emergency_stop('sweep 중단 — 그리퍼와 goal을 비운 뒤 h → y.', terminate=False)
                return False
            self._batch_progress.finish()
            self._batch_completed = self._batch_progress.completed

    def _run_sweep_cycle(self):
        timed = self._sweep_timed
        if not timed('observe', self._wait_for_targets):
            return False
        scene = self._targets['pick']['sweep_scene']
        # Destination stays in the collision scene throughout approach/pick.
        if not self._sweep_obstacle(False,scene['drop_top']):
            return False
        if not self.moveit.clear_collision_boxes([self._sweep_obstacle_id(True)]):
            return False
        if self.aborted:
            return False
        self._batch_progress.reserve()
        pre,grasp = timed('pick_approach', self._approach_ready,
                         'sweep Pick-Ready', 'pick', self.grasp_offset, self.approach_offset)
        if pre is None:
            return False
        with self._sweep_speed(contact=True):
            if not timed('pick_descent', self._goto_cartesian, 'sweep 집기 하강', grasp):
                return False
            if not timed('gripper_close', self.close_gripper):
                return False
            # No joint-space fallback inside clutter.
            if not timed('pick_retreat', self._goto_cartesian, 'sweep 집기 후퇴', pre):
                return False
        if not self._sweep_attach_pick():
            return False
        # Same joint waypoint used by the original working pick/place sequence.
        # The held box participates in collision checking along the curved path.
        if not timed('carry_camera', self._goto_joints, 'sweep 파지 후 촬영 자세', self.camera_ready_pose):
            return False
        if not self._sweep_obstacle(True,scene['source_top']+.01):
            return False
        diagonal = math.sqrt(sum(d*d for d in self._targets['pick']['dims']))
        if not self._sweep_obstacle(False,scene['drop_top']+diagonal+.01):
            return False
        chosen = timed('drop_ik', self._choose_sweep_drop,
                       self._pick_grasp_q, self._pick_contact_position)
        if chosen is None:
            self.get_logger().error('관측된 pile 위에서 도달 가능한 배출 자세가 없습니다.')
            return False
        release,floor = chosen
        if not timed('transfer_release', self._sweep_transfer_release,release,floor):
            return False
        if self.aborted:
            return False
        # We released above the maximum possible pile height. Return directly,
        # using the open-gripper path validated before release; no extra ascent.
        path = self._sweep_return_path
        ok = timed('return_camera', self.moveit._execute_trajectory,path,60.)
        return self._after_motion('sweep 촬영 복귀',ok,path,
                                  joint_target=dict(zip(self.arm_joints,self.camera_ready_pose)))

    def _sweep_transfer_release(self, release, floor):
        # Include the possible newly dropped box before planning transfer and
        # return, so letting go cannot invalidate the checked return route.
        pick = self._targets['pick']
        diagonal = math.sqrt(sum(d*d for d in pick['dims']))
        if not self._sweep_obstacle(False,pick['sweep_scene']['drop_top']+diagonal+.01):
            return False
        if self.aborted:
            return False
        started = time.monotonic()
        ok,path = self.moveit.move_to_pose(release,plan_only=True)
        self._sweep_details['drop_plan'] = time.monotonic()-started
        if not ok or path is None or self.aborted:
            return False
        state = trajectory_end_state(path, self._release_current_state())
        if state is None or not self._gripper_sweep_clear(state):
            return False
        opened = with_joint(state,self.gripper_joint,self.gripper_open_position)
        # Return is checked before letting go, with an open gripper.
        ok,back = self.moveit.move_to_joints(self.arm_joints,self.camera_ready_pose,
                                            start_state=opened,plan_only=True)
        if not ok or back is None or self.aborted:
            return False
        self._sweep_return_path = back
        started = time.monotonic()
        ok = self.moveit._execute_trajectory(path,60.)
        if not self._after_motion('sweep 배출 위치 이동',ok,path):
            return False
        self._sweep_details['drop_execute'] = time.monotonic()-started
        actual = self._achieved_ee_pose()
        if actual is None or math.dist(
                (actual.position.x,actual.position.y,actual.position.z),
                (release.position.x,release.position.y,release.position.z)) > .012:
            return False
        qa=(actual.orientation.x,actual.orientation.y,actual.orientation.z,actual.orientation.w)
        qr=(release.orientation.x,release.orientation.y,release.orientation.z,release.orientation.w)
        if abs(sum(a*b for a,b in zip(qa,qr))) < math.cos(math.radians(5.)/2):
            return False
        current = self._release_current_state()
        if current is None or not self._gripper_sweep_clear(current):
            return False
        if not self.open_gripper() or self.aborted:
            return False
        # Only withdraw after measured opening, not just mock action success.
        if not self._sweep_verify_open():
            return False
        if not self.moveit.clear_attached_box(HELD_BOX_ID,self.ee_link):
            return False
        self._sweep_drop_q = qr
        return True

    def _sweep_verify_open(self):
        deadline = time.monotonic()+2.
        guard = self._command_guard
        while time.monotonic() < deadline and not self.aborted:
            with guard.lock:
                if guard.reason(time.monotonic()):
                    return False
                width = guard.feedback.get(self.gripper_joint, float('nan'))
                if math.isfinite(width) and abs(width-self.gripper_open_position) < .005:
                    return True
            self._terminate.wait(.02)
        return False
