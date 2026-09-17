"""
MoveIt 2 를 rclpy 액션/서비스로 직접 호출하는 얇은 클라이언트.

왜 moveit_py 를 쓰지 않는가
--------------------------
ROS 2 Humble 의 moveit_py 에는 compute_cartesian_path 가 노출돼 있지 않다.
명세 4·5·7·8 단계는 Z 축 직선 보간(Cartesian Path)을 요구하므로 moveit_py 로는
구현 자체가 불가능하다. 여기서는 move_group 이 이미 띄워주는 표준 인터페이스를
직접 호출한다 — 추가 의존성이 없고 Humble/Jazzy 양쪽에서 동일하게 동작한다.

    /move_action            (moveit_msgs/action/MoveGroup)        플래닝 + 실행
    /compute_cartesian_path (moveit_msgs/srv/GetCartesianPath)    직선 경로 계산
    /execute_trajectory     (moveit_msgs/action/ExecuteTrajectory) 계산된 궤적 실행

스레드 모델
-----------
이 클래스의 메서드는 전부 블로킹이며, 노드를 spin 하는 스레드가 아닌 별도의
워커 스레드에서 호출되는 것을 전제로 한다. 퓨처 대기에 rclpy 의 spin 계열 함수를
쓰지 않고 threading.Event 를 쓰는 이유가 이것이다 (executor 와 충돌하지 않는다).
"""

import math
import threading

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetPositionFK,
    GetPositionIK,
    GetStateValidity,
    GetPlanningScene,
)
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


def _wait(future, timeout):
    """executor 를 건드리지 않고 퓨처 완료를 기다린다. 시간 초과 시 False."""
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout)


def make_start_state(joint_names, positions):
    """
    관절값으로 RobotState 를 만든다.

    is_diff=True 로 두면 MoveIt 이 현재 상태 위에 이 관절들만 덮어쓴다. 덕분에
    그리퍼 등 나머지 관절을 일일이 채우지 않아도 된다.
    """
    state = RobotState()
    state.is_diff = True
    state.joint_state = JointState()
    state.joint_state.name = list(joint_names)
    state.joint_state.position = [float(v) for v in positions]
    return state


def trajectory_end_state(trajectory, base_state=None):
    """궤적의 마지막 점을 다음 계획의 시작 상태로 쓸 RobotState 로 변환한다."""
    points = trajectory.joint_trajectory.points
    if not points:
        return None
    values = {} if base_state is None else dict(zip(
        base_state.joint_state.name, base_state.joint_state.position))
    values.update(zip(trajectory.joint_trajectory.joint_names, points[-1].positions))
    return make_start_state(values.keys(), values.values())


class MoveItClient:
    """move_group 에 대한 최소한의 동기식 래퍼."""

    def __init__(self, node, group_name, ee_link, base_frame, callback_group):
        self._node = node
        self._log = node.get_logger()
        self.group_name = group_name
        self.ee_link = ee_link
        self.base_frame = base_frame

        self._move_group = ActionClient(
            node, MoveGroup, 'move_action', callback_group=callback_group)
        self._execute = ActionClient(
            node, ExecuteTrajectory, 'execute_trajectory', callback_group=callback_group)
        self._cartesian = node.create_client(
            GetCartesianPath, 'compute_cartesian_path', callback_group=callback_group)
        self._fk = node.create_client(
            GetPositionFK, 'compute_fk', callback_group=callback_group)
        self._ik = node.create_client(
            GetPositionIK, 'compute_ik', callback_group=callback_group)
        self._apply_scene = node.create_client(
            ApplyPlanningScene, 'apply_planning_scene', callback_group=callback_group)
        self._state_validity = node.create_client(
            GetStateValidity, 'check_state_validity', callback_group=callback_group)
        self._get_scene = node.create_client(
            GetPlanningScene, 'get_planning_scene', callback_group=callback_group)

        # 실행 중인 액션 goal — 비상 정지 때 취소해야 하므로 붙잡아 둔다.
        self._active_goal = None
        self._goal_lock = threading.Lock()
        self._unsettled_goals = set()

        # 플래닝 튜닝값 — 컨트롤러가 파라미터에서 읽어 덮어쓴다.
        self.planning_time = 5.0
        self.planning_attempts = 10
        self.velocity_scaling = 0.1
        self.acceleration_scaling = 0.1
        self.goal_position_tolerance = 0.005     # [m]
        self.goal_orientation_tolerance = 0.02   # [rad]

    # ------------------------------------------------------------------
    # 준비
    # ------------------------------------------------------------------
    def wait_for_servers(self, timeout=30.0):
        """move_group 의 액션/서비스가 전부 올라올 때까지 기다린다."""
        self._log.info('move_group 인터페이스 대기 중...')
        if not self._move_group.wait_for_server(timeout_sec=timeout):
            self._log.error("'move_action' 액션 서버를 찾지 못했습니다. move_group 이 떴는지 확인하세요.")
            return False
        if not self._execute.wait_for_server(timeout_sec=timeout):
            self._log.error("'execute_trajectory' 액션 서버를 찾지 못했습니다.")
            return False
        if not self._cartesian.wait_for_service(timeout_sec=timeout):
            self._log.error("'compute_cartesian_path' 서비스를 찾지 못했습니다.")
            return False
        if not self._fk.wait_for_service(timeout_sec=timeout):
            self._log.error("'compute_fk' 서비스를 찾지 못했습니다.")
            return False
        if not self._ik.wait_for_service(timeout_sec=timeout):
            self._log.error("'compute_ik' 서비스를 찾지 못했습니다.")
            return False
        if not self._apply_scene.wait_for_service(timeout_sec=timeout):
            self._log.error("'apply_planning_scene' 서비스를 찾지 못했습니다.")
            return False
        if not self._state_validity.wait_for_service(timeout_sec=timeout):
            self._log.error("'check_state_validity' 서비스를 찾지 못했습니다.")
            return False
        self._log.info('move_group 준비 완료.')
        return True

    # ------------------------------------------------------------------
    # 플래닝 씬
    # ------------------------------------------------------------------
    def check_state_validity(self, state, timeout=5.0):
        """전체 로봇(열린 손가락 포함)의 충돌 검사. 서비스 오류도 실패로 처리."""
        request = GetStateValidity.Request()
        request.robot_state = state
        future = self._state_validity.call_async(request)
        if not _wait(future, timeout):
            self._log.error('로봇 상태 충돌 검사 시간 초과.')
            return False
        response = future.result()
        if response is None:
            return False
        if not response.valid:
            pairs = sorted({f'{c.contact_body_1} / {c.contact_body_2}'
                            for c in response.contacts})
            self._log.warn('로봇 상태 검사 실패: ' + (', '.join(pairs) or '관절 한계/제약'))
        return bool(response.valid)

    def remove_collision_box(self, object_id, timeout=10.0):
        """실제로 치운 것으로 확인된 물체만 호출자가 명시적으로 제거한다."""
        obj = CollisionObject(id=object_id, operation=CollisionObject.REMOVE)
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.world.collision_objects.append(obj)
        future = self._apply_scene.call_async(ApplyPlanningScene.Request(scene=scene))
        if not _wait(future, timeout):
            return False
        response = future.result()
        return response is not None and response.success

    def clear_collision_boxes(self, object_ids, timeout=5.0):
        """Remove only listed world objects; already absent objects are success."""
        if not self._get_scene.wait_for_service(timeout_sec=timeout):
            return False
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        future = self._get_scene.call_async(request)
        if not _wait(future, timeout) or future.result() is None:
            return False
        present = {obj.id for obj in future.result().scene.world.collision_objects}
        for name in set(object_ids) & present:
            if not self.remove_collision_box(name, timeout):
                return False
        future = self._get_scene.call_async(request)
        if not _wait(future, timeout) or future.result() is None:
            return False
        remaining = {obj.id for obj in future.result().scene.world.collision_objects}
        return not (set(object_ids) & remaining)

    def motion_requests_settled(self):
        """False while an earlier request could still be accepted/executing."""
        with self._goal_lock:
            return not self._unsettled_goals

    def add_collision_box(self, object_id, frame_id, pose, size, timeout=10.0):
        """
        플래닝 씬에 고정 충돌 상자를 추가한다 (같은 id 가 있으면 교체).

        이 객체가 들어가면 move_group 이 경로를 만들 때 알아서 피한다.
        """
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in size]

        obj = CollisionObject()
        obj.header.frame_id = frame_id
        obj.header.stamp = self._node.get_clock().now().to_msg()
        obj.id = object_id
        obj.primitives.append(box)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects.append(obj)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._apply_scene.call_async(request)
        if not _wait(future, timeout):
            self._log.error(f"플래닝 씬에 '{object_id}' 추가 시간 초과.")
            return False
        response = future.result()
        if response is None or not response.success:
            self._log.error(f"플래닝 씬에 '{object_id}' 추가 실패.")
            return False
        return True

    def set_attached_box(self, object_id, link, pose, size, touch_links, timeout=5.0):
        """Model a held box in link coordinates for all subsequent planning."""
        obj = CollisionObject(id=object_id, operation=CollisionObject.ADD)
        obj.header.frame_id = link
        obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size))]
        obj.primitive_poses = [pose]
        attached = AttachedCollisionObject(link_name=link, object=obj, touch_links=touch_links)
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        future = self._apply_scene.call_async(ApplyPlanningScene.Request(scene=scene))
        return (_wait(future,timeout) and future.result() is not None
                and future.result().success)

    def clear_attached_box(self, object_id, link, timeout=5.0):
        """Only after measured release or the operator's empty-gripper reset."""
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [AttachedCollisionObject(
            link_name=link, object=CollisionObject(id=object_id,operation=CollisionObject.REMOVE))]
        future = self._apply_scene.call_async(ApplyPlanningScene.Request(scene=scene))
        if not (_wait(future,timeout) and future.result() is not None and future.result().success):
            return False
        # Detaching creates a world object. Remove AFTER applying the detach,
        # independent of the server's ordering of world and robot scene diffs.
        return self.clear_collision_boxes([object_id],timeout)

    # ------------------------------------------------------------------
    # Joint Space — 관절 공간 보간
    # ------------------------------------------------------------------
    def move_to_joints(self, joint_names, positions,
                       start_state=None, plan_only=False, timeout=60.0, scene_diff=None):
        """
        관절 목표로 플래닝하고 (plan_only 가 아니면) 실행한다.

        :returns: (성공 여부, 계획된 궤적 또는 None)
        """
        constraints = Constraints()
        for name, value in zip(joint_names, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        return self._plan_and_execute(constraints, timeout, start_state, plan_only,
                                      scene_diff=scene_diff)

    # ------------------------------------------------------------------
    # Pose 목표 — 관절 공간 보간으로 데카르트 포즈에 도달
    # ------------------------------------------------------------------
    def move_to_pose(self, pose, start_state=None, plan_only=False, timeout=60.0,
                     free_yaw=False, tilt_tolerance=None, position_only=False):
        """
        엔드 이펙터 포즈 목표로 플래닝하고 (plan_only 가 아니면) 실행한다 (3·6 단계).

        경로 자체는 OMPL 이 관절 공간에서 보간한다. 직선 이동이 필요한 구간은
        move_cartesian() 을 써야 한다.

        position_only=True 면 자세 제약을 아예 넣지 않는다 (위치만). 그리퍼
        방향은 IK 가 자유롭게 고른다. "IK 가 이상한지 / 입력 자세가 이상한지"
        를 가르는 실험용.

        자세 제약(pose.orientation 프레임 기준):
          - Z 축(접근축) 둘레 회전 = yaw(그리퍼 jaw 정렬). free_yaw=True 면
            무제한(2π), 아니면 goal_orientation_tolerance.
          - X·Y 축 둘레 회전 = 접근축 기울기(pitch/roll). tilt_tolerance 가
            주어지면 그 값, 아니면 goal_orientation_tolerance.

        손목 가동범위가 좁은 팔(PiPER joint5 ±1.2217)에서, 직육면체 물체는
        tilt_tolerance 를 열어 (yaw 는 유지) 수직 파지 IK 를 성립시킨다.

        :returns: (성공 여부, 계획된 궤적 또는 None)
        """
        constraints = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link
        pc.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.goal_position_tolerance]
        region = BoundingVolume()
        region.primitives.append(sphere)
        region_pose = Pose()
        region_pose.position = pose.position
        region_pose.orientation.w = 1.0
        region.primitive_poses.append(region_pose)
        pc.constraint_region = region
        constraints.position_constraints.append(pc)

        if position_only:
            return self._plan_and_execute(constraints, timeout, start_state, plan_only)

        tilt = tilt_tolerance if tilt_tolerance else self.goal_orientation_tolerance
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link
        oc.orientation = pose.orientation
        # X·Y 둘레 = 접근축 기울기. tilt_tolerance 만큼 허용해 IK 여유를 준다.
        oc.absolute_x_axis_tolerance = tilt
        oc.absolute_y_axis_tolerance = tilt
        # Z 둘레 = yaw. free_yaw 면 사실상 무제한(2π), 아니면 마커에 정렬.
        oc.absolute_z_axis_tolerance = (
            2.0 * math.pi if free_yaw else self.goal_orientation_tolerance)
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)

        return self._plan_and_execute(constraints, timeout, start_state, plan_only)

    def compute_fk(self, joint_names, positions, timeout=10.0):
        """
        관절값으로 ee_link 의 base_frame 기준 Pose 를 구한다 (정기구학). 실패 시 None.

        free_yaw 로 Pick/Place-Ready 를 계획한 뒤, 실제로 도달한 그리퍼 자세를
        읽어 이어지는 Cartesian 직선 구간의 waypoint 자세로 쓰기 위한 것이다.
        그래야 직선 하강 중에 yaw 가 뒤틀리지 않는다.
        """
        request = GetPositionFK.Request()
        request.header.frame_id = self.base_frame
        request.fk_link_names = [self.ee_link]
        request.robot_state = make_start_state(joint_names, positions)
        future = self._fk.call_async(request)
        if not _wait(future, timeout):
            self._log.error('compute_fk 응답 시간 초과.')
            return None
        response = future.result()
        if (response is None
                or response.error_code.val != MoveItErrorCodes.SUCCESS
                or not response.pose_stamped):
            self._log.error('compute_fk 실패.')
            return None
        return response.pose_stamped[0].pose

    def compute_ik(self, pose, start_state=None, ik_timeout=0.05, timeout=3.0,
                   avoid_collisions=True):
        """
        base_frame 기준 pose 에 대한 IK 해가 (충돌 없이) 존재하면 True.

        후보 파지 자세를 순서대로 값싸게 걸러내는 용도 (플래닝 없이 IK 만).
        """
        request = GetPositionIK.Request()
        r = request.ik_request
        r.group_name = self.group_name
        r.ik_link_name = self.ee_link
        r.robot_state = start_state if start_state is not None else RobotState()
        r.robot_state.is_diff = True
        r.avoid_collisions = avoid_collisions
        r.pose_stamped.header.frame_id = self.base_frame
        r.pose_stamped.pose = pose
        r.timeout.sec = int(ik_timeout)
        r.timeout.nanosec = int((ik_timeout - int(ik_timeout)) * 1e9)
        future = self._ik.call_async(request)
        if not _wait(future, timeout):
            return False
        response = future.result()
        return (response is not None
                and response.error_code.val == MoveItErrorCodes.SUCCESS)

    # ------------------------------------------------------------------
    # Cartesian Path — 작업 공간 직선 보간
    # ------------------------------------------------------------------
    def move_cartesian(self, waypoints, max_step=0.005, min_fraction=1.0,
                       start_state=None, plan_only=False, timeout=60.0):
        """
        시작 상태에서 waypoints 를 잇는 직선 경로를 계산하고 (plan_only 가 아니면)
        실행한다 (4·5·7·8 단계).

        compute_cartesian_path 서비스는 시간 파라미터화까지 마친 궤적을 돌려주므로
        결과를 execute_trajectory 로 바로 넘길 수 있다.

        [Humble] GetCartesianPath.Request 에는 속도/가속 스케일 필드가 없어서
        (Iron 부터 추가됨) 서비스가 관절 한계 풀 스피드로 시간 파라미터화한다.
        그대로 실행하면 짧은 구간에서 급가속/급감속해 "쿵" 한다. 받은 궤적의
        시간축을 1/velocity_scaling 배로 늘려(속도·가속도도 그만큼 축소) 관절
        공간 이동과 비슷한 속도로 맞춘다. _retime_trajectory 참고.

        :returns: (성공 여부, 계획된 궤적 또는 None)
        """
        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self._node.get_clock().now().to_msg()
        # start_state 가 없으면 현재 상태에서 출발한다.
        request.start_state = start_state if start_state is not None else RobotState()
        request.start_state.is_diff = True
        request.group_name = self.group_name
        request.link_name = self.ee_link
        request.waypoints = list(waypoints)
        request.max_step = float(max_step)
        request.jump_threshold = 0.0                # 관절 점프 검사 비활성화
        request.avoid_collisions = True

        future = self._cartesian.call_async(request)
        if not _wait(future, timeout):
            self._log.error('compute_cartesian_path 응답 시간 초과.')
            return False, None
        response = future.result()

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self._log.error(
                f'Cartesian 경로 계산 실패 (error_code={response.error_code.val}).')
            return False, None
        if not math.isfinite(response.fraction) or response.fraction < min_fraction - 1e-6:
            self._log.error(
                f'Cartesian 경로가 {response.fraction * 100.0:.1f}% 만 풀렸습니다 '
                f'(최소 {min_fraction * 100.0:.0f}% 필요). '
                '목표가 작업 범위를 벗어났거나 충돌이 예상됩니다.')
            return False, None

        solution = response.solution
        # Uniform retiming must respect BOTH limits (a scales with time^-2).
        self._retime_trajectory(solution, min(self.velocity_scaling,
                                              math.sqrt(self.acceleration_scaling)))

        if plan_only:
            self._log.info(
                f'Cartesian 경로 {response.fraction * 100.0:.1f}% 해결 (계획만).')
            return True, solution

        self._log.info(f'Cartesian 경로 {response.fraction * 100.0:.1f}% 해결 — 실행합니다.')
        return self._execute_trajectory(solution, timeout), solution

    @staticmethod
    def _retime_trajectory(trajectory, scaling):
        """
        궤적을 균일 시간 스케일링한다. scaling=0.1 이면 시간축을 10배로 늘리고
        각 점의 속도는 0.1배, 가속도는 0.01배로 줄인다 (위치는 그대로).

        Humble 의 compute_cartesian_path 가 속도 스케일을 못 받아 풀 스피드로
        돌려주는 것을 관절 공간 이동 속도에 맞추기 위한 것.
        """
        if not scaling or scaling >= 1.0 or scaling <= 0.0:
            return
        k = 1.0 / scaling
        for point in trajectory.joint_trajectory.points:
            total = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            total *= k
            point.time_from_start.sec = int(total)
            point.time_from_start.nanosec = int(round((total - int(total)) * 1e9))
            point.velocities = [v * scaling for v in point.velocities]
            point.accelerations = [a * scaling * scaling for a in point.accelerations]

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------
    def _plan_and_execute(self, goal_constraints, timeout, start_state=None,
                          plan_only=False, scene_diff=None):
        if scene_diff is not None and not plan_only:
            raise ValueError('Hypothetical scene is for planning only')
        goal = MoveGroup.Goal()
        request = goal.request

        request.group_name = self.group_name
        request.num_planning_attempts = self.planning_attempts
        request.allowed_planning_time = self.planning_time
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling
        # start_state 를 주면 그 자세에서 출발하는 것으로 계획한다. 프리뷰 모드에서
        # 앞 단계의 끝점을 다음 단계의 시작점으로 이어 붙이는 데 쓴다.
        request.start_state = start_state if start_state is not None else RobotState()
        request.start_state.is_diff = True
        request.goal_constraints.append(goal_constraints)
        # 작업 공간 경계를 넉넉히 잡아둔다 (기본값이 0 이면 OMPL 이 경고를 낸다).
        request.workspace_parameters.header.frame_id = self.base_frame
        request.workspace_parameters.min_corner.x = -2.0
        request.workspace_parameters.min_corner.y = -2.0
        request.workspace_parameters.min_corner.z = -2.0
        request.workspace_parameters.max_corner.x = 2.0
        request.workspace_parameters.max_corner.y = 2.0
        request.workspace_parameters.max_corner.z = 2.0

        # plan_only 가 False 면 move_group 이 플래닝과 실행을 함께 수행한다.
        goal.planning_options.plan_only = bool(plan_only)
        if scene_diff is not None:
            goal.planning_options.planning_scene_diff = scene_diff
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        result = self._send_goal(self._move_group, goal, timeout, 'move_action')
        if result is None:
            return False, None
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._log.error(f'플래닝/실행 실패 (error_code={result.error_code.val}).')
            return False, None
        return True, result.planned_trajectory

    def _execute_trajectory(self, trajectory, timeout):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        result = self._send_goal(self._execute, goal, timeout, 'execute_trajectory')
        if result is None:
            return False
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._log.error(f'궤적 실행 실패 (error_code={result.error_code.val}).')
            return False
        return True

    def _send_goal(self, client, goal, timeout, label, *, return_wrapper=False):
        """Track requests through their terminal result, including late acceptance."""
        token = object()
        accepted = threading.Event()
        expired = threading.Event()
        holder = {}
        with self._goal_lock:
            self._unsettled_goals.add(token)

        def settled(_=None):
            with self._goal_lock:
                self._unsettled_goals.discard(token)

        def on_sent(future):
            try:
                handle = future.result()
                holder['handle'] = handle
                if handle is None or not handle.accepted:
                    settled()
                    return
                result_future = handle.get_result_async()
                holder['result'] = result_future
                result_future.add_done_callback(settled)
                if expired.is_set():
                    handle.cancel_goal_async()
            except Exception as exc:
                # Keep an unresolved token: reset must not reopen control when
                # acceptance or completion of an earlier request is unknown.
                self._log.error(f'{label}: 목표 응답 확인 실패: {exc}')
            finally:
                accepted.set()

        try:
            send_future = client.send_goal_async(goal)
            send_future.add_done_callback(on_sent)
        except Exception:
            settled()
            raise
        if not accepted.wait(timeout):
            expired.set()
            handle = holder.get('handle')
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
            self._log.error(f'{label}: 목표 전송 시간 초과. 늦게 승인된 목표도 취소합니다.')
            return None
        handle, result_future = holder.get('handle'), holder.get('result')
        if handle is None or not handle.accepted or result_future is None:
            self._log.error(f'{label}: 목표가 거부되었거나 응답을 확인하지 못했습니다.')
            return None
        with self._goal_lock:
            self._active_goal = handle
        try:
            if not _wait(result_future, timeout):
                handle.cancel_goal_async()
                self._log.error(f'{label}: 실행 결과 시간 초과. 목표를 취소합니다.')
                return None
            wrapper = result_future.result()
            if wrapper is None:
                return None
            return wrapper if return_wrapper else wrapper.result
        finally:
            with self._goal_lock:
                if self._active_goal is handle:
                    self._active_goal = None

    def cancel_active_goal(self):
        """
        실행 중인 액션 goal 을 취소한다 (비상 정지용).

        move_group 이 궤적 실행을 중단시킨다. 응답을 기다리지 않는다 —
        비상 상황에서 블로킹하면 안 되기 때문이다.
        """
        with self._goal_lock:
            handle = self._active_goal
        if handle is None:
            return False
        try:
            handle.cancel_goal_async()
            self._log.warn('실행 중인 궤적 goal 을 취소했습니다.')
            return True
        except Exception as exc:  # noqa: BLE001 - 정지 경로에서는 어떤 예외도 삼킨다
            self._log.error(f'goal 취소 실패: {exc}')
            return False
