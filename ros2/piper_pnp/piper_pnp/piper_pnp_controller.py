"""
piper_pnp_controller
====================

AgileX PiPER eye-in-hand ArUco Pick & Place 컨트롤러 (9단계 상태 머신).

동작 개요
---------
Camera-Ready 자세에서 ArUco 마커 두 개(Pick 용 / Place 용)를 인식하고, 각 마커의
위치와 자세로부터 파지 포즈를 만들어 집어서 옮긴다.

파지 자세는 다음 한 줄로 결정된다::

    R_grasp = R_marker * Rx(180도)

  * 평면에 놓인 마커 -> roll/pitch 가 0 이라 yaw 만 남고, 그리퍼는 똑바로 아래를
    향한다 (인형뽑기 방식).
  * 기울어진 마커   -> 기울기를 그대로 물려받아 그리퍼도 같이 기운다.

접근/후퇴 10cm 는 월드 Z 축이 아니라 **마커 법선**(= 그리퍼 접근 축)을 따른다.
마커가 평면에 놓여 있으면 두 축이 정확히 일치하므로 명세의 "Z축 위 10cm" 와
같은 동작이 되고, 기울어진 경우에만 곧게 파고드는 쪽으로 자연스럽게 바뀐다.

이동 방식은 목적에 따라 엄격히 나뉜다.

    Joint Space (관절 공간 보간)  : 1, 2, 3, 5, 6, 8, 9 단계
    Cartesian Path (직선 보간)    : 4, 5, 7, 8 단계의 접근/후퇴 구간
"""

import math
import signal
import statistics
import threading
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from std_srvs.srv import Empty, SetBool, Trigger
from std_msgs.msg import Header, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from moveit_msgs.msg import DisplayTrajectory

from piper_pnp.geometry import (
    approach_position,
    grasp_candidates,
    grasp_orientation,
    quat_from_rpy,
    quat_multiply,
    quat_rotate_vector,
    transform_pose,
    yaw_of,
)
from piper_pnp.moveit_client import MoveItClient, make_start_state, trajectory_end_state
from piper_pnp.real_control import RealControlMixin
from piper_pnp.release_checks import ReleaseChecksMixin
from piper_pnp.batch_control import BatchControlMixin
from piper_pnp.sweep_control import SweepControlMixin
from piper_pnp.feed_drop import FeedDropMixin
from piper_pnp.batch_workspace import BatchWorkspace, BatchProgress
from piper_pnp.restart_control import RestartControlMixin
from piper_pnp.object_models import ObjectCatalog
from piper_pnp.object_manipulation import grip_width, level_place_candidates, place_positions, check_level_release
from piper_pnp.place_alignment import (
    aligned_place_candidates, held_box_orientation, long_axis_error,
)

# ros2_aruco 가 워크스페이스에 없어도 (시뮬레이션 전용 사용) 노드가 뜨도록 한다.
try:
    from ros2_aruco_interfaces.msg import ArucoMarkers
except ImportError:  # pragma: no cover - 설치 여부에 따라 갈린다
    ArucoMarkers = None

PICK = 'pick'
PLACE = 'place'


class PiperPnpController(RestartControlMixin, FeedDropMixin, SweepControlMixin, BatchControlMixin, ReleaseChecksMixin, RealControlMixin, Node):
    """9단계 Pick & Place 상태 머신."""

    def __init__(self):
        super().__init__('piper_pnp_controller')

        # 액션/서비스 콜백이 워커 스레드의 대기와 맞물려 돌아가야 하므로 재진입 그룹을 쓴다.
        self._cb_group = ReentrantCallbackGroup()

        self._declare_parameters()
        self._read_parameters()

        # ---------------- TF ----------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---------------- 타겟 상태 ----------------
        # key -> {'position': (x,y,z), 'orientation': (x,y,z,w)} — 전부 base_frame 기준
        self._targets = {}
        # 확정 전 최근 검출 샘플: key -> deque[(t_monotonic, position, orientation)]
        self._target_samples = {PICK: deque(), PLACE: deque()}
        self._targets_lock = threading.Lock()
        self._targets_ready = threading.Event()
        self._target_not_before_ns = 0
        self._last_target_stamp = {}
        self._target_stamps = {}
        self._frozen = False          # Step 2 에서 좌표를 확정한 뒤에는 갱신하지 않는다
        self._sample_object_identity = None
        self._start_requested = threading.Event()

        # 프리뷰 모드에서 다음 단계의 시작 상태(= 앞 단계 궤적의 끝점).
        # 일반 모드에서는 None 이며, 그때는 MoveIt 이 실제 현재 상태를 쓴다.
        self._preview_state = None
        self._pick_grasp_q = None      # Step 3 에서 실제 도달한 파지 자세 (place 미러링용)
        self._pick_contact_position = None
        self._report = []

        # 비상 정지 / 단계 승인
        self._abort = threading.Event()      # 현재 동작/사이클 중단
        self._terminate = threading.Event()  # 명시적인 프로세스 종료 (Ctrl-C)
        self._continue_event = threading.Event()
        self._waiting_confirmation = False

        # ---------------- 퍼블리셔 / 서브스크라이버 ----------------
        self._marker_pub = self.create_publisher(Marker, self.marker_topic_out, 10)
        # RViz 의 MotionPlanning "Planned Path" 가 이 토픽을 애니메이션한다.
        self._display_pub = self.create_publisher(
            DisplayTrajectory, 'display_planned_path', 10)
        self.create_timer(1.0, self._publish_target_markers, callback_group=self._cb_group)
        if self._object_catalog is not None:
            self.create_subscription(String, '/cuboid_pose_bridge/target', self._on_cuboid_target,
                                     1, callback_group=self._cb_group)

        # PoseArray(/aruco_poses)는 ID 가 없어 인덱스로 pick/place 를 가른다.
        # ros2_aruco 는 /aruco_markers 와 /aruco_poses 를 둘 다 발행하므로, 실기에서
        # 이 폴백을 켜두면 인덱스 매핑이 ID 기반 결과를 덮어써 pick/place 가 뒤섞인다.
        # 기본은 OFF. ros2_aruco_interfaces 가 없거나(순수 시뮬) use_pose_array_fallback
        # 을 켰을 때만 구독한다. README 4절의 PoseArray 주입 테스트가 후자.
        use_pose_array = self.use_pose_array_fallback or ArucoMarkers is None
        if use_pose_array:
            self.create_subscription(
                PoseArray, self.aruco_poses_topic, self._on_aruco_poses, 10,
                callback_group=self._cb_group)

        if ArucoMarkers is not None:
            self.create_subscription(
                ArucoMarkers, self.aruco_markers_topic, self._on_aruco_markers, 10,
                callback_group=self._cb_group)
        else:
            self.get_logger().warn(
                'ros2_aruco_interfaces 를 임포트할 수 없어 ID 기반 인식이 비활성화됩니다. '
                f'{self.aruco_poses_topic} (PoseArray, 인덱스 0=pick / 1=place) 만 사용합니다.')

        if use_pose_array and ArucoMarkers is not None:
            self.get_logger().warn(
                'use_pose_array_fallback=true — /aruco_poses(인덱스 기반)가 '
                'ID 기반 결과를 덮어쓸 수 있습니다. 실기에서는 끄세요.')

        # ---------------- MoveIt ----------------
        self.moveit = MoveItClient(
            self, self.planning_group, self.ee_link, self.base_frame, self._cb_group)
        self.moveit.planning_time = self.planning_time
        self.moveit.planning_attempts = self.planning_attempts
        self.moveit.velocity_scaling = self.velocity_scaling
        self.moveit.acceleration_scaling = self.acceleration_scaling

        # ---------------- 그리퍼 ----------------
        self._gripper_client = None
        if self.use_real_gripper:
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient
            self._gripper_client = ActionClient(
                self, FollowJointTrajectory, self.gripper_action,
                callback_group=self._cb_group)

        # ---------------- 실기 팔 서비스 (비상 정지용) ----------------
        # agx_arm_ctrl 이 제공한다. 시뮬레이션에는 없으므로 호출 전에 준비 여부를 본다.
        self._control_enable_client = self.create_client(
            SetBool, self.control_enable_service, callback_group=self._cb_group)
        self._emergency_stop_client = self.create_client(
            Empty, self.emergency_stop_service, callback_group=self._cb_group)

        self._init_real_control()
        self._init_restart_control()
        self.create_timer(0.1, self._expire_targets, callback_group=self._cb_group)

        # ---------------- 제어 서비스 ----------------
        self.create_service(
            Trigger, '~/start', self._on_start_request, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/stop', self._on_stop_request, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/resume', self._on_resume_request, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/continue', self._on_continue_request, callback_group=self._cb_group)

        self.get_logger().info(
            f'컨트롤러 준비 — group={self.planning_group}, ee={self.ee_link}, '
            f'pick_id={self.pick_marker_id}, place_id={self.place_marker_id}, '
            f'real_gripper={self.use_real_gripper}')

    # ==================================================================
    # 파라미터
    # ==================================================================
    def _declare_parameters(self):
        self.declare_parameter('object_models_file', '')
        self.declare_parameter('object_table_z', 0.0)
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('ee_link', 'tcp_link')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter(
            'arm_joints', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])

        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('planning_attempts', 10)
        self.declare_parameter('velocity_scaling', 0.1)
        self.declare_parameter('acceleration_scaling', 0.1)
        self.declare_parameter('cartesian_max_step', 0.005)
        self.declare_parameter('cartesian_min_fraction', 1.0)

        self.declare_parameter('zero_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('camera_ready_pose', [0.0, 1.2, -1.4, 0.0, 0.8, 0.0])

        self.declare_parameter('aruco_markers_topic', '/aruco_markers')
        self.declare_parameter('aruco_poses_topic', '/aruco_poses')
        # /aruco_poses(PoseArray, ID 없음, 인덱스=pick/place) 구독 여부.
        # 실기에서는 끈다 (ID 기반 /aruco_markers 를 덮어쓰기 때문). README 4절의
        # PoseArray 로컬 주입 테스트에서만 켠다.
        self.declare_parameter('use_pose_array_fallback', False)
        self.declare_parameter('pick_marker_id', 0)
        self.declare_parameter('place_marker_id', 1)
        self.declare_parameter('marker_wait_timeout', 60.0)
        self.declare_parameter('target_max_age_s', 0.5)
        self.declare_parameter('guard_real_commands', False)
        self.declare_parameter('feedback_max_age_s', 0.25)
        self.declare_parameter('feedback_joint_tolerance', 0.03)
        self.declare_parameter('feedback_arrival_timeout', 5.0)
        self.declare_parameter('tf_timeout', 1.0)
        # Step 2 좌표 확정(_frozen)을 순간 글리치가 아니라 안정된 검출로만 하도록.
        # 최근 marker_stable_window 초 안에 marker_stable_samples 개 이상 모이고,
        # 그 위치들이 중앙값에서 marker_stable_spread [m] 이내로 모여 있을 때만
        # 확정한다. 확정값은 (한 프레임이 아니라) 중앙값. 검출이 튀는 동안은
        # 확정을 미루므로, 그래도 안 잡히면 marker_wait_timeout 에서 실패한다.
        self.declare_parameter('marker_stable_samples', 4)
        self.declare_parameter('marker_stable_window', 3.0)
        self.declare_parameter('marker_stable_spread', 0.01)
        # yaw와 전체 회전각 편차 모두 이 한도 이내여야 한다.
        self.declare_parameter('marker_stable_yaw_deg', 15.0)
        self.declare_parameter('approach_offset', 0.10)
        self.declare_parameter('grasp_offset', -0.01)
        self.declare_parameter('place_offset', 0.01)
        # Place-Ready(6단계) 접근 높이. Pick 은 approach_offset 을 쓰지만
        # (pick IK 여유 때문에 낮게 묶여 있음) Place 는 이 값으로 place_offset
        # 위에 하강 여유를 둔다. place_offset 보다 커야 한다.
        self.declare_parameter('place_approach_offset', 0.06)
        # true 면 Place 자세를 새로 탐색하지 않고, 집을 때 자세(q_pick)를 미러링해
        # 물체를 지면과 수평으로 내려놓는다. yaw 는 place 마커에 정렬.
        #   q_place = q_marker_place · q_marker_pick⁻¹ · q_pick
        # IK 불가면 박스 긴 변의 수평 방향을 유지하는 기울기만 허용한다.
        self.declare_parameter('place_match_pick_tilt', True)
        self.declare_parameter('place_alignment_tolerance_deg', 5.0)
        self.declare_parameter('place_tilt_azimuths_deg',
                               [0., 180., 90., 270., 45., 135., 225., 315.])
        # 파지 자세 결정 방식 (Pick/Place-Ready, 3·6단계).
        #   candidates    : 이산 자세 후보(yaw/tilt)를 편차 작은 순으로 /compute_ik
        #                   로 걸러 첫 통과 자세를 '정확히' 실행. 안 되면 tolerance
        #                   로 폴백. (권장 — modern 논문식 candidate generation)
        #   tolerance     : OrientationConstraint 에 tilt tolerance 박스를 줌.
        #                   grasp_tilt_tolerance / free_grasp_yaw 사용.
        #   position_only : 자세 제약 없음, 위치만. 접근/하강 월드 수직. (디버그)
        self.declare_parameter('grasp_ik_mode', 'candidates')
        self.declare_parameter('grasp_tilt_tolerance', 0.8)
        self.declare_parameter('free_grasp_yaw', False)
        # candidates 모드: 후보 나열. yaw 는 마커 기준 오프셋[deg], tilt 는 접근축을
        # 마커 법선에서 기울이는 크기[deg], azimuth 는 tilt 방향[deg](tilt=0 이면 무시).
        # yaw 0/90/180/270 은 2조 그리퍼엔 같은 파지지만 joint6 관절값이 달라,
        # yaw 0° 가 한계로 IK 안 될 때 나머지가 풀리기도 한다. tilt 작은 것이
        # 항상 우선이므로 tilt=0 인 4개 yaw 를 다 시도한 뒤에야 기울인다.
        self.declare_parameter('grasp_yaw_candidates_deg', [0.0, 90.0, 180.0, 270.0])
        self.declare_parameter('grasp_tilt_candidates_deg', [0.0, 15.0, 30.0, 45.0])
        self.declare_parameter('grasp_tilt_azimuths_deg', [0.0, 180.0, 90.0, 270.0])
        self.declare_parameter('grasp_ik_check_timeout', 0.05)

        self.declare_parameter(
            'arm_controller_action', '/arm_controller/follow_joint_trajectory')
        self.declare_parameter('controller_wait_timeout', 30.0)

        self.declare_parameter('use_real_gripper', False)
        self.declare_parameter('gripper_action', '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_joint', 'gripper')
        self.declare_parameter('gripper_open_position', 0.07)
        self.declare_parameter('gripper_close_position', 0.0)
        self.declare_parameter('gripper_settle_time', 1.0)

        # 프리뷰: 실행 없이 9단계를 이어 붙여 계획만 하고 RViz 에 보여준다.
        self.declare_parameter('preview_only', False)
        self.declare_parameter('preview_step_delay', 2.0)

        # 테이블: 플래닝 씬에 넣어 팔이 상판을 쓸지 않도록 한다.
        self.declare_parameter('add_table', True)
        self.declare_parameter('table_size', [1.2, 1.2, 0.1])
        self.declare_parameter('table_top_z', -0.01)
        self.declare_parameter('table_center_xy', [0.0, 0.0])

        # 타겟 정합성 검사 (PiPER 작업 반경 기준)
        self.declare_parameter('min_reach', 0.12)
        self.declare_parameter('max_reach', 0.75)

        # 안전: 실기 팔 서비스 (시뮬에는 없으므로 없으면 건너뛴다)
        self.declare_parameter('control_enable_service', '/control_enable')
        self.declare_parameter('emergency_stop_service', '/emergency_stop')

        # 단계별 확인: 각 단계 실행 전에 승인을 기다린다.
        # 승인은 ~/continue 서비스로 들어온다 (보통 estop 노드의 c 키).
        self.declare_parameter('step_confirm', False)
        self.declare_parameter('step_confirm_timeout', 300.0)

        self.declare_parameter('auto_start', True)
        # true 면 홈(Step 9) 복귀 후 estop 콘솔의 c(또는 ~/continue)를 받을 때마다
        # 전체 사이클을 다시 실행한다. 마커는 매 사이클 다시 인식한다. 실험 반복용.
        self.declare_parameter('loop_mode', False)
        self.declare_parameter('batch_config_file', '')
        self.declare_parameter('continuous_feed', False, ParameterDescriptor(read_only=True))
        self.declare_parameter('marker_topic_out', '/pnp_targets')

    def _read_parameters(self):
        catalog_path = self.get_parameter('object_models_file').value
        self._object_catalog = ObjectCatalog.load(catalog_path) if catalog_path else None
        self.object_table_z = float(self.get_parameter('object_table_z').value)
        get = self.get_parameter
        self.planning_group = get('planning_group').value
        self.ee_link = get('ee_link').value
        self.base_frame = get('base_frame').value
        self.arm_joints = list(get('arm_joints').value)

        self.planning_time = get('planning_time').value
        self.planning_attempts = get('planning_attempts').value
        self.velocity_scaling = get('velocity_scaling').value
        self.acceleration_scaling = get('acceleration_scaling').value
        self.cartesian_max_step = get('cartesian_max_step').value
        self.cartesian_min_fraction = get('cartesian_min_fraction').value

        self.zero_pose = list(get('zero_pose').value)
        self.camera_ready_pose = list(get('camera_ready_pose').value)

        self.aruco_markers_topic = get('aruco_markers_topic').value
        self.aruco_poses_topic = get('aruco_poses_topic').value
        self.use_pose_array_fallback = get('use_pose_array_fallback').value
        self.pick_marker_id = get('pick_marker_id').value
        self.place_marker_id = get('place_marker_id').value
        self.marker_wait_timeout = get('marker_wait_timeout').value
        self.target_max_age = float(get('target_max_age_s').value)
        self.guard_real_commands = bool(get('guard_real_commands').value)
        self.feedback_max_age = float(get('feedback_max_age_s').value)
        self.feedback_joint_tolerance = float(get('feedback_joint_tolerance').value)
        self.feedback_arrival_timeout = float(get('feedback_arrival_timeout').value)
        if not all(math.isfinite(v) and v > 0 for v in (
                self.target_max_age, self.feedback_max_age,
                self.feedback_joint_tolerance, self.feedback_arrival_timeout)):
            raise ValueError('유효 시각/도달 오차 파라미터는 양수여야 합니다.')
        self.tf_timeout = get('tf_timeout').value
        self.marker_stable_samples = get('marker_stable_samples').value
        self.marker_stable_window = get('marker_stable_window').value
        self.marker_stable_spread = get('marker_stable_spread').value
        self.marker_stable_yaw = math.radians(get('marker_stable_yaw_deg').value)
        self.approach_offset = get('approach_offset').value
        self.grasp_offset = get('grasp_offset').value
        self.place_offset = get('place_offset').value
        self.place_approach_offset = get('place_approach_offset').value
        self.place_match_pick_tilt = get('place_match_pick_tilt').value
        self.place_alignment_tolerance = math.radians(
            get('place_alignment_tolerance_deg').value)
        if not 0 < self.place_alignment_tolerance < math.pi/4:
            raise ValueError('place_alignment_tolerance_deg는 0도 초과 45도 미만이어야 합니다.')
        self.place_tilt_azimuths = [math.radians(v) for v in get('place_tilt_azimuths_deg').value]
        self.grasp_ik_mode = get('grasp_ik_mode').value
        self.grasp_tilt_tolerance = get('grasp_tilt_tolerance').value
        self.free_grasp_yaw = get('free_grasp_yaw').value
        self.grasp_yaw_candidates = [
            math.radians(v) for v in get('grasp_yaw_candidates_deg').value]
        self.grasp_tilt_candidates = [
            math.radians(v) for v in get('grasp_tilt_candidates_deg').value]
        self.grasp_tilt_azimuths = [
            math.radians(v) for v in get('grasp_tilt_azimuths_deg').value]
        self.grasp_ik_check_timeout = get('grasp_ik_check_timeout').value

        self.arm_controller_action = get('arm_controller_action').value
        self.controller_wait_timeout = get('controller_wait_timeout').value

        self.use_real_gripper = get('use_real_gripper').value
        self.gripper_action = get('gripper_action').value
        self.gripper_joint = get('gripper_joint').value
        self.gripper_open_position = get('gripper_open_position').value
        self.gripper_close_position = get('gripper_close_position').value
        self.gripper_settle_time = get('gripper_settle_time').value

        self.preview_only = get('preview_only').value
        self.preview_step_delay = get('preview_step_delay').value

        self.add_table = get('add_table').value
        self.table_size = list(get('table_size').value)
        self.table_top_z = get('table_top_z').value
        self.table_center_xy = list(get('table_center_xy').value)

        self.min_reach = get('min_reach').value
        self.max_reach = get('max_reach').value

        self.control_enable_service = get('control_enable_service').value
        self.emergency_stop_service = get('emergency_stop_service').value
        self.step_confirm = get('step_confirm').value
        self.step_confirm_timeout = get('step_confirm_timeout').value

        self.auto_start = get('auto_start').value
        self.loop_mode = get('loop_mode').value
        self.marker_topic_out = get('marker_topic_out').value
        batch_path = get('batch_config_file').value
        self._batch_workspace = BatchWorkspace.load(batch_path) if batch_path else None
        self.continuous_feed = get('continuous_feed').value
        if self.continuous_feed and not (
                self._batch_workspace and self._batch_workspace.operator_feed
                and not self._batch_workspace.sweep):
            raise ValueError('continuous_feed는 기존 feed 작업 영역에서만 사용할 수 있습니다.')
        if self._batch_workspace is not None and self._batch_workspace.feed_drop:
            if not self.continuous_feed or self._object_catalog is None:
                raise ValueError('feed_drop requires continuous_feed and metric object observations')
        if self._batch_workspace is not None and self._batch_workspace.sweep:
            if self._object_catalog is None:
                raise ValueError('sweep requires atomic metric object observations')
            if not (0 < self.velocity_scaling <= .50 and 0 < self.acceleration_scaling <= .25):
                raise ValueError('sweep speed limits: velocity <= .50, acceleration <= .25')
        if not math.isfinite(self.object_table_z):
            raise ValueError('object_table_z must be finite')
        if self._object_catalog is not None:
            if not self.place_match_pick_tilt or self.grasp_ik_mode != 'candidates':
                raise ValueError('Metric object models require grasp candidates and validated place orientation')
            if self._batch_workspace:
                self._object_catalog.validate_workspace(self._batch_workspace)
        self._batch_completed = 0
        if self._batch_workspace is not None:
            if (self.auto_start or self.step_confirm or not self.loop_mode
                    or self.preview_only or not self.guard_real_commands
                    or not self.use_real_gripper or self.base_frame != 'base_link'):
                raise ValueError('연속 이송은 run_robot.sh batch의 실기/시작 승인 설정이 필요합니다.')
            for position in self._batch_workspace.places:
                if not self._target_is_sane(PLACE, position):
                    raise ValueError(f'놓기 자리가 작업 범위를 벗어났습니다: {position}')
            # An interrupted experiment keeps services alive but run_restartable
            # requires an operator-confirmed home/reset before any new cycle.
            self._batch_progress = BatchProgress(
                batch_path, self._batch_workspace, allow_interrupted=True)
            self._batch_completed = self._batch_progress.completed
        self.declare_parameter(
            'batch_operator_feed', bool(self._batch_workspace and self._batch_workspace.operator_feed
                                        and not self.continuous_feed),
            ParameterDescriptor(read_only=True))

        for pre_label, pre, off_label, off in (
                ('approach_offset', self.approach_offset,
                 'grasp_offset', self.grasp_offset),
                ('place_approach_offset', self.place_approach_offset,
                 'place_offset', self.place_offset)):
            if pre <= off:
                raise ValueError(
                    f'{pre_label}({pre})는 {off_label}({off})보다 커야 합니다. '
                    '그렇지 않으면 접근 지점이 파지/놓기 지점보다 아래가 되어 '
                    '직선 하강이 성립하지 않습니다.')

        if self.grasp_ik_mode not in ('candidates', 'tolerance', 'position_only'):
            raise ValueError(
                f"grasp_ik_mode 는 candidates / tolerance / position_only 중 "
                f"하나여야 합니다 (받은 값: {self.grasp_ik_mode}).")

        for name, values in (('zero_pose', self.zero_pose),
                             ('camera_ready_pose', self.camera_ready_pose)):
            if len(values) != len(self.arm_joints):
                raise ValueError(
                    f"파라미터 '{name}' 의 길이({len(values)})가 "
                    f'arm_joints 개수({len(self.arm_joints)})와 다릅니다.')

    # ==================================================================
    # ArUco 수신
    # ==================================================================
    def _on_cuboid_target(self, msg):
        if self._frozen or self._object_catalog is None:
            return
        try:
            stamp,frame,p,q,metadata=self._object_catalog.decode(msg.data)
            header=Header(stamp=Time(nanoseconds=stamp).to_msg(),frame_id=frame)
            self._store_target(PICK,_make_pose(p,q),header,metadata=metadata)
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            self.get_logger().warn(f'박스 종류/자세 메시지를 거부합니다: {exc}',throttle_duration_sec=3.)

    def _on_aruco_markers(self, msg):
        """ID 가 실려오는 주 경로 (ros2_aruco 의 ArucoMarkers)."""
        if self._frozen:
            return
        ids = list(msg.marker_ids)
        # 무엇이 검출되는지 5초마다 한 줄. pick/place 가 겹쳐 뜨면 여기서 원인이
        # 보인다 (예: [0, 0] = ID 0 마커 두 개, [0] = place 마커 미검출).
        self.get_logger().info(
            f'검출 마커 ID: {ids} '
            f'(pick={self.pick_marker_id}, place={self.place_marker_id})',
            throttle_duration_sec=5.0)
        if ids.count(self.pick_marker_id) > 1 or ids.count(self.place_marker_id) > 1:
            self.get_logger().warn(
                f'같은 ID 가 여러 번 검출됩니다: {ids}. 물리 마커 ID 를 확인하세요.',
                throttle_duration_sec=5.0)
        for marker_id, pose in zip(ids, msg.poses):
            if marker_id == self.pick_marker_id:
                self._store_target(PICK, pose, msg.header)
            elif marker_id == self.place_marker_id:
                self._store_target(PLACE, pose, msg.header)

    def _on_aruco_poses(self, msg):
        """
        ID 가 없는 보조 경로 (PoseArray).

        로컬/시뮬레이션 테스트에서 `ros2 topic pub` 으로 타겟을 주입할 때 쓴다.
        인덱스 0 을 Pick, 인덱스 1 을 Place 로 해석한다.
        ArucoMarkers 가 이미 같은 타겟을 채웠다면 그쪽이 우선이다.
        """
        if self._frozen:
            return
        for index, key in enumerate((PICK, PLACE)):
            if index < len(msg.poses):
                self._store_target(key, msg.poses[index], msg.header)

    def _target_is_sane(self, key, position):
        """
        변환된 타겟이 물리적으로 말이 되는지 확인한다.

        검출 오류나 잘못된 TF 로 터무니없는 좌표가 들어오면, 그대로 플래닝에
        넘기지 않고 여기서 걸러 명확한 이유를 남긴다.
        """
        if not all(math.isfinite(v) for v in position):
            self.get_logger().error(
                f'{key} 타겟에 유한하지 않은 값이 있습니다: {position}')
            return False

        distance = math.sqrt(sum(v * v for v in position))
        if distance > self.max_reach:
            self.get_logger().warn(
                f'{key} 타겟이 작업 반경 밖입니다 ({distance:.3f} m > '
                f'{self.max_reach:.3f} m). 무시합니다.',
                throttle_duration_sec=5.0)
            return False
        if distance < self.min_reach:
            self.get_logger().warn(
                f'{key} 타겟이 로봇에 너무 가깝습니다 ({distance:.3f} m < '
                f'{self.min_reach:.3f} m). 무시합니다.',
                throttle_duration_sec=5.0)
            return False

        # 접촉 지점이 테이블 위여야 한다. Pick 은 마커 아래로 파고들고
        # (grasp_offset<0), Place 는 마커 위에서 연다(place_offset>0).
        contact_offset = self.place_offset if key == PLACE else self.grasp_offset
        lowest = position[2] + min(0.0, contact_offset)
        if lowest < self.table_top_z:
            self.get_logger().warn(
                f'{key} 타겟의 접촉 지점이 테이블 상판 아래입니다 '
                f'(z={lowest:.3f} < {self.table_top_z:.3f}). 무시합니다.',
                throttle_duration_sec=5.0)
            return False
        return True

    @staticmethod
    def _stable_estimate(samples, spread, yaw_spread):
        """
        samples: [(t, position, orientation)].

        위치들의 축별 중앙값 주변 spread [m] 이내 + yaw 들이 원형평균에서
        yaw_spread [rad] 이내이고 전체 회전각도 이 한도 이내면
        (position, orientation, 위치편차) 를, 아니면
        (None, None, 위치편차) 를 돌려준다. orientation 은 원형평균 yaw 에 가장
        가까운 샘플의 것 (쿼터니언 평균의 함정 회피, ArUco 뒤집힘 필터).
        """
        positions = [s[1] for s in samples]
        med = tuple(statistics.median(p[i] for p in positions) for i in range(3))
        worst = max(math.dist(p, med) for p in positions)
        if worst > spread:
            return None, None, worst

        yaws = [yaw_of(s[2]) for s in samples]
        mean_yaw = math.atan2(sum(math.sin(y) for y in yaws),
                              sum(math.cos(y) for y in yaws))

        def dev(y):
            return abs((y - mean_yaw + math.pi) % (2.0 * math.pi) - math.pi)

        if max(dev(y) for y in yaws) > yaw_spread:
            return None, None, worst
        nearest = min(samples, key=lambda s: dev(yaw_of(s[2])))
        # yaw가 같아도 상단면의 roll/pitch가 뒤집히면 거부한다. q와 -q는 동일 회전.
        ref = nearest[2]
        for _, _, q in samples:
            norm = math.sqrt(sum(v*v for v in q) * sum(v*v for v in ref))
            if norm < 1e-12:
                return None, None, worst
            angle = 2*math.acos(min(1.0, abs(sum(a*b for a,b in zip(q,ref)))/norm))
            if angle > yaw_spread:
                return None, None, worst
        return med, nearest[2], worst

    def _store_target(self, key, pose, header, metadata=None):
        """마커 포즈를 base_frame 으로 변환해 최근 샘플에 쌓고, 안정되면 확정한다."""
        if key == PICK and getattr(self, '_object_catalog', None) is not None and metadata is None:
            return  # A legacy marker must never bypass model/catalog validation.
        workspace = getattr(self, '_batch_workspace', None)
        if workspace is not None and key == PLACE:
            return  # 반복 모드에서는 컨트롤러가 빈 놓기 자리를 지정한다.
        stamp_ns = Time.from_msg(header.stamp).nanoseconds
        now_ns = self.get_clock().now().nanoseconds
        age = (now_ns-stamp_ns)/1e9
        q = pose.orientation
        values = (q.x, q.y, q.z, q.w)
        if (not 0 <= age <= self.target_max_age or stamp_ns <= self._target_not_before_ns
                or not all(math.isfinite(v) for v in values)
                or sum(v*v for v in values) < 1e-12):
            return
        converted = self._to_base_frame(pose, header)
        if converted is None:
            return
        position, orientation = converted
        if workspace is not None and not workspace.allows_pick(position):
            return
        if not self._target_is_sane(key, position):
            return

        now = time.monotonic()
        with self._targets_lock:
            if (self._frozen or stamp_ns <= self._target_not_before_ns
                    or stamp_ns <= self._last_target_stamp.get(key, 0)):
                return
            if (self.get_clock().now().nanoseconds-stamp_ns)/1e9 > self.target_max_age:
                return
            self._last_target_stamp[key] = stamp_ns
            buf = self._target_samples[key]
            if metadata is not None:
                identity=(metadata['model_id'],metadata['track_id'])
                if identity != getattr(self,'_sample_object_identity',None):
                    buf.clear()
                    if hasattr(self, '_sweep_scene_samples'):
                        self._sweep_scene_samples.clear()
                    self._targets.pop(key,None)
                    self._target_stamps.pop(key,None)
                    self._targets_ready.clear()
                    self._sample_object_identity=identity
            sweep_scene = None
            if workspace is not None and workspace.drop:
                sweep_scene = self._accept_sweep_scene(metadata)
                if sweep_scene is None:
                    buf.clear()
                    self._targets.pop(key,None)
                    self._target_stamps.pop(key,None)
                    self._targets_ready.clear()
                    return
            buf.append((now, position, orientation))
            while buf and now - buf[0][0] > self.marker_stable_window:
                buf.popleft()
            if len(buf) < self.marker_stable_samples or sweep_scene is False:
                return
            med, quat, worst = self._stable_estimate(
                list(buf), self.marker_stable_spread, self.marker_stable_yaw)
            if med is None:
                self._targets.pop(key, None)
                self._target_stamps.pop(key, None)
                self._targets_ready.clear()
                unstable_mm = worst * 1000.0
                self.get_logger().warn(
                    f'{key} 마커가 불안정합니다 (위치편차 {unstable_mm:.0f}mm 또는 '
                    f'회전 편차 > {math.degrees(self.marker_stable_yaw):.0f}°). 대기.',
                    throttle_duration_sec=3.0)
                return
            first_time = key not in self._targets
            self._targets[key] = {'position': med, 'orientation': quat}
            if metadata is not None:
                self._targets[key].update(metadata)
            if sweep_scene:
                self._targets[key]['sweep_scene'] = sweep_scene
            self._target_stamps[key] = stamp_ns
            have_both = PICK in self._targets and PLACE in self._targets
            n_samples = len(buf)

        if first_time:
            self.get_logger().info(
                f'{key} 마커 확정 — {self.base_frame} 기준 '
                f'({med[0]:.3f}, {med[1]:.3f}, {med[2]:.3f}) '
                f'[{n_samples} 샘플, 편차 {worst * 1000.0:.0f}mm]')
        if have_both:
            with self._targets_lock:
                self._expire_targets_locked()

    def _to_base_frame(self, pose, header):
        """
        포즈를 base_frame 으로 변환한다.

        명세 5-1: TF 조회에 외삽 타임아웃을 적용한다. 마커 타임스탬프 시점의 변환이
        아직 버퍼에 없을 수 있으므로, tf_timeout 만큼 기다렸다가 그래도 없으면 포기한다.
        """
        source_frame = header.frame_id
        if not source_frame:
            self.get_logger().warn('frame_id 가 비어 있는 마커 메시지를 무시합니다.')
            return None

        try:
            # tf2_ros 는 rclpy.time.Time 을 요구한다. 메시지의 builtin_interfaces/Time 을
            # 그대로 넘기면 .nanoseconds 속성이 없어 실패한다.
            # stamp 가 0 이면 tf2 는 "가장 최근 변환"으로 해석한다.
            tf = self._tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=self.tf_timeout),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'{source_frame} -> {self.base_frame} TF 조회 실패: {exc}',
                throttle_duration_sec=5.0)
            return None

        t = tf.transform.translation
        r = tf.transform.rotation
        return transform_pose(
            (t.x, t.y, t.z),
            (r.x, r.y, r.z, r.w),
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w),
        )

    # ==================================================================
    # 타겟 -> 포즈 계산
    # ==================================================================
    def _target_poses(self, key, contact_offset=None, pre_offset=None):
        """
        (접근 포즈, 파지/놓기 포즈) 를 돌려준다. 타겟이 없으면 (None, None).

        contact_offset: 접촉 지점 offset (Pick=grasp_offset, Place=place_offset).
        pre_offset    : 접근 지점 offset (Pick=approach_offset,
                        Place=place_approach_offset). None 이면 각각 기본값.
        """
        with self._targets_lock:
            target = self._targets.get(key)
        if target is None:
            return None, None

        if contact_offset is None:
            contact_offset = self.grasp_offset
        if pre_offset is None:
            pre_offset = self.approach_offset

        position = target['position']
        marker_q = target['orientation']

        if key == PLACE and getattr(getattr(self, '_batch_workspace', None), 'feed_drop', False):
            # Position-only release: show the actual fixed TCP target, without
            # the legacy approach/contact offsets or an implied descent.
            return _make_pose(position, (0., 0., 0., 1.)), _make_pose(position, (0., 0., 0., 1.))

        if self.grasp_ik_mode == 'position_only':
            # 자세 무시: 마커 중앙 기준 월드 수직으로만 offset. 자세는 자리표시자.
            pre_position = (position[0], position[1], position[2] + pre_offset)
            grasp_position = (position[0], position[1], position[2] + contact_offset)
            grasp_q = (0.0, 0.0, 0.0, 1.0)
        else:
            grasp_q = grasp_orientation(marker_q)
            # 두 지점 모두 마커 법선 위에 놓인다. contact_offset 이 0 이면 접촉
            # 지점은 마커 중심 그대로이고, 양수면 덜 내려가고 음수면 더 파고든다.
            grasp_position = approach_position(position, marker_q, contact_offset)
            pre_position = approach_position(position, marker_q, pre_offset)

        return (_make_pose(pre_position, grasp_q), _make_pose(grasp_position, grasp_q))

    @staticmethod
    def _poses_along_axis(marker_pos, q, pre_offset, contact_offset):
        """
        자세 q 의 접근축(+Z, 물체 쪽)을 따라 놓인 (Ready, Grasp) 포즈.

            point(off) = marker_pos - off * approach_axis

        off=pre_offset(>0) 이면 물체에서 물러난 지점, off=contact_offset(음수)면
        물체 안으로 파고든 지점. 둘 다 자세는 q.
        """
        axis = quat_rotate_vector(q, (0.0, 0.0, 1.0))
        pre = tuple(marker_pos[i] - pre_offset * axis[i] for i in range(3))
        grasp = tuple(marker_pos[i] - contact_offset * axis[i] for i in range(3))
        return _make_pose(pre, q), _make_pose(grasp, q)

    def _choose_grasp_candidate(self, marker_pos, marker_q, pre_offset, contact_offset):
        """
        파지 자세 후보를 편차 작은 순으로 /compute_ik 로 걸러, Ready 지점에서 IK
        해가 있는 첫 자세를 돌려준다. 전부 실패하면 None.
        """
        # ArUco 평면 양의성으로 마커 법선이 아래로 뒤집혀 들어올 수 있다.
        # 테이블 위 마커의 법선은 위여야 하므로, 아래면 180° 되돌린다.
        if quat_rotate_vector(marker_q, (0.0, 0.0, 1.0))[2] < 0.0:
            marker_q = quat_multiply(marker_q, quat_from_rpy(math.pi, 0.0, 0.0))
            self.get_logger().warn('Pick 마커 자세가 뒤집혀 검출됨 — 180° 보정.')
        nominal_axis = quat_rotate_vector(grasp_orientation(marker_q), (0.0, 0.0, 1.0))
        cands = grasp_candidates(
            marker_q, self.grasp_yaw_candidates,
            self.grasp_tilt_candidates, self.grasp_tilt_azimuths)
        for i, q in enumerate(cands):
            with self._targets_lock:
                metric_pick = self._targets.get(PICK, {})
            if 'dims' in metric_pick and grip_width(metric_pick,q)+.004>self.gripper_open_position:
                continue
            # 마커 자세가 뒤집혀 들어오면 접근축이 위를 향할 수 있다 — 기각.
            if not self._axis_points_down(q):
                continue
            pre_pose, grasp_pose = self._poses_along_axis(
                marker_pos, q, pre_offset, contact_offset)
            # 접근 지점은 충돌까지, 접촉 지점은 도달 가능성만 확인한다 (접촉점은
            # 물체가 채우는 공간이라 테이블 박스와 겹치는 게 정상).
            if (self.moveit.compute_ik(
                    pre_pose, ik_timeout=self.grasp_ik_check_timeout)
                    and self.moveit.compute_ik(
                        grasp_pose, ik_timeout=self.grasp_ik_check_timeout,
                        avoid_collisions=False)):
                # 같은 박스를 90도 다르게 집으면, 정렬된 놓기 자세의 도달 가능성도
                # 달라진다. 집기 전에 현재 목적지의 정렬/개방 IK까지 확인한다.
                if getattr(getattr(self, '_batch_workspace', None), 'sweep', False):
                    if not self._plan_sweep_pick(q, pre_pose, grasp_pose):
                        continue
                elif self.place_match_pick_tilt and not getattr(
                        getattr(self, '_batch_workspace', None), 'feed_drop', False):
                    with self._targets_lock:
                        place = self._targets.get(PLACE)
                    if place is None or self._choose_place_orientation(
                            place['position'], place['orientation'],
                            self.place_approach_offset, self.place_offset,
                            pick_grasp_q=q, announce=False) is None:
                        continue
                axis = quat_rotate_vector(q, (0.0, 0.0, 1.0))
                dot = max(-1.0, min(1.0, sum(axis[j] * nominal_axis[j]
                                             for j in range(3))))
                self.get_logger().info(
                    f'파지 자세 후보 {i + 1}/{len(cands)} 채택 '
                    f'(법선에서 {math.degrees(math.acos(dot)):.0f}° 기움)')
                return q
        self.get_logger().warn(
            f'{len(cands)}개 파지 자세 후보에서 집기/놓기 도달 검사를 통과하지 못했습니다.')
        return None

    @staticmethod
    def _axis_points_down(q, min_down=0.1):
        """접근축(+Z)이 아래를 min_down 이상 향하면 True (파지 가능한 자세)."""
        return quat_rotate_vector(q, (0.0, 0.0, 1.0))[2] < -min_down

    def _metric_place_poses(self, pick, q, marker_pos, pre_offset, contact_offset, pick_q=None):
        if pick_q is None:pick_q=self._pick_grasp_q
        workspace=getattr(self,'_batch_workspace',None)
        table_z=workspace.table_z if workspace else self.object_table_z
        ready,contact=place_positions(pick,pick_q,q,marker_pos[:2],table_z,self.grasp_offset,
                                      pre_offset-contact_offset,getattr(self,'_pick_contact_position',None))
        return _make_pose(ready,q),_make_pose(contact,q)

    def _choose_place_orientation(self, marker_pos, marker_q, pre_offset, contact_offset,
                                  pick_grasp_q=None, plan_release=False, announce=True):
        """집은 상대 자세를 유지하면서 박스 긴 변을 place X축과 나란히 놓는다."""
        with self._targets_lock:
            pick_target = self._targets.get(PICK)
        pick_q = self._pick_grasp_q if pick_grasp_q is None else pick_grasp_q
        if pick_target is None or pick_q is None:
            return None
        if plan_release:
            self._place_ready_plan = None
        metric=pick_target.get('place_mode')=='geometry'
        cands = (level_place_candidates(pick_target,pick_q,marker_q) if metric else
                 aligned_place_candidates(pick_target['orientation'], pick_q, marker_q,
                                          self.grasp_tilt_candidates, self.place_tilt_azimuths))
        for q in cands:
            if self.aborted:
                return None
            if not self._axis_points_down(q):
                continue
            if metric:
                pre_pose,grasp_pose=self._metric_place_poses(pick_target,q,marker_pos,pre_offset,contact_offset,pick_q)
            else:
                pre_pose, grasp_pose = self._poses_along_axis(marker_pos, q, pre_offset, contact_offset)
            if not (self.moveit.compute_ik(
                        pre_pose, ik_timeout=self.grasp_ik_check_timeout)
                    and self.moveit.compute_ik(
                        grasp_pose, ik_timeout=self.grasp_ik_check_timeout,
                        avoid_collisions=False)):
                continue
            if not self._place_open_ik_clear(pre_pose, grasp_pose):
                continue
            if plan_release:
                trajectory = self._plan_aligned_place_release(pre_pose, grasp_pose)
                if trajectory is None:
                    continue
                self._place_ready_plan = trajectory
            if not announce:
                return q
            if metric:
                self.get_logger().info(f"Place: {pick_target['model_id']} 실제 치수와 파지 상대 변환으로 높이 계산, 정사각형 면 수평 배치.")
                return q
            box_q = held_box_orientation(q, pick_q, pick_target['orientation'])
            normal = quat_rotate_vector(box_q, (0., 0., 1.))
            target_normal = quat_rotate_vector(marker_q, (0., 0., 1.))
            tilt = math.degrees(math.acos(max(-1., min(1., sum(
                a*b for a,b in zip(normal, target_normal))))))
            self.get_logger().info(
                f'Place 자세: 박스 긴 변 정렬 (목표 {math.degrees(yaw_of(marker_q)):.0f}° '
                f'/ 180° 대칭 허용, 박스 기울기 {tilt:.0f}°).')
            return q
        if announce:
            self.get_logger().error(
                '박스 배치 방향과 그리퍼 개방 검사를 만족하는 놓기 자세가 없습니다. '
                '모델의 대칭 범위를 벗어나거나 수평 조건을 완화하지 않습니다.' if metric else
                '박스 긴 변을 정렬하고 그리퍼를 열 수 있는 놓기 자세가 없습니다. '
                '90° 회전이나 방향 자유 자세로 대체하지 않습니다.')
        return None

    def _plan_aligned_place_release(self, pre, contact):
        """이동 전에 실제 접근 궤적 끝점에서 방향·개방·후퇴를 계획만으로 확인."""
        state = self._release_current_state()
        if state is None or self.aborted:
            return None
        ok, path = self.moveit.move_to_pose(pre, start_state=state, plan_only=True)
        if not ok or path is None or self.aborted:
            return None
        end = trajectory_end_state(path, state)
        if end is None:
            return None
        ready = self.moveit.compute_fk(list(end.joint_state.name), list(end.joint_state.position))
        if ready is None:
            return None
        o = ready.orientation
        q = (o.x, o.y, o.z, o.w)
        axis = quat_rotate_vector(q, (0., 0., 1.))
        xyz = lambda p: (p.position.x, p.position.y, p.position.z)
        distance = math.dist(xyz(pre), xyz(contact))
        actual_contact = _make_pose(tuple(v+distance*a for v,a in zip(xyz(ready), axis)), q)
        if (not self._check_place_alignment(actual_contact)
                or not self._plan_release_clearance(end, ready, actual_contact, check_camera=True)):
            return None
        return path if not self.aborted else None

    def _check_place_alignment(self, pose=None):
        """명령한 그리퍼 yaw가 아니라, 파지 상대 회전을 반영한 박스 긴 변 검사."""
        if not self.place_match_pick_tilt:
            return True
        with self._targets_lock:
            pick = self._targets.get(PICK)
            place = self._targets.get(PLACE)
        if pose is None:
            pose = self._achieved_ee_pose()
        if pose is None or pick is None or place is None or self._pick_grasp_q is None:
            self.get_logger().error('놓기 방향을 확인할 파지 자세/관절 피드백이 없습니다.')
            return False
        o = pose.orientation
        if pick.get('place_mode')=='geometry':
            workspace=getattr(self,'_batch_workspace',None)
            valid=check_level_release(pick,self._pick_grasp_q,(o.x,o.y,o.z,o.w),
                (pose.position.x,pose.position.y,pose.position.z),place['orientation'],
                workspace.table_z if workspace else self.object_table_z,self.grasp_offset,
                self.place_alignment_tolerance,getattr(self,'_pick_contact_position',None))
            if not valid:self.get_logger().error('놓기 박스의 수평/방향/예측 바닥 높이 검사가 실패했습니다.')
            return valid
        box_q = held_box_orientation(
            (o.x, o.y, o.z, o.w), self._pick_grasp_q, pick['orientation'])
        error = long_axis_error(box_q, place['orientation'])
        if not math.isfinite(error) or error > self.place_alignment_tolerance:
            self.get_logger().error(
                f'놓기 박스 긴 변 방향 오차 {math.degrees(error):.1f}° > '
                f'{math.degrees(self.place_alignment_tolerance):.1f}°. 그리퍼를 열지 않습니다.')
            return False
        self.get_logger().info(f'놓기 긴 변 방향 확인: 예측 오차 {math.degrees(error):.1f}°.')
        return True

    def _approach_ready(self, label, key, contact_offset, pre_offset):
        """
        grasp_ik_mode 에 따라 Pick/Place-Ready 로 이동하고 (Ready, Grasp) 포즈를
        돌려준다. 실패 시 (None, None). Pick 성공 시 실제 도달 자세를
        self._pick_grasp_q 에 저장한다 (Place 미러링용).

        Place 는 place_match_pick_tilt 면 박스 긴 변 정렬을 보존하는 자세만 쓴다.
        해당 자세를 찾지 못하면 실패하며, 일반 파지 후보로 대체하지 않는다.

          candidates    : 이산 후보를 IK 로 걸러 첫 통과 자세를 정확히 실행.
          tolerance     : tilt tolerance 박스.
          position_only : 자세 제약 없음 (월드 수직 접근).

        contact_offset : 접촉 지점 offset (Pick=grasp_offset, Place=place_offset).
        pre_offset     : 접근 지점 offset (Pick=approach_offset,
                         Place=place_approach_offset).
        """
        with self._targets_lock:
            target = self._targets.get(key)
        if target is None:
            self.get_logger().error(f'{key} 타겟이 없습니다.')
            return None, None
        marker_pos, marker_q = target['position'], target['orientation']

        result = self._approach_ready_impl(
            label, key, marker_pos, marker_q, contact_offset, pre_offset)

        if key == PICK and result[1] is not None:
            o = result[1].orientation
            self._pick_grasp_q = (o.x, o.y, o.z, o.w)
            p = result[1].position
            self._pick_contact_position = (p.x,p.y,p.z)
        return result

    def _approach_ready_impl(self, label, key, marker_pos, marker_q,
                             contact_offset, pre_offset):
        if key == PLACE and self.place_match_pick_tilt:
            check_plan = self.guard_real_commands or self.preview_only
            q = self._choose_place_orientation(
                marker_pos, marker_q, pre_offset, contact_offset, plan_release=check_plan)
            if q is not None:
                pick=self._targets.get(PICK,{})
                if pick.get('place_mode')=='geometry':
                    pre,grasp=self._metric_place_poses(pick,q,marker_pos,pre_offset,contact_offset)
                else:
                    pre, grasp = self._poses_along_axis(marker_pos, q, pre_offset, contact_offset)
                if check_plan:
                    # 검사한 접근 궤적을 그대로 실행해 IK 분기가 바뀌지 않게 한다.
                    path = self._place_ready_plan
                    if self.aborted:
                        return None, None
                    ok = self.preview_only or self.moveit._execute_trajectory(path, 60.)
                    self._last_pose_trajectory = path if ok else None
                    if not self._after_motion(label, ok, path):
                        return None, None
                elif not self._goto_pose(label, pre):
                    # 실제 피드백 게이트가 없는 기존 dummy 시뮬레이션 경로.
                    return None, None
                ready, contact = self._align_grasp_to_achieved(pre, grasp)
                if not self._check_place_alignment(contact):
                    return None, None
                return ready, contact
            return None, None

        mode = self.grasp_ik_mode
        if mode == 'candidates':
            planning_started = time.monotonic()
            q = self._choose_grasp_candidate(
                marker_pos, marker_q, pre_offset, contact_offset)
            if q is not None:
                pre, grasp = self._poses_along_axis(
                    marker_pos, q, pre_offset, contact_offset)
                if getattr(getattr(self, '_batch_workspace', None), 'sweep', False) and key == PICK:
                    self._sweep_details['pick_plan'] = time.monotonic()-planning_started
                    path = self._sweep_pick_path
                    if self.aborted:
                        return None,None
                    execution_started = time.monotonic()
                    ok = self.preview_only or self.moveit._execute_trajectory(path,60.)
                    if not self._after_motion(label,ok,path):
                        return None,None
                    self._sweep_details['pick_execute'] = time.monotonic()-execution_started
                elif not self._goto_pose(label, pre):
                    return None, None
                # feed/grasp retain the original constant-distance descent.
                # The experimental contact-plane correction belongs to sweep.
                if key == PICK and getattr(getattr(self, '_batch_workspace', None), 'sweep', False):
                    return self._align_grasp_to_achieved(pre, grasp, anchor_contact=True)
                return self._align_grasp_to_achieved(pre, grasp)
            if key == PICK and self.place_match_pick_tilt:
                if getattr(getattr(self, '_batch_workspace', None), 'sweep', False):
                    self.get_logger().error('접근·짧은 하강 경로를 만족하는 집기 자세가 없습니다.')
                elif getattr(getattr(self, '_batch_workspace', None), 'feed_drop', False):
                    self.get_logger().error('접근·파지 위치에 도달 가능한 집기 자세가 없습니다.')
                else:
                    self.get_logger().error('집기와 긴 변 정렬 놓기가 모두 가능한 파지 자세가 없습니다.')
                return None, None
            mode = 'tolerance'

        pre, grasp = self._target_poses(key, contact_offset, pre_offset)
        ok = self._goto_pose(
            label, pre,
            free_yaw=self.free_grasp_yaw,
            tilt_tolerance=(self.grasp_tilt_tolerance if mode == 'tolerance' else None),
            position_only=(mode == 'position_only'))
        if not ok:
            return None, None
        return self._align_grasp_to_achieved(pre, grasp)

    # ==================================================================
    # RViz 시각화
    # ==================================================================
    def _publish_target_markers(self):
        """확정된 타겟을 RViz Marker 로 반복 발행한다 (구독 시점과 무관하게 보이도록)."""
        with self._targets_lock:
            keys = list(self._targets.keys())
        if not keys:
            return

        colors = {PICK: (0.1, 0.9, 0.2), PLACE: (0.2, 0.5, 1.0)}
        base_id = {PICK: 0, PLACE: 10}

        for key in keys:
            pre_pose, grasp_pose = self._target_poses(key)
            if grasp_pose is None:
                continue
            r, g, b = colors[key]

            sphere = Marker()
            sphere.header.frame_id = self.base_frame
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = 'pnp_targets'
            sphere.id = base_id[key]
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = grasp_pose
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.03
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 0.9
            self._marker_pub.publish(sphere)

            # 접근 방향(마커 법선)을 화살표로: 접근 지점 -> 파지 지점
            arrow = Marker()
            arrow.header = sphere.header
            arrow.ns = 'pnp_targets'
            arrow.id = base_id[key] + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.points = [
                Point(x=pre_pose.position.x, y=pre_pose.position.y, z=pre_pose.position.z),
                Point(x=grasp_pose.position.x, y=grasp_pose.position.y,
                      z=grasp_pose.position.z),
            ]
            arrow.scale.x = 0.006     # 자루 지름
            arrow.scale.y = 0.014     # 머리 지름
            arrow.scale.z = 0.02      # 머리 길이
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, 0.9
            if key == PLACE and getattr(getattr(self, '_batch_workspace', None), 'feed_drop', False):
                arrow.action = Marker.DELETE  # XYZ release has no descent direction.
            self._marker_pub.publish(arrow)

            text = Marker()
            text.header = sphere.header
            text.ns = 'pnp_targets'
            text.id = base_id[key] + 2
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = _make_pose(
                (pre_pose.position.x, pre_pose.position.y, pre_pose.position.z + 0.04),
                (0.0, 0.0, 0.0, 1.0))
            text.text = key.upper()
            text.scale.z = 0.03
            text.color.r, text.color.g, text.color.b, text.color.a = r, g, b, 1.0
            self._marker_pub.publish(text)

    # ==================================================================
    # 그리퍼 — 명세 5-3
    # ==================================================================
    def open_gripper(self):
        """그리퍼를 연다."""
        return self._command_gripper(self.gripper_open_position, 'OPEN')

    def close_gripper(self):
        """그리퍼를 닫는다."""
        return self._command_gripper(self.gripper_close_position, 'CLOSE')

    def _command_gripper(self, position, label):
        """
        use_real_gripper 파라미터에 따라 갈린다.

          False -> 더미 로그만 남긴다. 하드웨어 없이 상태 머신 전체를 검증할 때 쓴다.
          True  -> gripper_controller 의 FollowJointTrajectory 액션을 호출한다.
        """
        if self.preview_only:
            self.get_logger().info(f'[preview] 그리퍼 {label} (건너뜀)')
            return True
        if not self.use_real_gripper:
            self.get_logger().info(f'[DUMMY] 그리퍼 {label} (목표 {position:.3f} m)')
            self._sleep(self.gripper_settle_time)
            return True

        from builtin_interfaces.msg import Duration as DurationMsg
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'그리퍼 액션 서버({self.gripper_action})를 찾지 못했습니다.')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.gripper_joint]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start = DurationMsg(sec=1, nanosec=0)
        goal.trajectory.points.append(point)

        self.get_logger().info(f'그리퍼 {label} (목표 {position:.3f} m)')
        from action_msgs.msg import GoalStatus
        result = self.moveit._send_goal(
            self._gripper_client, goal, 15.0, 'gripper ' + label, return_wrapper=True)
        if (result is None or result.status != GoalStatus.STATUS_SUCCEEDED
                or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
            self.get_logger().error('그리퍼 액션이 실패했습니다.')
            return False
        self._sleep(self.gripper_settle_time)
        return True

    # ==================================================================
    # 이동 헬퍼 (로그 포함)
    # ==================================================================
    def _prefix(self):
        return '[preview] ' if self.preview_only else ''

    def _goto_joints(self, label, positions):
        self.get_logger().info(f'{self._prefix()}[Joint Space] {label}')
        ok, trajectory = self.moveit.move_to_joints(
            self.arm_joints, positions,
            start_state=self._preview_state, plan_only=self.preview_only)
        return self._after_motion(
            label, ok, trajectory, joint_target=dict(zip(self.arm_joints, positions)))

    def _goto_pose(self, label, pose, free_yaw=False, tilt_tolerance=None,
                   position_only=False):
        p = pose.position
        notes = []
        if position_only:
            notes.append('위치만')
        if free_yaw:
            notes.append('yaw 자유')
        if tilt_tolerance and not position_only:
            notes.append(f'tilt≤{tilt_tolerance:.2f}')
        suffix = f' ({", ".join(notes)})' if notes else ''
        self.get_logger().info(
            f'{self._prefix()}[Joint Space] {label} -> '
            f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f}){suffix}')
        ok, trajectory = self.moveit.move_to_pose(
            pose, start_state=self._preview_state, plan_only=self.preview_only,
            free_yaw=free_yaw, tilt_tolerance=tilt_tolerance,
            position_only=position_only)
        if not ok:
            self.get_logger().error('IK 해가 없거나 경로를 찾지 못했습니다.')
        # 이어지는 Cartesian 구간이 실제 도달 자세를 쓰도록 궤적을 남겨둔다.
        self._last_pose_trajectory = trajectory if ok else None
        return self._after_motion(label, ok, trajectory)

    def _achieved_ee_pose(self):
        """직전 _goto_pose 가 도달한 그리퍼 Pose (base_frame). 실패 시 None."""
        if self.guard_real_commands and not self.preview_only:
            guard = self._command_guard
            with guard.lock:
                if guard.reason(time.monotonic()):
                    return None
                positions = [guard.feedback[j] for j in self.arm_joints]
            return self.moveit.compute_fk(self.arm_joints, positions)
        trajectory = getattr(self, '_last_pose_trajectory', None)
        if trajectory is None:
            return None
        end = trajectory_end_state(trajectory)
        if end is None:
            return None
        return self.moveit.compute_fk(
            list(end.joint_state.name), list(end.joint_state.position))

    def _align_grasp_to_achieved(self, pre_pose, grasp_pose, *, anchor_contact=False,
                                achieved_pose=None):
        """
        Ready 자세를 계획한 뒤 호출한다. yaw 정렬 + tilt 허용으로 계획하면 실제
        도달 자세가 명목 자세와 달라지므로, FK 로 실제 도달 Pose 를 읽어:

          - Ready waypoint = 실제 도달 Pose (자세 + 위치)
          - Grasp waypoint = 그 위치에서 **그리퍼 접근축(+Z)** 방향으로 명목
            하강 거리만큼 이동, 자세는 도달 자세 그대로

        이렇게 하면 접근이 기울어져도 Step 4 직선 하강이 그리퍼 축을 따라 곧게
        미끄러져 들어간다. FK 실패 시 명목 pose 를 그대로 쓴다.

        position_only 모드면 그리퍼 축이 임의라, 위치는 명목(월드 수직) 그대로
        두고 자세만 실제 도달 자세로 stamp 한다 → Step 4 하강이 순수 병진(수직).

        anchor_contact=True인 집기에서는 명목 접촉점을 지나는 목표면을 유지한다.
        실제 접근축과 이 면의 교점까지 거리만 보정하므로 접근 위치 오차가
        삽입 깊이 오차로 누적되지 않는다. 목표면은 기울어져 있어도 된다.
        """
        fk = achieved_pose if achieved_pose is not None else self._achieved_ee_pose()
        if fk is None:
            self.get_logger().warn(
                '도달 자세 FK 조회 실패 — Cartesian 구간은 명목 파지 자세를 씁니다.')
            return pre_pose, grasp_pose

        q = (fk.orientation.x, fk.orientation.y, fk.orientation.z, fk.orientation.w)
        p0 = (fk.position.x, fk.position.y, fk.position.z)

        if self.grasp_ik_mode == 'position_only':
            return (self._stamp_orientation(pre_pose, q),
                    self._stamp_orientation(grasp_pose, q))

        # 명목 하강 거리 (pre -> grasp 사이 직선 거리).
        descent = math.dist(
            (pre_pose.position.x, pre_pose.position.y, pre_pose.position.z),
            (grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z))
        axis = quat_rotate_vector(q, (0.0, 0.0, 1.0))  # 그리퍼 접근축 (물체 쪽)
        if anchor_contact:
            # Intersect the actual approach ray with the ORIGINAL contact plane.
            # Repeating the nominal distance from an approach pose 3 mm too low
            # also moved the contact 3 mm too low, sometimes into the table.
            o = grasp_pose.orientation
            normal = quat_rotate_vector((o.x,o.y,o.z,o.w), (0.,0.,1.))
            denom = sum(a*n for a,n in zip(axis,normal))
            target = (grasp_pose.position.x,grasp_pose.position.y,grasp_pose.position.z)
            corrected = sum((t-p)*n for t,p,n in zip(target,p0,normal))/denom if denom>.98 else float('nan')
            if not math.isfinite(corrected) or corrected<=0 or abs(corrected-descent)>.015:
                self.get_logger().error('실제 접근 자세와 집기 목표면의 차이가 큽니다. 하강하지 않습니다.')
                return None,None
            self.get_logger().info(
                f'Pick 하강 거리: 명목 {descent*1000:.1f}mm → {corrected*1000:.1f}mm; '
                '접근 오차를 보정하고 인식 기준 집기 깊이는 유지합니다.')
            descent = corrected
        p_grasp = tuple(p0[i] + descent * axis[i] for i in range(3))
        return _make_pose(p0, q), _make_pose(p_grasp, q)

    @staticmethod
    def _stamp_orientation(pose, q):
        """position 은 그대로, orientation 만 q 로 바꾼 새 Pose."""
        return _make_pose(
            (pose.position.x, pose.position.y, pose.position.z), q)

    def _goto_cartesian(self, label, pose):
        p = pose.position
        self.get_logger().info(
            f'{self._prefix()}[Cartesian] {label} -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
        ok, trajectory = self.moveit.move_cartesian(
            [pose],
            max_step=self.cartesian_max_step,
            min_fraction=self.cartesian_min_fraction,
            start_state=self._preview_state, plan_only=self.preview_only)
        return self._after_motion(label, ok, trajectory)

    def _goto_retreat(self, label, pre_pose):
        """
        직선 상승(후퇴). 실패하면 관절 공간으로 같은 pre 자세까지 올라간다.

        후퇴의 목적은 파지한 물체를 수직으로 들어 올려 주변과의 접촉을 피하는
        것이다. 이 팔은 작업 반경 끝에서 직선 경로의 중간 IK 가 자주 끊겨
        (fraction 부족) 후퇴 단계 전체가 실패하곤 한다. 완벽한 직선이 안 되면
        관절 공간 이동으로라도 그리퍼를 그 지점까지 빼내는 편이 낫다.
        """
        p = pre_pose.position
        self.get_logger().info(
            f'{self._prefix()}[Cartesian] {label} -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
        ok, trajectory = self.moveit.move_cartesian(
            [pre_pose],
            max_step=self.cartesian_max_step,
            min_fraction=self.cartesian_min_fraction,
            start_state=self._preview_state, plan_only=self.preview_only)
        if not ok:
            if self.guard_real_commands and not self.preview_only and self.use_real_gripper:
                state = self._release_current_state()
                if state is None or not self.moveit.check_state_validity(state):
                    self.get_logger().error(
                        f'{label}: 현재 자세가 충돌 상태이거나 확인 불가 — 대체 경로를 실행하지 않습니다.')
                    return self._after_motion(label, False, trajectory)
            self.get_logger().warn(
                f'{label}: 직선 상승이 안 풀려 관절 공간 이동으로 대체합니다 (best effort).')
            ok, trajectory = self.moveit.move_to_pose(
                pre_pose, start_state=self._preview_state,
                plan_only=self.preview_only, free_yaw=False)
        return self._after_motion(label, ok, trajectory)

    def _after_motion(self, label, ok, trajectory, joint_target=None):
        """이동 결과를 기록하고, 프리뷰 모드면 궤적을 RViz 에 띄운 뒤 상태를 잇는다."""
        moveit_ok = ok
        if ok and self.guard_real_commands and not self.preview_only:
            ok = self._verify_real_arrival(trajectory, joint_target=joint_target)
        if not ok and self.guard_real_commands and not self.preview_only:
            reason = '실제 도달 확인 실패' if moveit_ok else 'MoveIt 계획/실행 실패'
            self.emergency_stop(f'{label}: {reason}')
        self._report.append((label, ok))
        if not ok:
            self.get_logger().error(f'{label} 실패.')
            return False

        if self.preview_only and trajectory is not None:
            self._publish_display(trajectory)
            end_state = trajectory_end_state(trajectory)
            if end_state is not None:
                # 다음 단계는 이 궤적이 끝난 자세에서 출발한 것으로 계획한다.
                self._preview_state = end_state
            self._sleep(self.preview_step_delay)
        return True

    def _publish_display(self, trajectory):
        """RViz 가 애니메이션할 수 있도록 궤적을 발행한다."""
        joint_trajectory = trajectory.joint_trajectory
        if not joint_trajectory.points:
            return
        message = DisplayTrajectory()
        message.trajectory_start = make_start_state(
            joint_trajectory.joint_names, joint_trajectory.points[0].positions)
        message.trajectory.append(trajectory)
        self._display_pub.publish(message)

    def _setup_planning_scene(self):
        """테이블을 충돌 객체로 등록한다. 이게 없으면 팔이 상판을 쓸고 지나간다."""
        if not self.add_table:
            self.get_logger().warn(
                'add_table=false — 플래닝 씬에 테이블이 없습니다. '
                '팔이 상판을 통과하는 경로가 나올 수 있습니다.')
            return True

        size_x, size_y, size_z = (float(v) for v in self.table_size)
        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = float(self.table_center_xy[0])
        pose.position.y = float(self.table_center_xy[1])
        # 상판이 정확히 table_top_z 에 오도록 상자 중심을 두께의 절반만큼 내린다.
        pose.position.z = float(self.table_top_z) - size_z / 2.0

        if not self.moveit.add_collision_box(
                'table', self.base_frame, pose, (size_x, size_y, size_z)):
            return False
        self.get_logger().info(
            f'플래닝 씬에 테이블 추가 — 상판 z={self.table_top_z:.3f}, '
            f'크기 {size_x}x{size_y}x{size_z} ({self.base_frame} 기준)')
        return True

    # ==================================================================
    # 9단계 상태 머신
    # ==================================================================
    def run(self):
        """
        서버 준비 후 시퀀스를 실행한다. 성공하면 True.

        loop_mode 면 한 사이클(홈 → 9단계 → 홈)이 끝날 때마다 홈에서 멈춰,
        estop 콘솔의 c(또는 ~/continue)를 받으면 상태를 초기화하고 다시 실행한다.
        마커는 매 사이클 Step 2 에서 다시 인식한다. Ctrl-C 로 종료.
        """
        if not self.moveit.wait_for_servers():
            return False
        if not self._wait_for_arm_controller():
            return False
        if not self._setup_planning_scene():
            return False

        if not self.auto_start:
            self.get_logger().info(
                'auto_start=false — 시작 대기 중. 다음 명령으로 시작하세요:\n'
                '    ros2 service call /piper_pnp_controller/start std_srvs/srv/Trigger')
            self._start_requested.wait()
            if self._abort.is_set():
                self.get_logger().warn('시작 전에 중단되었습니다.')
                return False

        if self._batch_workspace is not None:
            if self._batch_workspace.sweep:
                return self._run_sweep()
            return self._run_batch()

        cycle = 0
        last_ok = False
        while rclpy.ok() and not self._terminate.is_set():
            cycle += 1
            if self.loop_mode:
                self.get_logger().info('#' * 62)
                self.get_logger().info(f' 사이클 {cycle} 시작')
                self.get_logger().info('#' * 62)

            last_ok = self._run_cycle()
            self.print_report()

            if not self.loop_mode:
                return last_ok
            if self._terminate.is_set():
                return False

            stopped = self._abort.is_set()
            if stopped:
                self.get_logger().warn(
                    '정지됨. estop 콘솔에서 r 로 게이트를 열고 c 로 다음 사이클을 '
                    '시작하세요. (종료는 Ctrl-C)')
            elif not last_ok:
                self.get_logger().warn('사이클 실패 — 홈 자세로 복귀합니다.')
                self._goto_joints('Zero Pose', self.zero_pose)

            self.get_logger().info('#' * 62)
            self.get_logger().info(
                f" 사이클 {cycle} {'정지' if stopped else ('완료' if last_ok else '실패')}. "
                '다음 사이클은 estop 콘솔에서 c. 종료는 Ctrl-C.')
            self.get_logger().info('#' * 62)

            # c 를 기다린다. 정지 상태(_abort)면 r 로 먼저 해제해야 진행.
            self._continue_event.clear()
            while not self._terminate.is_set():
                if not self._continue_event.wait(1.0):
                    continue
                self._continue_event.clear()
                if self._abort.is_set():
                    self.get_logger().warn(
                        '아직 정지 상태입니다 — estop 콘솔에서 r 로 해제 후 다시 c.')
                    continue
                break
            if self._terminate.is_set():
                return False
            self._reset_cycle_state()

        return last_ok

    def _reset_cycle_state(self):
        """다음 사이클을 위해 타겟/리포트 상태를 초기화한다."""
        with self._targets_lock:
            self._targets.clear()
            self._sample_object_identity = None
            self._target_stamps.clear()
            self._last_target_stamp.clear()
            self._target_samples[PICK].clear()
            self._target_samples[PLACE].clear()
            if hasattr(self, '_sweep_scene_samples'):
                self._sweep_scene_samples.clear()
            self._frozen = False
        self._targets_ready.clear()
        self._report = []
        self._report_printed = False
        self._preview_state = None
        self._last_pose_trajectory = None
        self._pick_grasp_q = None
        self._pick_contact_position = None

    def _run_cycle(self, *, from_camera=False, return_home=True):
        """기존 9단계. 연속 feed는 성공한 Step 8에서 다음 Step 2 관측으로 연결한다.

        from_camera는 직전 사이클의 개방 및 촬영 자세 도달이 모두 성공했을 때만
        _run_batch가 설정한다. feed_drop은 집기 후퇴 뒤 곧바로 중앙으로 이동해
        공중 개방하고, 다음 관측을 위해 촬영 자세로 복귀한다.
        """
        self.get_logger().info('=' * 62)
        if self.preview_only:
            self.get_logger().info(' PiPER Pick & Place — 프리뷰 (계획만, 실행 안 함)')
        else:
            self.get_logger().info(' PiPER Pick & Place 시작')
        self.get_logger().info('=' * 62)

        # ---------- Step 1. Zero Pose ----------
        if not from_camera:
            if not self._begin_step(1, 'Zero Pose (대기)'):
                return False
            if not self._prepare_real_control():
                self.get_logger().error('실기 명령 준비 실패 — 동작을 시작하지 않습니다.')
                return False
            if not self._goto_joints('Zero Pose', self.zero_pose):
                return False
            # 첫 집기 전에 개방한다. 연속 이송의 이후 사이클은 Step 7에서 이미 개방했다.
            if not self.open_gripper():
                return False

        # ---------- Step 2. Camera-Ready + 타겟 확정 ----------
        if not self._begin_step(2, 'Camera-Ready Pose (탐색)'):
            return False
        if not from_camera and not self._goto_joints('Camera-Ready Pose', self.camera_ready_pose):
            return False
        if not self._wait_for_targets():
            return False
        feed_drop = getattr(getattr(self, '_batch_workspace', None), 'feed_drop', False)
        if feed_drop and not self._prepare_feed_drop():
            return False
        if getattr(self, '_batch_workspace', None) is not None:
            if self.aborted:
                return False
            self._batch_progress.reserve()

        # ---------- Step 3. Pick-Ready ----------
        if not self._begin_step(3, 'Pick-Ready Pose (접근)'):
            return False
        pick_pre, pick_grasp = self._approach_ready(
            'Pick-Ready Pose', PICK, self.grasp_offset, self.approach_offset)
        if pick_pre is None:
            return False

        # ---------- Step 4. Grasp ----------
        if not self._begin_step(4, 'Grasp (파지)'):
            return False
        if not self._goto_cartesian('Pick 타겟까지 직선 하강', pick_grasp):
            return False
        if not self.close_gripper():
            return False

        # ---------- Step 5. 집기 후퇴 ----------
        if not self._begin_step(5, '집기 후퇴 → goal 직행' if feed_drop else 'Camera-Ready Pose (복귀 1)'):
            return False
        if not self._goto_retreat('직선 상승', pick_pre):
            return False
        if feed_drop and not self._sweep_attach_pick():
            return False
        if not feed_drop and not self._goto_joints('Camera-Ready Pose', self.camera_ready_pose):
            return False

        if feed_drop:
            if not self._run_feed_drop():
                return False
        else:
            # ---------- Step 6. Place-Ready ----------
            if not self._begin_step(6, 'Place-Ready Pose (이동)'):
                return False
            place_pre, place_grasp = self._approach_ready(
                'Place-Ready Pose', PLACE, self.place_offset, self.place_approach_offset)
            if place_pre is None:
                return False

            # ---------- Step 7. Place ----------
            if not self._begin_step(7, 'Place (놓기)'):
                return False
            if not self._check_place_release(place_pre, place_grasp):
                return False
            if not self._goto_cartesian('Place 타겟까지 직선 하강', place_grasp):
                return False
            # 실제 하강 끝점으로 방향과 개방 여유를 다시 확인한다.
            if not self._check_place_alignment():
                return False
            if not self._check_place_release(place_pre):
                return False
            if not self.open_gripper():
                return False

            # ---------- Step 8. 복귀 2 ----------
            if not self._begin_step(8, 'Camera-Ready Pose (복귀 2)'):
                return False
            if not self._goto_retreat('직선 상승', place_pre):
                return False
            if not self._goto_joints('Camera-Ready Pose', self.camera_ready_pose):
                return False

        # ---------- Step 9. Zero Pose ----------
        if return_home:
            if not self._begin_step(9, 'Zero Pose (종료)'):
                return False
            if not self._goto_joints('Zero Pose', self.zero_pose):
                return False

        self.get_logger().info('=' * 62)
        if self.preview_only:
            self.get_logger().info(
                ' 프리뷰 완료 — 실제로 실행하려면 preview_only:=false 로 다시 실행하세요')
        else:
            self.get_logger().info(' Pick & Place 완료')
        self.get_logger().info('=' * 62)
        return True

    def print_report(self):
        """단계별 성공/실패 요약. 프리뷰에서 특히 유용하다."""
        if not self._report or getattr(self, '_report_printed', False):
            return
        self._report_printed = True
        self.get_logger().info('-' * 62)
        self.get_logger().info(' 단계별 결과')
        for label, ok in self._report:
            self.get_logger().info(f"   {'OK  ' if ok else 'FAIL'}  {label}")
        self.get_logger().info('-' * 62)

    def _wait_for_arm_controller(self):
        """
        arm_controller 의 FollowJointTrajectory 액션 서버를 기다린다.

        런치 직후에는 컨트롤러 스포너가 아직 끝나지 않았을 수 있다. 그 상태에서
        move_group 에 목표를 보내면 "Unable to identify any set of controllers" 로
        실행이 실패하므로, 첫 이동 전에 여기서 한 번 막아준다.
        """
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient

        self.get_logger().info(f'{self.arm_controller_action} 준비 대기 중...')
        client = ActionClient(
            self, FollowJointTrajectory, self.arm_controller_action,
            callback_group=self._cb_group)
        try:
            if not client.wait_for_server(timeout_sec=self.controller_wait_timeout):
                self.get_logger().error(
                    f'{self.arm_controller_action} 액션 서버를 찾지 못했습니다. '
                    "컨트롤러 스포너 로그를 확인하세요 "
                    "(ros2 control list_controllers).")
                return False
        finally:
            client.destroy()
        self.get_logger().info('컨트롤러 준비 완료.')
        return True

    def _expire_targets_locked(self):
        if self._frozen:
            return
        if getattr(self, '_batch_workspace', None) is not None:
            self._set_batch_place_locked()
        now_ns = self.get_clock().now().nanoseconds
        for key in (PICK, PLACE):
            stamp = self._target_stamps.get(key)
            if stamp is not None and not 0 <= (now_ns-stamp)/1e9 <= self.target_max_age:
                self._targets.pop(key, None)
                self._target_stamps.pop(key, None)
        if PICK in self._targets and PLACE in self._targets:
            self._targets_ready.set()
        else:
            self._targets_ready.clear()

    def _expire_targets(self):
        with self._targets_lock:
            self._expire_targets_locked()

    def _wait_for_targets(self):
        # Camera-Ready 도달 이후 촬영된 서로 다른 관측만 새로 모은다.
        with self._targets_lock:
            self._target_not_before_ns = self.get_clock().now().nanoseconds
            self._targets.clear()
            self._target_stamps.clear()
            self._last_target_stamp.clear()
            for buf in self._target_samples.values():
                buf.clear()
            if hasattr(self, '_sweep_scene_samples'):
                self._sweep_scene_samples.clear()
            self._frozen = False
            self._targets_ready.clear()
            if getattr(self, '_batch_workspace', None) is not None:
                self._set_batch_place_locked()
        batch = getattr(self, '_batch_workspace', None) is not None
        self.get_logger().info('Camera-Ready 이후 새 박스 관측을 기다립니다.' if batch
                               else 'Camera-Ready 이후 새 Pick/Place 관측을 기다립니다.')
        deadline = time.monotonic() + self.marker_wait_timeout
        next_detail = time.monotonic()+3.
        while not self.aborted:
            if time.monotonic() >= deadline:
                if not batch:
                    break
                self.get_logger().info('집기 영역에서 안정된 박스가 보이지 않습니다. 촬영 자세로 계속 대기합니다.')
                deadline = time.monotonic() + self.marker_wait_timeout
            self._targets_ready.wait(0.1)
            with self._targets_lock:
                self._expire_targets_locked()
                if getattr(getattr(self, '_batch_workspace', None), 'sweep', False) and time.monotonic() >= next_detail:
                    self.get_logger().info(
                        f'sweep 관측 대기: pose={len(self._target_samples[PICK])}, '
                        f'depth={len(getattr(self,"_sweep_scene_samples",()))}, '
                        f'필요 관측={self.marker_stable_samples}; 유효 자세/깊이 조건 확인 중')
                    next_detail = time.monotonic()+3.
                if self._targets_ready.is_set():
                    self._frozen = True
                    for key in (PICK, PLACE):
                        target = self._targets[key]
                        self.get_logger().info(
                            f'{key} 확정: {target["position"]}, quaternion_xyzw={target["orientation"]}')
                    return True
        self.get_logger().error('새로운 Pick/Place 관측을 확정하지 못했습니다.')
        return False

    # ==================================================================
    # 기타
    # ==================================================================
    # ==================================================================
    # 비상 정지
    # ==================================================================
    @property
    def aborted(self):
        return self._abort.is_set()

    def emergency_stop(self, reason, terminate=None):
        """
        정지 체인. 순서가 중요하다.

          1) control_enable(false)  게이트를 먼저 닫아 새 명령이 팔에 가지 않게 한다.
          2) emergency_stop         현재 자세를 붙잡는다. 전원을 끊지 않으므로
                                    팔이 자중으로 떨어지지 않는다.
          3) goal 취소              move_group 이 궤적 실행을 중단한다.

        게이트를 먼저 닫지 않으면 2)로 자세를 잡아도 아직 흐르고 있는 궤적 명령과
        서로 밀어내게 된다.

        terminate=True 는 Ctrl-C 같은 명시적 종료다. 실기에서는 기본 정지가
        현재 사이클만 중단하고 서비스를 유지한다. h → y 확인 후에만 홈 복귀와
        새 실험 초기화를 수행한다. 정지 자체로 자동 복귀하지 않는다.

        [주의] enable_agx_arm(false) 는 절대 쓰지 않는다. 토크가 풀려 팔이 떨어진다.
        """
        if terminate is None:
            terminate = not self.loop_mode and not getattr(self, '_restart_supported', False)
        if hasattr(self, '_restart_lock'):
            self._cancel_restart_request()
        self._command_guard.close()
        already = getattr(self, '_stop_sent', False)
        self._abort.set()
        self._stop_sent = True
        if terminate:
            self._terminate.set()
        # 대기 중인 지점들을 전부 깨운다 (시작 대기 / 단계 승인 대기)
        self._continue_event.set()
        self._start_requested.set()
        if already:
            return

        self.get_logger().warn('=' * 62)
        self.get_logger().warn(f' 비상 정지: {reason}')
        self.get_logger().warn('=' * 62)

        if self.preview_only:
            self.moveit.cancel_active_goal()
            return
        if self._call_set_bool(self._control_enable_client, False,
                               self.control_enable_service):
            self.get_logger().warn(' 1) 제어 게이트 차단')
        if self._call_empty(self._emergency_stop_client, self.emergency_stop_service):
            self.get_logger().warn(' 2) 현재 자세 유지 명령 전송')
        if self.moveit.cancel_active_goal():
            self.get_logger().warn(' 3) 실행 중 궤적 취소')

    def resume(self):
        """기존 시뮬레이션/loop 해제. 실기는 h → y로 새 실험을 준비한다."""
        if (getattr(self, '_batch_workspace', None) is not None or
                getattr(self, '_restart_supported', False)):
            self.get_logger().warn('연속 이송 정지 후에는 h → y로 홈/이송 횟수를 초기화하세요.')
            return False
        self._abort.clear()
        self._continue_event.clear()
        if self.guard_real_commands:
            self._command_guard.close()
            self.get_logger().info('정지 해제. 다음 사이클 승인 후 다시 동기화합니다.')
            return True
        opened = self._call_set_bool(self._control_enable_client, True,
                                     self.control_enable_service)
        self.get_logger().info(
            '제어 게이트를 다시 열었습니다.' if opened
            else '정지 상태를 해제했습니다 (제어 게이트 서비스 없음).')
        return True

    def _call_set_bool(self, client, value, name):
        if not client.service_is_ready():
            self.get_logger().warn(
                f"'{name}' 서비스가 없습니다 (시뮬레이션이면 정상).",
                throttle_duration_sec=30.0)
            return False
        request = SetBool.Request()
        request.data = value
        client.call_async(request)      # 정지 경로에서는 응답을 기다리지 않는다
        return True

    def _call_empty(self, client, name):
        if not client.service_is_ready():
            self.get_logger().warn(
                f"'{name}' 서비스가 없습니다 (시뮬레이션이면 정상).",
                throttle_duration_sec=30.0)
            return False
        client.call_async(Empty.Request())
        return True

    # ==================================================================
    # 단계 진입 / 승인
    # ==================================================================
    def _begin_step(self, number, title):
        """단계 헤더를 찍고, 중단 여부와 단계 승인을 확인한다."""
        if self._abort.is_set():
            self.get_logger().warn(f'중단됨 — Step {number} 을 시작하지 않습니다.')
            return False

        self.get_logger().info(f'--- Step {number}/9: {title} ---')

        if not self.step_confirm or self.preview_only:
            return True

        return self._wait_for_confirmation(f'Step {number}', timeout=self.step_confirm_timeout)

    def _wait_for_confirmation(self, label, timeout=None):
        if self.aborted:
            return False
        self._continue_event.clear()
        self._waiting_confirmation = True
        self.get_logger().info(
            f'  [승인 대기] {label}: estop 터미널에서 c 를 누르거나\n'
            '             ros2 service call /piper_pnp_controller/continue '
            'std_srvs/srv/Trigger')
        approved = self._continue_event.wait(timeout)
        self._waiting_confirmation = False
        if not approved:
            self.get_logger().error(
                f'단계 승인 대기 시간 초과 ({timeout:.0f}s) — 중단합니다.')
            self._abort.set()
            return False
        if self._abort.is_set():
            self.get_logger().warn('승인 대기 중 중단되었습니다.')
            return False
        return True

    # ==================================================================
    # 서비스 콜백
    # ==================================================================
    def _on_start_request(self, request, response):
        del request
        self._start_requested.set()
        response.success = True
        response.message = 'Pick & Place 시작'
        return response

    def _on_stop_request(self, request, response):
        del request
        self.emergency_stop('정지 서비스 호출')
        response.success = True
        response.message = '정지했습니다'
        return response

    def _on_resume_request(self, request, response):
        del request
        response.success = self.resume()
        if not response.success:
            response.message = '정지 후 재시작: 그리퍼와 goal을 모두 비우고 h → y. r로 재개하지 않습니다.'
            return response
        response.message = (
            '정지 해제. loop_mode 면 c 로 다음 사이클, 아니면 시퀀스를 재시작하세요.'
            if self.loop_mode else '정지 해제 (시퀀스는 재시작해야 합니다)')
        return response

    def _on_continue_request(self, request, response):
        del request
        if self.aborted or getattr(self, '_restart_waiting', False):
            response.success = False
            response.message = '정지/재시작 대기 중입니다. 그리퍼와 goal을 비우고 h → y로 초기화하세요.'
            return response
        if getattr(self, '_batch_workspace', None) is not None:
            if not self._waiting_confirmation or self.aborted:
                response.success = False
                response.message = '현재 c 승인 대기 중이 아닙니다. 실행 중 입력은 다음 사이클에 예약되지 않습니다.'
                return response
        response.success = True
        workspace = getattr(self, '_batch_workspace', None)
        if workspace is None:
            response.message = '다음 단계 진행'
        elif workspace.operator_feed and not getattr(self, 'continuous_feed', False):
            response.message = '박스 투입 완료 승인 — 9단계 한 사이클 자동 진행'
        else:
            response.message = '연속 이송 시작 승인 — 다음 단계와 다음 박스는 자동 진행'
        self._continue_event.set()
        return response

    def _sleep(self, seconds):
        """노드가 spin 중인 상태에서 워커 스레드를 재우는 단순 대기."""
        threading.Event().wait(seconds)


def _make_pose(position, orientation):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = (float(v) for v in orientation)
    return pose


def _wait_future(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout)


def main(args=None):
    rclpy.init(args=args)
    node = PiperPnpController()

    # 노드는 별도 스레드에서 spin 하고, 상태 머신은 메인 스레드에서 블로킹으로 돈다.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def on_sigint(signum, frame):
        """
        Ctrl-C 를 안전하게 만든다.

        기본 동작은 노드만 죽이고 끝나는데, 그러면 이미 arm_controller 로 넘어간
        궤적이 그대로 끝까지 실행된다. 즉 Ctrl-C 로는 팔이 서지 않는다.
        여기서 정지 체인을 먼저 돌려 실제로 멈추게 한다.

        신호 핸들러 안에서 블로킹하면 안 되므로 별도 스레드로 돌린다.
        한 번 더 누르면 즉시 종료한다.
        """
        del signum, frame
        if node.aborted:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        # Ctrl-C 는 항상 종료 의도 — loop_mode 여도 프로세스를 내린다.
        threading.Thread(
            target=node.emergency_stop, args=('Ctrl-C',),
            kwargs={'terminate': True}, daemon=True).start()

    # rclpy.init() 이 자체 핸들러를 걸어두므로 그 뒤에 덮어써야 한다.
    signal.signal(signal.SIGINT, on_sigint)

    try:
        node.run_restartable()
    except KeyboardInterrupt:
        pass
    finally:
        if node.guard_real_commands and not node.preview_only:
            node.emergency_stop('컨트롤러 종료')
            time.sleep(0.2)
        node.print_report()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
