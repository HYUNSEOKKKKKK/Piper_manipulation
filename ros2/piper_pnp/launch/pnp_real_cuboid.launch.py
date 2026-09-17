"""RGB-D cuboid 실기 런치. 기본값은 계획만 수행한다.

mock 명령은 /pnp/control/joint_states로 분리한다. 실제 명령은 컨트롤러가
현재 관절값으로 mock을 동기화한 뒤 /control/joint_states로 중계한다.
드라이버는 자동 enable 및 외부 명령 수신이 꺼진 상태로 시작한다.

실행 모드와 현재 물체 치수는 scripts/run_robot.sh, scripts/run_bridge.sh를 사용한다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    from piper_pnp.launch_utils import moveit_bringup, pnp_controller_node

    def arg(name):
        return LaunchConfiguration(name).perform(context)

    actions = moveit_bringup(
        rsp_joint_states='feedback/joint_states',
        control_joint_states='pnp/control/joint_states',
        use_rviz=(arg('use_rviz').lower() == 'true'),
    )

    # ---------------- 실제 팔 드라이버 ----------------
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('agx_arm_ctrl'),
                    'launch', 'start_single_agx_arm.launch.py')
            ),
            launch_arguments={
                'can_port': arg('can_port'),
                'arm_type': 'piper',
                'effector_type': 'agx_gripper',
                'auto_enable': 'false',
                'control_enabled': 'false',
            }.items(),
        )
    )

    # ---------------- RealSense D435 (color + aligned depth) ----------------
    # ArUco 경로와 달리 depth 가 필요하다. align_depth 로 depth 를 color 프레임에
    # 정렬해 aligned_depth_to_color/image_raw 로 낸다 (bridge 가 구독).
    if arg('use_camera').lower() == 'true':
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('realsense2_camera'),
                        'launch', 'rs_launch.py')
                ),
                launch_arguments={
                    'camera_name': 'camera',
                    'enable_color': 'true',
                    'enable_depth': 'true',
                    'align_depth.enable': 'true',
                    'enable_sync': 'true',
                    'rgb_camera.color_profile': '640x480x30',
                    'depth_module.depth_profile': '640x480x30',
                    'pointcloud.enable': 'false',
                }.items(),
            )
        )

    # ---------------- (ArUco 없음) ----------------
    # /aruco_markers 는 perception/ros_pose_bridge.py 가 defm 환경에서 발행한다.

    # ---------------- 상태 머신 ----------------
    actions.append(
        pnp_controller_node({
            'auto_start': arg('auto_start').lower() == 'true',
            'preview_only': arg('preview_only').lower() == 'true',
            'guard_real_commands': True,
            'grasp_offset': float(arg('grasp_offset')),
            'step_confirm': arg('step_confirm').lower() == 'true',
            'loop_mode': arg('loop_mode').lower() == 'true',
            'batch_config_file': arg('batch_config_file'),
            'continuous_feed': arg('continuous_feed').lower() == 'true',
            'object_models_file': arg('object_models_file'),
            'velocity_scaling': float(arg('velocity_scaling')),
            'acceleration_scaling': float(arg('acceleration_scaling')),
            'gripper_settle_time': float(arg('gripper_settle_time')),
            'use_real_gripper': arg('use_real_gripper').lower() == 'true',
            'pick_marker_id': int(arg('pick_marker_id')),
            'place_marker_id': int(arg('place_marker_id')),
        })
    )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('object_models_file', default_value=''),
        DeclareLaunchArgument('velocity_scaling', default_value='0.1'),
        DeclareLaunchArgument('acceleration_scaling', default_value='0.1'),
        DeclareLaunchArgument('gripper_settle_time', default_value='1.0'),
        DeclareLaunchArgument('batch_config_file', default_value='',
            description='연속 이송의 집기 영역/놓기 자리 JSON. 빈 값이면 기존 단일 사이클.'),
        DeclareLaunchArgument('continuous_feed', default_value='false', choices=['true', 'false'],
            description='feed 경로로 c 한 번에 연속 이송. 박스 사이 홈 복귀/재승인 생략.'),
        DeclareLaunchArgument('grasp_offset', default_value='0.02',
            description='첫 실기 기본값은 표면 2cm 위. 실제 파지는 -0.02 등 실측값 사용.'),
        DeclareLaunchArgument(
            'can_port', default_value='can0', description='PiPER 가 물린 CAN 포트.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true', choices=['true', 'false'],
            description='RViz 를 함께 띄울지 여부.'),
        DeclareLaunchArgument(
            'use_camera', default_value='true', choices=['true', 'false'],
            description='realsense2_camera 드라이버를 띄울지 여부. '
                        '카메라를 다른 곳에서 이미 띄웠다면 false.'),
        DeclareLaunchArgument(
            'auto_start', default_value='false', choices=['true', 'false'],
            description='실기 기본값은 false — 시작 서비스를 직접 호출해야 움직인다.'),
        DeclareLaunchArgument(
            'use_real_gripper', default_value='false', choices=['true', 'false'],
            description='true 면 gripper_controller 로 실제 그리퍼를 제어한다.'),
        DeclareLaunchArgument(
            'step_confirm', default_value='true', choices=['true', 'false'],
            description='true 면 각 단계 실행 전에 승인을 기다린다. 승인은 estop 콘솔의 '
                        'c 키 또는 ~/continue 서비스로 한다.'),
        DeclareLaunchArgument(
            'preview_only', default_value='true', choices=['true', 'false'],
            description='true 면 9단계를 실행하지 않고 계획만 이어 붙여 RViz 에 보여준다.'),
        DeclareLaunchArgument(
            'loop_mode', default_value='false', choices=['true', 'false'],
            description='true 면 한 사이클 후 홈에서 멈춰 c 를 받으면 다시 실행한다.'),
        DeclareLaunchArgument(
            'pick_marker_id', default_value='0',
            description='Pick 대상 ID (bridge 의 pick_marker_id 와 일치해야 함).'),
        DeclareLaunchArgument(
            'place_marker_id', default_value='1',
            description='Place 대상 ID (bridge 의 place_marker_id 와 일치해야 함).'),
        OpaqueFunction(function=_setup),
    ])
