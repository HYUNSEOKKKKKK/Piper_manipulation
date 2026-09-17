"""
pnp_real.launch.py — 실제 하드웨어 Pick & Place.

pnp_demo.launch.py 에 실기 구성요소를 더한 것이다:

    agx_arm_ctrl        CAN 으로 실제 PiPER 팔 제어 (agx_arm_ros)
    realsense2_camera   D435 드라이버
    ros2_aruco          ArUco 마커 검출 -> /aruco_markers, /aruco_poses

조인트 상태 배선 (AgileX 업스트림과 동일한 구조)
-----------------------------------------------
    MoveIt -> arm_controller -> mock system -> /control/joint_states
                                                    |
                                    agx_arm_ctrl 가 구독 -> CAN -> 실제 팔
                                                    |
                                          /feedback/joint_states (실제 관절값)
                                                    |
                                  robot_state_publisher / move_group / RViz 가 구독

즉 ros2_control 의 mock system 은 "명령 통로"이고, 화면에 보이는 로봇 상태는
언제나 실제 팔의 피드백이다.

[안전] 기본값이 auto_start:=false 다. 마커가 보이자마자 팔이 움직이지 않도록
직접 시작 신호를 줘야 한다:

    ros2 service call /piper_pnp_controller/start std_srvs/srv/Trigger

사용 예:
    ros2 launch piper_pnp pnp_real.launch.py can_port:=can0
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
        # 실제 팔의 피드백을 로봇 상태로 쓴다.
        rsp_joint_states='feedback/joint_states',
        # mock system 의 출력은 agx_arm_ctrl 로 가는 명령 통로다.
        control_joint_states='control/joint_states',
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
                'auto_enable': 'true',
            }.items(),
        )
    )

    # ---------------- RealSense D435 ----------------
    # camera_name 기본값이 'camera' 라 프레임 이름이 camera_link 로 잡히고,
    # 우리 URDF 의 link6 -> camera_link 조인트에 그대로 이어 붙는다.
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
                    'enable_depth': 'false',
                    'pointcloud.enable': 'false',
                }.items(),
            )
        )

    # ---------------- ArUco 검출 ----------------
    actions.append(
        Node(
            package='ros2_aruco',
            executable='aruco_node',
            name='aruco_node',
            output='screen',
            parameters=[{
                'marker_size': float(arg('marker_size')),
                'aruco_dictionary_id': arg('aruco_dictionary_id'),
                'image_topic': arg('image_topic'),
                'camera_info_topic': arg('camera_info_topic'),
            }],
        )
    )

    # ---------------- 상태 머신 ----------------
    actions.append(
        pnp_controller_node({
            'auto_start': arg('auto_start').lower() == 'true',
            'preview_only': arg('preview_only').lower() == 'true',
            'step_confirm': arg('step_confirm').lower() == 'true',
            'loop_mode': arg('loop_mode').lower() == 'true',
            'use_real_gripper': arg('use_real_gripper').lower() == 'true',
            'pick_marker_id': int(arg('pick_marker_id')),
            'place_marker_id': int(arg('place_marker_id')),
        })
    )
    return actions


def generate_launch_description():
    return LaunchDescription([
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
            'use_real_gripper', default_value='true', choices=['true', 'false'],
            description='true 면 gripper_controller 로 실제 그리퍼를 제어한다.'),
        DeclareLaunchArgument(
            'step_confirm', default_value='false', choices=['true', 'false'],
            description='true 면 각 단계 실행 전에 승인을 기다린다. 승인은 estop 콘솔의 '
                        'c 키 또는 ~/continue 서비스로 한다. '
                        '(ros2 run piper_pnp piper_pnp_estop 이 떠 있어야 한다)'),
        DeclareLaunchArgument(
            'preview_only', default_value='false', choices=['true', 'false'],
            description='true 면 9단계를 실행하지 않고 계획만 이어 붙여 RViz 에 보여준다. '
                        '로봇은 움직이지 않는다.'),
        DeclareLaunchArgument(
            'loop_mode', default_value='false', choices=['true', 'false'],
            description='true 면 한 사이클(홈→9단계→홈) 후 홈에서 멈춰, estop 콘솔의 '
                        'c(또는 ~/continue)를 받으면 다시 실행한다. Ctrl-C 로 종료.'),
        DeclareLaunchArgument(
            'pick_marker_id', default_value='0', description='Pick 대상 ArUco 마커 ID.'),
        DeclareLaunchArgument(
            'place_marker_id', default_value='1', description='Place 대상 ArUco 마커 ID.'),
        DeclareLaunchArgument(
            'marker_size', default_value='0.048',
            description='ArUco 마커 검정 사각형 한 변의 길이 [m]. 실제 인쇄물과 '
                        '반드시 일치해야 한다 (현재 실측 48mm).'),
        DeclareLaunchArgument(
            'aruco_dictionary_id', default_value='DICT_5X5_250',
            description='마커 생성에 쓴 ArUco 사전.'),
        DeclareLaunchArgument(
            'image_topic', default_value='/camera/camera/color/image_raw',
            description='컬러 이미지 토픽.'),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='/camera/camera/color/camera_info',
            description='카메라 인트린식 토픽.'),
        OpaqueFunction(function=_setup),
    ])
