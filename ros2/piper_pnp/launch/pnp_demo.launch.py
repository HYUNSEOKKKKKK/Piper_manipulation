"""
pnp_demo.launch.py — 시뮬레이션(fake hardware) Pick & Place 데모.

실제 로봇과 카메라 없이 전체 파이프라인을 검증한다. 다음을 한 번에 띄운다:

    robot_state_publisher   커스텀 URDF -> TF
    move_group              MoveIt 2 플래닝
    ros2_control_node       mock_components/GenericSystem (가상 로봇)
    컨트롤러 스포너          joint_state_broadcaster / arm_controller / gripper_controller
    RViz                    MotionPlanning + PnP 타겟 마커
    piper_pnp_controller    9단계 상태 머신

타겟은 별도 터미널에서 토픽으로 주입한다 (README 참고):

    ros2 topic pub --once /aruco_poses geometry_msgs/msg/PoseArray ...

사용 예:
    ros2 launch piper_pnp pnp_demo.launch.py
    ros2 launch piper_pnp pnp_demo.launch.py use_rviz:=false auto_start:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _setup(context):
    from ament_index_python.packages import get_package_share_directory
    from launch_ros.actions import Node

    from piper_pnp.launch_utils import moveit_bringup, pnp_controller_node

    def arg(name):
        return LaunchConfiguration(name).perform(context)

    marker_ids = {
        'pick_marker_id': int(arg('pick_marker_id')),
        'place_marker_id': int(arg('place_marker_id')),
    }

    actions = moveit_bringup(
        # mock 하드웨어라 명령과 상태가 같은 토픽('joint_states')으로 순환한다.
        rsp_joint_states='joint_states',
        control_joint_states='joint_states',
        use_rviz=(arg('use_rviz').lower() == 'true'),
    )

    # 가상 마커 발행기: 실제 카메라 없이 카메라 시야 판정까지 재현한다.
    if arg('use_fake_aruco').lower() == 'true':
        actions.append(
            Node(
                package='piper_pnp',
                executable='fake_aruco_publisher',
                name='fake_aruco_publisher',
                output='screen',
                emulate_tty=True,
                parameters=[
                    get_package_share_directory('piper_pnp')
                    + '/config/fake_aruco_params.yaml',
                    marker_ids,
                ],
            )
        )

    actions.append(
        pnp_controller_node({
            'auto_start': arg('auto_start').lower() == 'true',
            'preview_only': arg('preview_only').lower() == 'true',
            'step_confirm': arg('step_confirm').lower() == 'true',
            'loop_mode': arg('loop_mode').lower() == 'true',
            # 시뮬레이션에는 잡을 물체가 없으므로 그리퍼는 더미 로그로 둔다.
            'use_real_gripper': False,
            **marker_ids,
        })
    )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='true', choices=['true', 'false'],
            description='RViz 를 함께 띄울지 여부.'),
        DeclareLaunchArgument(
            'auto_start', default_value='true', choices=['true', 'false'],
            description='true 면 두 마커를 받는 즉시 시퀀스를 시작한다. '
                        'false 면 /piper_pnp_controller/start 서비스 호출을 기다린다.'),
        DeclareLaunchArgument(
            'use_fake_aruco', default_value='true', choices=['true', 'false'],
            description='가상 마커 발행기를 띄울지 여부. false 면 ros2 topic pub 으로 '
                        '직접 타겟을 주입해야 한다.'),
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
        OpaqueFunction(function=_setup),
    ])
