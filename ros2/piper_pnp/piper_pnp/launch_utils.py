"""
런치 파일 공용 헬퍼.

pnp_demo.launch.py(시뮬레이션)와 pnp_real.launch.py(실기)가 MoveIt 스택을 띄우는
방식은 조인트 상태 토픽 배선만 다르고 나머지는 같다. 그 공통부를 여기 모아
두 런치 파일이 어긋나지 않게 한다.
"""

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from piper_pnp.moveit_config import build_moveit_config


def moveit_bringup(rsp_joint_states, control_joint_states, use_rviz, rviz_config=None):
    """
    MoveIt 2 스택 전체(RSP + move_group + ros2_control + 컨트롤러 스포너 [+ RViz])를
    구성해 launch 액션 리스트로 돌려준다.

    조인트 상태 배선
    ----------------
    rsp_joint_states
        robot_state_publisher / move_group / RViz 가 **읽는** 토픽.
        시뮬레이션에서는 mock 하드웨어가 내보내는 'joint_states',
        실기에서는 실제 팔이 올려주는 'feedback/joint_states' 를 준다.

    control_joint_states
        joint_state_broadcaster 가 **쓰는** 토픽.
        실기에서는 이것이 agx_arm_ctrl 로 들어가는 명령 통로가 된다
        ('control/joint_states').
    """
    moveit_config = build_moveit_config()
    package_share = get_package_share_directory('piper_pnp')

    actions = [
        # 로봇 모델 -> TF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[moveit_config.robot_description],
            remappings=[('joint_states', rsp_joint_states)],
        ),
        # 플래닝
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                moveit_config.to_dict(),
                {'publish_robot_description_semantic': True,
                 'publish_planning_scene': True,
                 'publish_geometry_updates': True,
                 'publish_state_updates': True,
                 'publish_transforms_updates': True},
            ],
            remappings=[('joint_states', rsp_joint_states)],
        ),
        # 컨트롤러 매니저 (mock 하드웨어)
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            output='screen',
            parameters=[
                moveit_config.robot_description,
                package_share + '/config/ros2_controllers.yaml',
            ],
            remappings=[('joint_states', control_joint_states)],
        ),
    ]

    for controller in ('joint_state_broadcaster', 'arm_controller', 'gripper_controller'):
        actions.append(
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[controller, '--controller-manager', '/controller_manager'],
                output='screen',
            )
        )

    if use_rviz:
        actions.append(
            Node(
                package='rviz2',
                executable='rviz2',
                output='log',
                arguments=['-d', rviz_config or (package_share + '/config/pnp.rviz')],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.planning_pipelines,
                    moveit_config.joint_limits,
                ],
                remappings=[('joint_states', rsp_joint_states)],
            )
        )

    return actions


def pnp_controller_node(overrides):
    """9단계 상태 머신 노드. config/pnp_params.yaml 위에 overrides 를 얹는다."""
    package_share = get_package_share_directory('piper_pnp')
    return Node(
        package='piper_pnp',
        executable='piper_pnp_controller',
        name='piper_pnp_controller',
        output='screen',
        emulate_tty=True,
        parameters=[package_share + '/config/pnp_params.yaml', overrides],
    )
