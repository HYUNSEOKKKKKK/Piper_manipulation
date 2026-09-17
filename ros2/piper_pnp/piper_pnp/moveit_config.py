"""
piper_pnp 전용 MoveIt 설정 빌더.

런치 파일들이 공통으로 쓴다. 여기서 만들어지는 설정은 전부 piper_pnp 패키지 안의
파일만 참조하므로(로봇 메시는 agx_arm_description 에서 가져온다), 업스트림
agx_arm_moveit 의 런치나 config 에 의존하지 않는다.
"""

from moveit_configs_utils import MoveItConfigsBuilder


def build_moveit_config():
    """piper_pnp 의 MoveItConfigs 를 생성한다."""
    return (
        MoveItConfigsBuilder('piper_d435', package_name='piper_pnp')
        .robot_description(file_path='urdf/custom_piper_d435.xacro')
        .robot_description_semantic(file_path='config/piper_pnp.srdf')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .to_moveit_configs()
    )
