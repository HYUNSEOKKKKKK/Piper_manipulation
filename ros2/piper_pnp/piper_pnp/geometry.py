"""
순수 기하 계산 — ROS 의존성이 전혀 없다.

ROS 없이도 `pytest` 로 검증할 수 있도록 일부러 분리했다.
쿼터니언은 전부 (x, y, z, w) 순서의 4-튜플로 다룬다 (ROS geometry_msgs 와 동일).
"""

import math

# 그리퍼를 마커 쪽으로 뒤집는 회전: X 축 180도.
#   Rx(pi) 는 Z -> -Z, Y -> -Y 로 보낸다.
# 마커 Z 축은 마커 평면에서 바깥(보통 위쪽)을 향하므로, 이 회전을 곱하면
# 그리퍼의 접근 축(+Z)이 마커를 향해 파고드는 방향이 된다.
_FLIP_X_180 = (1.0, 0.0, 0.0, 0.0)


def quat_multiply(q1, q2):
    """쿼터니언 곱 q1 * q2. 입력/출력 모두 (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_rotate_vector(q, v):
    """쿼터니언 q 로 벡터 v 를 회전시킨다. v' = q * (v,0) * q^-1 의 전개형."""
    x, y, z, w = q
    vx, vy, vz = v

    # t = 2 * (q_vec x v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v' = v + w * t + (q_vec x t)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_normalize(q):
    """쿼터니언 정규화. 크기가 0 이면 항등 쿼터니언을 돌려준다."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def marker_normal(q_marker):
    """마커의 법선(로컬 +Z 축)을 부모 좌표계 기준 단위 벡터로 돌려준다."""
    return quat_rotate_vector(quat_normalize(q_marker), (0.0, 0.0, 1.0))


def grasp_orientation(q_marker):
    """
    마커 자세로부터 그리퍼(tcp_link) 목표 자세를 만든다.

        R_grasp = R_marker * Rx(180도)

    이 한 줄이 두 경우를 모두 처리한다:
      * 평면에 놓인 마커 -> roll/pitch 가 0 이므로 yaw 만 남고, 그리퍼는 똑바로
        아래를 향한다 (인형뽑기 방식).
      * 기울어진 마커   -> 기울기를 그대로 물려받아 그리퍼도 같이 기운다.
    """
    return quat_normalize(quat_multiply(quat_normalize(q_marker), _FLIP_X_180))


def approach_position(p_marker, q_marker, offset):
    """
    마커 법선 방향으로 offset 만큼 떨어진 접근(Pre-grasp) 지점.

    마커가 평면에 놓여 있으면 법선이 월드 +Z 와 일치하므로
    결과는 "타겟의 Z 축 위 offset" 과 정확히 같아진다.
    """
    nx, ny, nz = marker_normal(q_marker)
    px, py, pz = p_marker
    return (px + nx * offset, py + ny * offset, pz + nz * offset)


def transform_pose(translation, rotation, position, orientation):
    """
    TF 변환을 포즈에 적용한다.

        p_out = t + R(q_t) * p_in
        q_out = q_t * q_in

    tf2_geometry_msgs.do_transform_pose 를 쓰지 않고 직접 구현한 이유는
    그 함수의 시그니처(Pose 냐 PoseStamped 냐)가 ROS 배포판마다 달라
    이식성 문제가 있기 때문이다.
    """
    rotated = quat_rotate_vector(rotation, position)
    out_position = (
        translation[0] + rotated[0],
        translation[1] + rotated[1],
        translation[2] + rotated[2],
    )
    out_orientation = quat_normalize(quat_multiply(rotation, orientation))
    return out_position, out_orientation


def quat_from_rpy(roll, pitch, yaw):
    """
    고정축 roll-pitch-yaw (X-Y-Z 순서) 를 쿼터니언 (x, y, z, w) 로 변환한다.

    URDF 의 <origin rpy="..."> 와 같은 규약이다.
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_inverse(q):
    """단위 쿼터니언의 역 (= 켤레). 입력/출력 (x, y, z, w)."""
    x, y, z, w = quat_normalize(q)
    return (-x, -y, -z, w)


def orientation_candidates(base_q, tilts, azimuths, yaw_offsets=(0.0,)):
    """
    base_q 를 기준으로 자세 후보를 '편차 작은 순'으로 나열한다.

      - tilts [rad]       : base_q 의 +Z 축을 기울이는 각. [0, ...] 로 시작.
      - azimuths [rad]    : tilt 방향 (tilt=0 이면 무시).
      - yaw_offsets [rad] : +Z 축 둘레 회전.

    반복 순서가 곧 선호도다 (tilt 작은 것 먼저). 같은 회전(q, -q)/중복은 거른다.
    반환은 (x, y, z, w) 쿼터니언 리스트.
    """
    out = []
    seen = set()
    for tilt in tilts:
        az_list = (0.0,) if abs(tilt) < 1e-9 else azimuths
        for az in az_list:
            # Rz(az) · Rx(tilt) · Rz(-az) : XY 평면에서 az 방향 축 둘레로 tilt.
            tilt_q = quat_multiply(
                quat_from_rpy(0.0, 0.0, az),
                quat_multiply(quat_from_rpy(tilt, 0.0, 0.0),
                              quat_from_rpy(0.0, 0.0, -az)))
            for yaw in yaw_offsets:
                q = quat_normalize(quat_multiply(
                    base_q, quat_multiply(tilt_q, quat_from_rpy(0.0, 0.0, yaw))))
                key = tuple(round(v, 4) for v in q)
                neg = tuple(round(-v, 4) for v in q)
                if key in seen or neg in seen:
                    continue
                seen.add(key)
                out.append(q)
    return out


def grasp_candidates(q_marker, yaw_offsets, tilts, azimuths):
    """
    파지 자세 후보. orientation_candidates 를 grasp_orientation(q_marker)
    (= R_marker · Rx(180°)) 기준으로 부른 것.
    """
    return orientation_candidates(
        grasp_orientation(q_marker), tilts, azimuths, yaw_offsets)


def yaw_of(q):
    """
    자세 q 의 yaw [rad] — 프레임 X 축을 수평면(월드 XY)에 투영한 방향.

    마커의 pitch/roll (검출 뒤집힘 포함)에 둔감하다. "마커는 수평" 가정에서
    yaw 만 뽑아 쓸 때 사용한다. X 축 투영이 0 에 가까우면(마커가 거의 수직)
    0 을 돌려준다.
    """
    x, y, _ = quat_rotate_vector(quat_normalize(q), (1.0, 0.0, 0.0))
    if x * x + y * y < 1e-12:
        return 0.0
    return math.atan2(y, x)
