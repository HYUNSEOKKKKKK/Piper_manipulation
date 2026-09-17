"""
piper_pnp.geometry 단위 테스트.

ROS 없이 돌아간다 — 로컬 PC에서 `conda activate piper_pnp && pytest` 로 검증 가능.
"""

import math

import pytest

from piper_pnp.geometry import (
    quat_from_rpy,
    approach_position,
    grasp_candidates,
    grasp_orientation,
    marker_normal,
    orientation_candidates,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
    transform_pose,
    yaw_of,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def approx(actual, expected, tol=1e-9):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, abs=tol)


def quat_about_axis(axis, angle):
    half = angle / 2.0
    s = math.sin(half)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def test_quat_multiply_identity():
    q = quat_about_axis((0, 0, 1), 0.7)
    approx(quat_multiply(q, IDENTITY), q)
    approx(quat_multiply(IDENTITY, q), q)


def test_quat_rotate_vector_yaw_90():
    q = quat_about_axis((0, 0, 1), math.pi / 2)
    approx(quat_rotate_vector(q, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))


def test_quat_normalize_zero_returns_identity():
    approx(quat_normalize((0.0, 0.0, 0.0, 0.0)), IDENTITY)


def test_flat_marker_grasp_points_straight_down():
    """평면에 놓인 마커: 그리퍼 접근 축이 정확히 아래(-Z)를 향해야 한다."""
    z_axis = quat_rotate_vector(grasp_orientation(IDENTITY), (0.0, 0.0, 1.0))
    approx(z_axis, (0.0, 0.0, -1.0))


def test_yaw_only_does_not_tilt_the_gripper():
    """평면 마커를 yaw 로 아무리 돌려도 접근 방향은 계속 수직 하향이어야 한다."""
    for yaw in (0.3, 1.0, math.pi / 2, -2.0):
        q = quat_about_axis((0, 0, 1), yaw)
        z_axis = quat_rotate_vector(grasp_orientation(q), (0.0, 0.0, 1.0))
        approx(z_axis, (0.0, 0.0, -1.0))


def test_tilted_marker_tilts_the_gripper():
    """기울어진 마커: 그리퍼 접근 축이 마커 법선의 정반대여야 한다."""
    q = quat_about_axis((0, 1, 0), math.radians(30))
    normal = marker_normal(q)
    approx(normal, (0.5, 0.0, math.cos(math.radians(30))))

    z_axis = quat_rotate_vector(grasp_orientation(q), (0.0, 0.0, 1.0))
    approx(z_axis, (-normal[0], -normal[1], -normal[2]))


def test_approach_offset_on_flat_marker_is_pure_z():
    """평면 마커에서는 접근 지점이 명세의 '타겟 Z축 위 10cm' 와 일치한다."""
    approx(approach_position((0.3, 0.0, 0.05), IDENTITY, 0.10), (0.3, 0.0, 0.15))


def test_approach_offset_follows_the_normal_when_tilted():
    q = quat_about_axis((0, 1, 0), math.radians(30))
    result = approach_position((0.3, 0.0, 0.05), q, 0.10)
    expected = (0.3 + 0.05, 0.0, 0.05 + 0.10 * math.cos(math.radians(30)))
    approx(result, expected)


def test_approach_distance_is_exactly_the_offset():
    q = quat_about_axis((1, 1, 0), 0.4)
    origin = (0.2, -0.1, 0.3)
    result = approach_position(origin, q, 0.10)
    dist = math.dist(result, origin)
    assert dist == pytest.approx(0.10, abs=1e-12)


def test_transform_pose_translation_and_rotation():
    """yaw 90도 + 평행이동 변환을 포즈에 적용."""
    q_tf = quat_about_axis((0, 0, 1), math.pi / 2)
    position, orientation = transform_pose((1.0, 2.0, 3.0), q_tf, (1.0, 0.0, 0.0), IDENTITY)
    approx(position, (1.0, 3.0, 3.0))
    approx(orientation, q_tf)


def test_transform_pose_identity_is_noop():
    position, orientation = transform_pose((0.0, 0.0, 0.0), IDENTITY, (1.0, 2.0, 3.0), IDENTITY)
    approx(position, (1.0, 2.0, 3.0))
    approx(orientation, IDENTITY)


def test_quat_from_rpy_identity():
    approx(quat_from_rpy(0.0, 0.0, 0.0), IDENTITY)


def test_quat_from_rpy_matches_axis_rotations():
    """각 축 단독 회전은 해당 축 쿼터니언과 같아야 한다."""
    for axis, index in (((1, 0, 0), 0), ((0, 1, 0), 1), ((0, 0, 1), 2)):
        angle = 0.7
        rpy = [0.0, 0.0, 0.0]
        rpy[index] = angle
        approx(quat_from_rpy(*rpy), quat_about_axis(axis, angle))


def test_quat_from_rpy_flat_marker_normal_is_up():
    """rpy 가 0 인 마커는 법선이 월드 +Z 를 향한다 (책상에 평평히 놓인 상태)."""
    approx(marker_normal(quat_from_rpy(0.0, 0.0, 0.0)), (0.0, 0.0, 1.0))


def test_quat_from_rpy_pitch_tilts_the_normal():
    """pitch 를 주면 법선이 그만큼 기운다."""
    q = quat_from_rpy(0.0, math.radians(30), 0.0)
    approx(marker_normal(q), (0.5, 0.0, math.cos(math.radians(30))))


def _axis_angle_deg(q_a, q_b):
    """두 자세의 접근축(+Z) 사이 각 [deg]."""
    a = quat_rotate_vector(q_a, (0.0, 0.0, 1.0))
    b = quat_rotate_vector(q_b, (0.0, 0.0, 1.0))
    dot = max(-1.0, min(1.0, sum(a[i] * b[i] for i in range(3))))
    return math.degrees(math.acos(dot))


def test_grasp_candidates_first_is_ideal_and_ordered_by_tilt():
    """첫 후보는 기준 파지 자세와 같고, 이후로 tilt 가 단조 증가한다."""
    mq = quat_from_rpy(0.0, 0.0, 0.3)  # 평평하지만 yaw 있는 마커
    yaws = [0.0, math.radians(90)]
    tilts = [0.0, math.radians(15), math.radians(30), math.radians(45)]
    az = [0.0, math.radians(180), math.radians(90), math.radians(270)]
    cands = grasp_candidates(mq, yaws, tilts, az)

    base = grasp_orientation(mq)
    assert _axis_angle_deg(cands[0], base) < 1e-6          # 첫 후보 = 이상적(수직)
    seen_tilts = [_axis_angle_deg(q, base) for q in cands]
    # tilt=0 후보들 먼저, 그다음 15, 30, 45. 비내림차순이어야 한다 (FP 여유).
    for a, b in zip(seen_tilts, seen_tilts[1:]):
        assert a <= b + 1e-6
    assert max(seen_tilts) <= 45.0 + 1e-6


def test_grasp_candidates_dedup_and_yaw_preserves_axis():
    """중복 회전은 제거되고, yaw 만 다른 후보는 접근축이 같다."""
    mq = quat_from_rpy(0.0, 0.0, 0.0)
    cands = grasp_candidates(mq, [0.0, math.radians(90)], [0.0], [0.0])
    assert len(cands) == 2
    assert _axis_angle_deg(cands[0], cands[1]) < 1e-6      # yaw 만 달라 축은 동일
    # 서로 다른 회전이어야 한다 (yaw 90° 차이).
    assert any(abs(cands[0][i] - cands[1][i]) > 1e-3 for i in range(4))


def test_quat_inverse_undoes_rotation():
    q = quat_from_rpy(0.3, -0.5, 1.2)
    approx(quat_normalize(quat_multiply(q, quat_inverse(q))), IDENTITY, tol=1e-9)


def test_orientation_candidates_base_is_first_and_axis_ordered():
    base = quat_from_rpy(0.1, 0.2, 0.3)
    cands = orientation_candidates(
        base, [0.0, math.radians(20), math.radians(40)],
        [0.0, math.radians(90)], yaw_offsets=(0.0,))
    assert _axis_angle_deg(cands[0], base) < 1e-6
    tilts = [_axis_angle_deg(q, base) for q in cands]
    for a, b in zip(tilts, tilts[1:]):
        assert a <= b + 1e-6


def test_yaw_of_matches_input_and_ignores_flip():
    for deg in (0, 30, 90, -45, 179):
        assert math.degrees(yaw_of(quat_from_rpy(0.0, 0.0, math.radians(deg)))) == pytest.approx(deg, abs=1e-6)
    # 마커가 X축 180° 로 뒤집혀 들어와도 yaw 는 유지돼야 한다.
    flipped = quat_from_rpy(math.pi, 0.0, math.radians(30))
    assert math.degrees(yaw_of(flipped)) == pytest.approx(30.0, abs=1e-6)
