"""Box alignment must survive different pick headings, jaw choices and tilt."""
import math
import threading
from types import MethodType, SimpleNamespace

import pytest
from geometry_msgs.msg import Pose

from piper_pnp.geometry import quat_from_rpy, quat_multiply, quat_rotate_vector
from piper_pnp.place_alignment import (
    aligned_place_candidates, held_box_orientation, long_axis_error,
)
from piper_pnp.piper_pnp_controller import PiperPnpController as Controller


def qdeg(roll=0., pitch=0., yaw=0.):
    return quat_from_rpy(*map(math.radians, (roll, pitch, yaw)))


@pytest.mark.parametrize('pick_yaw', [-73., 0., 38., 149.])
@pytest.mark.parametrize('jaw_yaw', [0., 90., 180., 270.])
@pytest.mark.parametrize('target_yaw', [0., 27.])
def test_all_candidates_keep_box_long_edge_heading_for_any_grasp(pick_yaw, jaw_yaw, target_yaw):
    # A non-level perception frame and tilted grasp must both be compensated.
    pick = qdeg(7., -11., pick_yaw)
    tool = quat_multiply(pick, quat_multiply(qdeg(180., 15.), qdeg(yaw=jaw_yaw)))
    target = qdeg(yaw=target_yaw)
    candidates = aligned_place_candidates(
        pick, tool, target, [0., math.radians(15), math.radians(30)],
        [0., math.pi, math.pi/2, 3*math.pi/2, math.pi/4])
    assert len(candidates) > 2
    for i, candidate in enumerate(candidates):
        box = held_box_orientation(candidate, tool, pick)
        assert long_axis_error(box, target) < 1e-7
        if i < 2:  # Try both level 180-degree-equivalent box poses first.
            assert quat_rotate_vector(box, (0., 0., 1.)) == pytest.approx((0., 0., 1.))


def test_gripper_heading_can_differ_90_degrees_while_boxes_are_aligned():
    pick = qdeg(yaw=38.)
    tools = [quat_multiply(pick, qdeg(180., 0., yaw)) for yaw in (0., 90.)]
    placed = [aligned_place_candidates(pick, tool, qdeg(), [0.], [0.])[0] for tool in tools]
    axes = [quat_rotate_vector(q, (1., 0., 0.)) for q in placed]
    assert abs(sum(a*b for a,b in zip(*axes))) < 1e-8
    for q, tool in zip(placed, tools):
        assert quat_rotate_vector(held_box_orientation(q, tool, pick), (1., 0., 0.)) == pytest.approx((1., 0., 0.))


def test_180_degree_symmetry_is_accepted_but_90_and_vertical_are_rejected():
    assert long_axis_error(qdeg(yaw=180.), qdeg()) == pytest.approx(0.)
    assert long_axis_error(qdeg(yaw=90.), qdeg()) == pytest.approx(math.pi/2)
    assert math.isinf(long_axis_error(qdeg(pitch=90.), qdeg()))


def harness():
    c = SimpleNamespace(
        _targets_lock=threading.Lock(), _targets={'pick': {'orientation': qdeg()},
        'place': {'position': (.35, -.12, .02), 'orientation': qdeg()}}, _pick_grasp_q=qdeg(180.),
        place_match_pick_tilt=True, place_alignment_tolerance=math.radians(5.), guard_real_commands=True,
        grasp_tilt_candidates=[0., math.radians(15)], grasp_tilt_azimuths=[0., math.pi/2],
        place_tilt_azimuths=[0., math.pi/2],
        grasp_ik_check_timeout=.05, aborted=False, moves=[], errors=[])
    c.get_logger=lambda: SimpleNamespace(info=lambda *a: None, error=c.errors.append)
    c._axis_points_down=Controller._axis_points_down
    c._poses_along_axis=Controller._poses_along_axis
    for name in ('_choose_place_orientation', '_check_place_alignment', '_approach_ready_impl'):
        setattr(c, name, MethodType(getattr(Controller, name), c))
    c._goto_pose=lambda *a: c.moves.append(a) or True
    c._place_open_ik_clear=lambda *a: True
    c.moveit=SimpleNamespace(compute_ik=lambda *a, **kw: False)
    return c


def test_failed_aligned_ik_never_falls_back_to_free_grasp_or_moves():
    c = harness()
    # No grasp_ik_mode/ordinary candidate method: entering that path is a bug.
    assert c._approach_ready_impl('place', 'place', (.35, -.12, .02), qdeg(), .035, .05) == (None, None)
    assert not c.moves and c.errors


def test_missing_pick_rotation_blocks_place_without_fallback():
    c = harness()
    c._pick_grasp_q = None
    assert c._approach_ready_impl('place', 'place', (.35, -.12, .02), qdeg(), .035, .05) == (None, None)
    assert not c.moves


def test_open_gripper_collision_rejects_candidate_even_when_closed_ik_passes():
    c = harness()
    c.moveit.compute_ik=lambda *a, **kw: True
    c._place_open_ik_clear=lambda *a: False
    assert c._choose_place_orientation((.35, -.12, .02), qdeg(), .05, .035) is None


@pytest.mark.parametrize('yaw,expected', [(0., True), (180., True), (4., True), (8., False), (90., False)])
def test_actual_tool_feedback_checks_predicted_box_heading(yaw, expected):
    c = harness()
    p = Pose()
    q = quat_multiply(qdeg(yaw=yaw), c._pick_grasp_q)
    p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w = q
    c._achieved_ee_pose=lambda: p
    assert c._check_place_alignment() is expected


def test_missing_actual_tool_feedback_blocks_opening():
    c = harness()
    c._achieved_ee_pose=lambda: None
    assert not c._check_place_alignment()


def test_lowered_place_offset_adds_five_mm_without_changing_ready_pose():
    q = qdeg(165., 0., 25.)
    pre, old = Controller._poses_along_axis((.35, -.12, .02), q, .05, .04)
    pre_new, new = Controller._poses_along_axis((.35, -.12, .02), q, .05, .035)
    xyz = lambda p: (p.position.x,p.position.y,p.position.z)
    assert xyz(pre) == xyz(pre_new)
    assert math.dist(xyz(old), xyz(new)) == pytest.approx(.005)
    assert math.dist(xyz(pre_new), xyz(new)) == pytest.approx(.015)


def test_pick_skips_jaw_orientation_that_cannot_place_aligned():
    c = harness()
    c.moveit.compute_ik=lambda *a, **kw: True
    c.grasp_yaw_candidates=[0., math.pi/2]
    c.grasp_tilt_candidates=[0.]
    c.place_approach_offset, c.place_offset = .05, .035
    attempted = []
    def place(*args, pick_grasp_q, announce):
        attempted.append(pick_grasp_q)
        # First jaw choice cannot reach aligned destination; second can.
        return pick_grasp_q if len(attempted) == 2 else None
    c._choose_place_orientation=place
    q = Controller._choose_grasp_candidate(c, (.4, .12, .05), qdeg(), .03, -.025)
    assert len(attempted) == 2 and q == attempted[1]
    assert not c.moves


def test_no_joint_pick_place_solution_does_not_fall_back_to_tolerance():
    c = harness()
    c.grasp_ik_mode='candidates'
    c._choose_grasp_candidate=lambda *a: None
    assert c._approach_ready_impl('pick', 'pick', (.4, .12, .05), qdeg(), -.025, .03) == (None, None)
    assert not c.moves


def test_place_rejects_blocked_release_plan_then_tries_next_aligned_candidate():
    c = harness()
    c.moveit.compute_ik=lambda *a, **kw: True
    checked = []
    path = object()
    def plan(pre, contact):
        checked.append((pre, contact))
        return path if len(checked) == 2 else None
    c._plan_aligned_place_release=plan
    q = c._choose_place_orientation((.35, -.12, .02), qdeg(), .05, .035, plan_release=True)
    assert q is not None and len(checked) == 2
    assert c._place_ready_plan is path and not c.moves


@pytest.mark.parametrize('preview', [False, True])
def test_place_executes_exactly_the_checked_approach_trajectory(preview):
    c = harness()
    c.preview_only=preview
    path=object()
    c._place_ready_plan=path
    c._choose_place_orientation=lambda *a, **kw: qdeg(180.)
    executed=[]
    c.moveit._execute_trajectory=lambda trajectory, timeout: executed.append(trajectory) or True
    c._after_motion=lambda label, ok, trajectory: ok and trajectory is path
    c._align_grasp_to_achieved=lambda pre, contact: (pre, contact)
    pre, contact = c._approach_ready_impl('place', 'place', (.35, -.12, .02), qdeg(), .035, .05)
    assert pre is not None and contact is not None
    assert executed == ([] if preview else [path])
    assert not c.moves  # A fresh _goto_pose would select an unchecked IK branch.
