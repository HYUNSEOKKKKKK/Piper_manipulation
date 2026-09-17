"""No hardware: prevent a release that traps the opened gripper beside a prior box."""
from types import SimpleNamespace
import time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_pnp.control_guard import CommandGuard
from piper_pnp.moveit_client import make_start_state, trajectory_end_state
from piper_pnp.release_checks import ReleaseChecksMixin


def positions(state):
    return dict(zip(state.joint_state.name, state.joint_state.position))


def trajectory(end):
    path = RobotTrajectory()
    path.joint_trajectory.joint_names = ['joint1']
    path.joint_trajectory.points = [JointTrajectoryPoint(positions=[float(end)])]
    return path


class Harness(ReleaseChecksMixin):
    def __init__(self, invalid=lambda q: False, retreat_ok=True):
        self.gripper_joint, self.gripper_open_position = 'gripper', .095
        self.cartesian_max_step = .005
        self.aborted = self.preview_only = False
        self.guard_real_commands = self.use_real_gripper = True
        self.checked, self.planned, self.stops, self._report = [], [], [], []
        self._command_guard = CommandGuard(['joint1', 'gripper'])
        self._command_guard.record('feedback', ['joint1', 'gripper'], [.2, .03], time.monotonic())
        def check(state):
            q = positions(state)
            self.checked.append(q)
            return not invalid(q)
        def plan(waypoints, **kw):
            assert kw['plan_only'] is True
            assert kw['min_fraction'] == 1.
            q = positions(kw['start_state'])
            self.planned.append(q)
            return (retreat_ok if q['gripper'] == .095 else True), trajectory(.4)
        self.moveit = SimpleNamespace(check_state_validity=check, move_cartesian=plan)

    def get_logger(self):
        return SimpleNamespace(info=lambda *a: None, error=lambda *a: None)

    def emergency_stop(self, reason):
        self.stops.append(reason)
        self.aborted = True


def test_arm_trajectory_chaining_keeps_hypothetical_open_gripper():
    state = make_start_state(['joint1','gripper'], [.1,.095])
    end = trajectory_end_state(trajectory(.4), state)
    assert positions(end) == {'joint1': .4, 'gripper': .095}
    assert positions(state) == {'joint1': .1, 'gripper': .095}


def test_closed_descent_open_retreat_and_intermediate_widths_are_checked():
    c = Harness()
    assert c._check_place_release(Pose(), Pose())
    assert c.planned == [{'joint1': .2, 'gripper': .03}, {'joint1': .4, 'gripper': .095}]
    assert all(q['joint1'] == .4 for q in c.checked)
    widths = sorted(q['gripper'] for q in c.checked)
    assert widths[0] == .03 and widths[-1] == .095
    assert max(b-a for a,b in zip(widths,widths[1:])) <= .002
    assert not c.stops


def test_prior_box_blocks_opening_before_any_retreat_execution():
    c = Harness(invalid=lambda q: q['gripper'] >= .09)
    assert not c._check_place_release(Pose(), Pose())
    assert len(c.planned) == 1  # only a planned descent; no commands at all
    assert c.stops and c._report[-1][1] is False


def test_open_endpoint_clear_does_not_hide_intermediate_finger_collision():
    c = Harness(invalid=lambda q: .059 < q['gripper'] < .063)
    assert not c._check_place_release(Pose())
    assert c.checked[0]['gripper'] == .095
    assert not c.planned and c.stops


def test_clear_open_state_but_no_cartesian_exit_blocks_release():
    c = Harness(retreat_ok=False)
    assert not c._check_place_release(Pose())
    assert c.planned == [{'joint1': .2, 'gripper': .095}]
    assert c.stops


def test_candidate_rejects_blocked_camera_return_with_gripper_open():
    c = Harness()
    c.arm_joints, c.camera_ready_pose = ['joint1'], [.1]
    returns = []
    def camera(joints, goal, **kwargs):
        assert kwargs['plan_only'] is True
        returns.append(positions(kwargs['start_state']))
        return False, None
    c.moveit.move_to_joints = camera
    assert not c._plan_release_clearance(c._release_current_state(), Pose(), check_camera=True)
    assert returns == [{'joint1': .4, 'gripper': .095}]


def test_actual_contact_state_rechecked_instead_of_nominal_descent_end():
    c = Harness(invalid=lambda q: q['joint1'] > .45 and q['gripper'] > .09)
    assert c._check_place_release(Pose(), Pose())
    c._command_guard.record('feedback', ['joint1','gripper'], [.5,.03], time.monotonic())
    assert not c._check_place_release(Pose())
    assert c.checked[-1] == {'joint1': .5, 'gripper': .095}


def test_stale_or_missing_gripper_feedback_blocks_release():
    c = Harness()
    c._command_guard.feedback_at -= 1.
    assert not c._check_place_release(Pose())
    assert not c.planned and not c.checked


def test_abort_during_width_checks_does_not_request_retreat():
    c = Harness()
    def check(state):
        c.aborted = True
        return True
    c.moveit.check_state_validity = check
    assert not c._check_place_release(Pose())
    assert not c.planned


def test_collision_at_retreat_start_never_uses_joint_space_fallback():
    from piper_pnp.piper_pnp_controller import PiperPnpController
    c = Harness(invalid=lambda q: True)
    c._prefix = lambda: ''
    c._preview_state = None
    c.cartesian_min_fraction = 1.
    c.moveit.move_cartesian = lambda *a,**k: (False,None)
    def unexpected(*a,**k):
        raise AssertionError('joint-space fallback from a colliding start')
    c.moveit.move_to_pose = unexpected
    c._after_motion = lambda label,ok,path: ok
    assert not PiperPnpController._goto_retreat(c,'retreat',Pose())
