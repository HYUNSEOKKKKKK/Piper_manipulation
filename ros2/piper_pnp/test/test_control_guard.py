import math
import pytest
from piper_pnp.control_guard import CommandGuard


def ready(gripper=False):
    names = ['joint1', 'joint2'] + (['gripper'] if gripper else [])
    guard = CommandGuard(names)
    values = [0.4, -0.2] + ([0.04] if gripper else [])
    assert guard.record('feedback', names, values, 10.0)
    assert guard.record('command', names, values, 10.0)
    return guard


def test_initial_commands_never_enable_automatically():
    guard = ready()
    assert not guard.enabled
    guard.open(10.1)
    assert guard.enabled


def test_mock_zero_is_rejected_when_robot_is_elsewhere():
    guard = ready()
    guard.record('command', ['joint1', 'joint2'], [0., 0.], 10.1)
    with pytest.raises(ValueError, match='synchronized'):
        guard.open(10.1)
    assert not guard.enabled


@pytest.mark.parametrize('kind', ['feedback', 'command'])
def test_stale_stream_cannot_open_gate(kind):
    guard = ready()
    setattr(guard, kind+'_at', 9.0)
    with pytest.raises(ValueError, match='stale'):
        guard.open(10.1)


def test_running_guard_detects_dropout_and_close_requires_resync():
    guard = ready()
    guard.open(10.1)
    assert guard.reason(10.3)
    guard.close()
    assert not guard.enabled
    with pytest.raises(ValueError):
        guard.open(10.3)


@pytest.mark.parametrize('names,values', [
    (['joint1'], [0.]), (['joint1','joint1'], [0.,0.]),
    (['joint1','joint2'], [0.]), (['joint1','joint2'], [math.nan,0.]),
    (['joint1','joint2'], [math.inf,0.]),
])
def test_malformed_observations_do_not_refresh_timestamp(names, values):
    guard = ready()
    assert not guard.record('feedback', names, values, 11.)
    assert guard.feedback_at == 10.


def test_gripper_alignment_uses_metres():
    guard = ready(gripper=True)
    guard.command['gripper'] += .01
    with pytest.raises(ValueError, match='gripper'):
        guard.open(10.1)


def test_joint_order_is_explicit():
    guard = ready()
    assert guard.record('command', ['joint2','joint1'], [-.2,.4], 10.)
    guard.open(10.1)
