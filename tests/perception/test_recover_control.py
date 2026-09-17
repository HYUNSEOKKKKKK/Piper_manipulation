"""Exercise recovery command sequencing without creating ROS nodes or sending commands."""
from collections import deque
import copy
import time
from types import SimpleNamespace as NS
import unittest

from recover_control import ControlRecovery


ENDED = dict(ctrl_mode=2, teach_status=2, arm_status=0, err_status=0,
             joints_rad=[0.0] * 6, joint_span_rad=0.0)
PC = dict(ENDED, ctrl_mode=1)


class FakeRecovery(ControlRecovery):
    def __init__(self, snapshots=None, active=False, gate=True, params=None):
        # No Node init: these tests cannot access any ROS transport or robot.
        self.report = {}
        self.actions = {'arm': [2] if active else [4]}
        self.samples = copy.deepcopy(snapshots or [ENDED, ENDED, PC])
        self.commands = []
        self.gate_success = gate
        self.params = params if params is not None else [False, True, True, False]

    def spin(self, seconds):
        pass

    def snapshot(self):
        return self.samples.pop(0)

    def call(self, kind, name, request):
        if name.endswith('/get_parameters'):
            return NS(values=[NS(type=1, bool_value=v) for v in self.params])
        self.commands.append((name, getattr(request, 'data', None)))
        return NS(success=self.gate_success, message='test gate result')


class RecoveryTests(unittest.TestCase):
    def test_ended_teaching_closes_gate_then_holds(self):
        node = FakeRecovery()
        self.assertIn('복구 완료', node.recover())
        self.assertEqual(node.commands, [('/control_enable', False), ('/emergency_stop', None)])
        self.assertEqual(node.report['after']['ctrl_mode'], 1)

    def test_already_pc_never_interrupts_existing_motion(self):
        node = FakeRecovery([PC], active=True)
        self.assertIn('이미 PC', node.recover())
        self.assertEqual(node.commands, [])

    def test_invalid_states_never_send_commands(self):
        for changes in ({'teach_status': 1}, {'teach_status': 3}, {'arm_status': 1},
                        {'err_status': 1}, {'joint_span_rad': 0.02}, {'ctrl_mode': 0}):
            with self.subTest(changes=changes):
                node = FakeRecovery([dict(ENDED, **changes)])
                with self.assertRaises(RuntimeError):
                    node.recover()
                self.assertEqual(node.commands, [])

    def test_active_trajectory_refuses_recovery(self):
        node = FakeRecovery(active=True)
        with self.assertRaisesRegex(RuntimeError, '실행 중인 동작'):
            node.recover()
        self.assertEqual(node.commands, [])

    def test_guarded_real_feed_required(self):
        for params in ([True, True, True, False], [False, False, True, False],
                       [False, True, False, False], [False, True, True, True]):
            node = FakeRecovery(params=params)
            with self.assertRaises(RuntimeError):
                node.recover()
            self.assertEqual(node.commands, [])

    def test_gate_failure_never_sends_hold(self):
        node = FakeRecovery(gate=False)
        with self.assertRaisesRegex(RuntimeError, '게이트'):
            node.recover()
        self.assertEqual(node.commands, [('/control_enable', False)])

    def test_new_motion_after_gate_close_never_sends_hold(self):
        node = FakeRecovery([ENDED, dict(ENDED, joint_span_rad=0.02)])
        with self.assertRaises(RuntimeError):
            node.recover()
        self.assertEqual(node.commands, [('/control_enable', False)])

    def test_unsuccessful_transition_is_reported(self):
        node = FakeRecovery([ENDED, ENDED, ENDED])
        with self.assertRaisesRegex(RuntimeError, '복구를 확인하지 못했습니다'):
            node.recover()

    def test_excessive_joint_change_is_reported(self):
        node = FakeRecovery([ENDED, ENDED, dict(PC, joints_rad=[0.1] * 6)])
        with self.assertRaisesRegex(RuntimeError, '자세 유지 범위'):
            node.recover()

    def test_fresh_finite_feedback_required(self):
        now = time.monotonic()
        node = FakeRecovery()
        node.states = deque([(now, NS(**{k: ENDED[k] for k in
                                      ('ctrl_mode', 'teach_status', 'arm_status', 'err_status')}))])
        node.joints = deque([(now, [0.] * 6) for _ in range(12)])
        self.assertEqual(ControlRecovery.snapshot(node)['joint_span_rad'], 0.)
        for samples in ([], [(now - 1, [0.] * 6)] * 12,
                        [(now, [float('nan')] * 6)] * 12, [(now, [0.] * 6)]):
            node.joints = deque(samples)
            with self.assertRaises(RuntimeError):
                ControlRecovery.snapshot(node)


if __name__ == '__main__':
    unittest.main()
