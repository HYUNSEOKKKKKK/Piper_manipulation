"""No hardware: repeated full cycles, fresh perception, occupancy and stop semantics."""
import json
import threading
import time
from types import SimpleNamespace, MethodType

import pytest
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger

from piper_pnp.batch_workspace import BatchWorkspace, BatchProgress
from piper_pnp.batch_control import BatchControlMixin
from piper_pnp.piper_pnp_controller import PiperPnpController as Controller
from test_integration import target_controller, observe


@pytest.fixture
def config():
    return dict(object_dims_m=[.077, .035, .030], pick_bounds_xy=[.2,.6,0.,.3],
                place_positions_m=[[.35,-.12,.02],[.47,-.12,.02]],
                place_exclusion_radius_m=.08, placed_box_envelope_m=[.10,.10,.10],
                table_z_m=0.)


def test_workspace_filters_source_and_all_reserved_places(config):
    w = BatchWorkspace(config)
    assert w.allows_pick((.396,.042,.039))
    assert w.allows_pick((.41,.159,.033))
    assert not w.allows_pick((.35,-.12,.04))
    assert not w.allows_pick((.47,-.12,.04))
    assert not w.allows_pick((.7,.1,.04))
    assert not w.allows_pick((float('nan'),.1,.04))
    with pytest.raises(IndexError): w.place(2)
    with pytest.raises(IndexError): w.place(-1)


@pytest.mark.parametrize('key,value', [
    ('pick_bounds_xy',[.6,.2,0.,.3]), ('object_dims_m',[.077,0.,.03]),
    ('place_positions_m',[]), ('place_positions_m',[[.4,.1,.02]]),
    ('place_positions_m',[[.35,-.12,.02],[.40,-.12,.02]]),
    ('place_positions_m',[[.35,-.12,float('nan')]]),
    ('place_exclusion_radius_m',.01),
    ('operator_feed','true'),
])
def test_bad_workspace_is_rejected(config, key, value):
    config[key] = value
    with pytest.raises(ValueError): BatchWorkspace(config)


@pytest.mark.parametrize('continuous', [False,True])
def test_feed_parameter_loading_preserves_progress_and_console_mode(config,tmp_path,continuous):
    config['operator_feed'] = True
    path = tmp_path/'feed.json'
    path.write_text(json.dumps(config))
    workspace = BatchWorkspace(config)
    record = BatchProgress(path,workspace)
    record.reserve(); record.finish(); record.reserve()
    before = record.path.read_bytes()
    params = {}
    c = SimpleNamespace(
        declare_parameter=lambda name,default,*a: params.setdefault(name,default),
        get_parameter=lambda name: SimpleNamespace(value=params[name]),
        _target_is_sane=lambda *a: True)
    Controller._declare_parameters(c)
    params.update(batch_config_file=str(path),continuous_feed=continuous,
                  auto_start=False,loop_mode=True,guard_real_commands=True,
                  preview_only=False,use_real_gripper=True)
    Controller._read_parameters(c)
    assert c.continuous_feed is continuous
    assert params['batch_operator_feed'] is (not continuous)
    assert c._batch_workspace.digest == workspace.digest
    assert c._batch_completed == 1 and c._batch_progress.in_progress
    assert record.path.read_bytes() == before  # mode switching never clears occupancy/interruption


@pytest.mark.parametrize('batch', [False,True])
def test_continuous_feed_requires_feed_workspace(config,tmp_path,batch):
    path = tmp_path/'batch.json'
    path.write_text(json.dumps(config))
    params = {}
    c = SimpleNamespace(
        declare_parameter=lambda name,default,*a: params.setdefault(name,default),
        get_parameter=lambda name: SimpleNamespace(value=params[name]))
    Controller._declare_parameters(c)
    params.update(batch_config_file=str(path) if batch else '',continuous_feed=True)
    with pytest.raises(ValueError,match='continuous_feed'):
        Controller._read_parameters(c)


def test_progress_survives_restart_and_blocks_uncertain_slot(config, tmp_path):
    w = BatchWorkspace(config)
    path = tmp_path/'workspace.json'
    p = BatchProgress(path,w)
    p.reserve()
    with pytest.raises(ValueError): BatchProgress(path,w)
    p.finish()
    restored = BatchProgress(path,w)
    assert restored.completed == 1
    restored.reserve()
    restored.finish()
    assert BatchProgress(path,w).completed == 2
    with pytest.raises(ValueError): restored.reserve()
    config['place_positions_m'][1][0] = .50
    with pytest.raises(ValueError): BatchProgress(path,BatchWorkspace(config))


def test_release_check_failure_stops_before_descent_and_open(config, tmp_path):
    c = CycleHarness(config,tmp_path,fail_at='release_check')
    assert not c._run_cycle()
    assert [s for s,_ in c.calls if s in ('open','close')] == ['open','close']
    assert 'Place 타겟까지 직선 하강' not in [s for s,_ in c.calls]
    assert c._batch_progress.in_progress


def test_contact_alignment_failure_keeps_box_gripped(config, tmp_path):
    c = CycleHarness(config,tmp_path,fail_at='alignment_check')
    assert not c._run_cycle()
    assert 'Place 타겟까지 직선 하강' in [s for s,_ in c.calls]
    assert [s for s,_ in c.calls if s in ('open','close')] == ['open','close']
    assert c._batch_progress.in_progress


def batch_targets(config):
    c = target_controller()
    c._batch_workspace = BatchWorkspace(config)
    c._batch_completed = 0
    c._set_batch_place_locked = MethodType(BatchControlMixin._set_batch_place_locked,c)
    c._to_base_frame = lambda p,h: ((.4,.1,.04),(0.,0.,0.,1.))
    c._expire_targets_locked()
    return c


def test_controller_owns_destination_and_keeps_it_while_pick_expires(config):
    c = batch_targets(config)
    observe(c,'place',10_000_000_000)  # incoming bridge place cannot override it
    assert not c._target_samples['place']
    for i in range(8): observe(c,'pick',10_000_000_000+i*1_000_000)
    assert c._targets_ready.is_set()
    assert c._targets['place']['position'] == (.35,-.12,.02)
    c.now_ns += 1_000_000_000
    c._expire_targets_locked()
    assert not c._targets_ready.is_set()
    assert 'pick' not in c._targets and 'place' in c._targets
    c._batch_completed = 1
    c._expire_targets_locked()
    assert c._targets['place']['position'] == (.47,-.12,.02)


def test_controller_rejects_destination_as_pick(config):
    c = batch_targets(config)
    c._to_base_frame = lambda p,h: ((.35,-.12,.04),(0.,0.,0.,1.))
    for i in range(8): observe(c,'pick',10_000_000_000+i*1_000_000)
    assert not c._target_samples['pick']
    assert not c._targets_ready.is_set()


def wait_until(condition, timeout=2.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if condition(): return
        time.sleep(.005)
    assert condition(), 'condition did not arrive'


@pytest.mark.parametrize('continuous', [False, True])
def test_empty_scene_waits_then_only_post_camera_frames_can_start(config, continuous):
    config['operator_feed'] = True
    c = batch_targets(config)
    c.continuous_feed = continuous
    logs = []
    c.get_logger = lambda: SimpleNamespace(info=lambda *a,**k: logs.append(a[0]),
                                          error=lambda *a,**k: logs.append(a[0]))
    result = []
    thread = threading.Thread(target=lambda: result.append(c._wait_for_targets()))
    thread.start()
    try:
        wait_until(lambda: any('계속 대기' in s for s in logs))
        assert not result  # original single-cycle timeout elapsed; batch still waits
        for i in range(8): observe(c,'pick',10_000_000_000+i*1_000_000)
        assert not c._target_samples['pick']  # old frames from previous cycle
        c.now_ns += 100_000_000
        for i in range(8): observe(c,'pick',c.now_ns-20_000_000+i*1_000_000)
        thread.join(2.)
        assert result == [True]
        assert c._frozen
    finally:
        c.aborted = True
        thread.join(2.)


class CycleHarness(BatchControlMixin):
    _run_cycle = Controller._run_cycle
    _begin_step = Controller._begin_step
    _wait_for_confirmation = Controller._wait_for_confirmation
    _on_continue_request = Controller._on_continue_request
    aborted = Controller.aborted

    def __init__(self, config, tmp_path, fail_at=None):
        self._batch_workspace = BatchWorkspace(config)
        self._batch_progress = BatchProgress(tmp_path/'config.json',self._batch_workspace)
        self._batch_completed = 0
        self._abort = threading.Event()
        self._terminate = threading.Event()
        self._continue_event = threading.Event()
        self._waiting_confirmation = False
        self.step_confirm = self.preview_only = False
        self.step_confirm_timeout = 2.
        self.zero_pose = self.camera_ready_pose = [0.]*6
        self.grasp_offset, self.approach_offset = -.02,.1
        self.place_offset, self.place_approach_offset = .01,.06
        self.base_frame = 'base_link'
        self.calls, self.logs, self.stops = [], [], []
        self.fail_at = fail_at
        self.moveit = SimpleNamespace(add_collision_box=lambda *a: self.record('obstacle',a[0]))

    def get_logger(self):
        return SimpleNamespace(info=lambda *a,**k: self.logs.append(a[0]),
                               warn=lambda *a,**k: self.logs.append(a[0]),
                               error=lambda *a,**k: self.logs.append(a[0]))

    def record(self, label, data=None):
        self.calls.append((label,data))
        return label != self.fail_at

    def _check_batch_bridge(self): return self.record('bridge')
    def _reset_cycle_state(self): self.record('reset')
    def _prepare_real_control(self): return self.record('prepare')
    def _goto_joints(self,label,pose): return self.record(label)
    def open_gripper(self): return self.record('open')
    def close_gripper(self): return self.record('close')
    def _wait_for_targets(self): return self.record('fresh_observation', self._batch_completed)
    def _approach_ready(self,*args):
        self.record(args[0])
        return Pose(),Pose()
    def _goto_cartesian(self,label,pose): return self.record(label)
    def _goto_retreat(self,label,pose): return self.record(label)
    def _check_place_release(self,*args): return self.record('release_check')
    def _check_place_alignment(self,*args): return self.record('alignment_check')
    def print_report(self): pass
    def emergency_stop(self,reason,terminate=False):
        self.stops.append(reason)
        self._abort.set()
        self._continue_event.set()
        if terminate: self._terminate.set()


def start_batch(c):
    result = []
    thread = threading.Thread(target=lambda: result.append(c._run_batch()))
    thread.start()
    wait_until(lambda: c._waiting_confirmation or not thread.is_alive())
    return thread,result


def test_one_approval_runs_two_full_cycles_with_gripper_and_no_slot_reuse(config,tmp_path):
    c = CycleHarness(config,tmp_path)
    thread,result = start_batch(c)
    try:
        assert c.calls == [('bridge',None)]  # no preparation, arm or gripper before c
        response = c._on_continue_request(None,Trigger.Response())
        assert response.success
        wait_until(lambda: c._batch_completed == 2)
        assert thread.is_alive() and not result  # full capacity remains waiting
        assert [v for label,v in c.calls if label == 'fresh_observation'] == [0,1]
        assert [s for s,_ in c.calls if s in ('open','close')] == ['open','close','open']*2
        assert len([s for s in c.logs if s.startswith('--- Step')]) == 18
        assert len([s for s in c.logs if '[승인 대기]' in s]) == 1
        assert len([s for s,_ in c.calls if s == 'reset']) == 2
        assert len([s for s,_ in c.calls if s == 'obstacle']) == 2
        assert not c._on_continue_request(None,Trigger.Response()).success
        assert BatchProgress(tmp_path/'config.json',c._batch_workspace).completed == 2
    finally:
        c._terminate.set()
        thread.join(2.)
    assert result == [True]


def test_feed_waits_at_home_until_next_box_approval(config,tmp_path):
    config['operator_feed'] = True
    c = CycleHarness(config,tmp_path)
    c.step_confirm_timeout = .02  # refill wait must not expire after a step timeout
    thread,result = start_batch(c)
    try:
        time.sleep(.05)
        assert not c.aborted and c.calls == [('bridge',None)]
        assert c._on_continue_request(None,Trigger.Response()).success
        wait_until(lambda: c._batch_completed == 1 and c._waiting_confirmation)
        before = list(c.calls)
        assert [s for s,_ in before if s != 'obstacle'][-1] == 'Zero Pose'
        time.sleep(.05)
        assert c.calls == before and not c.aborted
        assert [s for s,_ in before].count('fresh_observation') == 1
        assert c._on_continue_request(None,Trigger.Response()).success
        wait_until(lambda: c._batch_completed == 2)
        assert len([s for s in c.logs if s.startswith('--- Step')]) == 18
        assert len([s for s in c.logs if '[승인 대기]' in s]) == 2
        assert not c._on_continue_request(None,Trigger.Response()).success
    finally:
        c.emergency_stop('test',terminate=True)
        thread.join(2.)


def test_continuous_feed_one_c_three_boxes_only_initial_home(config,tmp_path):
    config['operator_feed'] = True
    config['place_positions_m'].append([.35,-.24,.02])
    c = CycleHarness(config,tmp_path)
    c.continuous_feed = True
    c.step_confirm_timeout = .02
    thread,result = start_batch(c)
    try:
        time.sleep(.05)
        assert c.calls == [('bridge',None)] and not c.aborted
        reply = c._on_continue_request(None,Trigger.Response())
        assert reply.success and '연속 이송 시작' in reply.message
        wait_until(lambda: any('모두 사용' in s for s in c.logs))
        labels = [label for label,_ in c.calls]
        assert c._batch_completed == 3 and thread.is_alive()
        assert labels.count('Zero Pose') == labels.count('prepare') == 1
        assert [v for label,v in c.calls if label=='fresh_observation'] == [0,1,2]
        # Original feed's complete pick/place motion sequence in every cycle.
        motion = ['Pick-Ready Pose','Pick 타겟까지 직선 하강','close','직선 상승',
                  'Camera-Ready Pose','Place-Ready Pose','release_check',
                  'Place 타겟까지 직선 하강','alignment_check','release_check',
                  'open','직선 상승','Camera-Ready Pose','obstacle']
        starts = [i for i,label in enumerate(labels) if label=='fresh_observation']
        for start in starts:
            assert labels[start+1:start+1+len(motion)] == motion
        assert labels.count('Camera-Ready Pose') == 1+2*3
        assert [label for label in labels if label in ('open','close')] == ['open']+['close','open']*3
        assert len([s for s in c.logs if '[승인 대기]' in s]) == 1
        assert labels[-2:] == ['Camera-Ready Pose','obstacle']
        assert not c._on_continue_request(None,Trigger.Response()).success
        restored = BatchProgress(tmp_path/'config.json',c._batch_workspace)
        assert restored.completed == 3 and not restored.in_progress
        before = list(c.calls)
        time.sleep(.03)
        assert c.calls == before  # capacity cannot recycle a used place
    finally:
        c._terminate.set()
        thread.join(2.)
    assert result == [True]


@pytest.mark.parametrize('failure', ['Pick 타겟까지 직선 하강','close',
                                   'Place 타겟까지 직선 하강','alignment_check',
                                   'Camera-Ready Pose','obstacle'])
def test_continuous_feed_failure_does_not_retry_or_home(config,tmp_path,failure):
    config['operator_feed'] = True
    c = CycleHarness(config,tmp_path)
    c.continuous_feed = True
    record = c.record
    c.record = lambda label,data=None: (record(label,data) and
                                      not (c._batch_completed == 1 and label == failure))
    thread,result = start_batch(c)
    c._on_continue_request(None,Trigger.Response())
    thread.join(2.)
    assert not thread.is_alive() and result == [False]
    assert c._batch_completed == 1 and c.stops
    assert [s for s,_ in c.calls].count('Zero Pose') == 1
    assert [v for s,v in c.calls if s=='fresh_observation'] == [0,1]
    assert c._batch_progress.in_progress
    assert not c._on_continue_request(None,Trigger.Response()).success
    restored = BatchProgress(tmp_path/'config.json',c._batch_workspace,allow_interrupted=True)
    assert restored.completed == 1 and restored.in_progress


def test_continuous_feed_empty_scene_wait_can_be_stopped_without_reserving(config,tmp_path):
    config['operator_feed'] = True
    c = CycleHarness(config,tmp_path)
    c.continuous_feed = True
    waiting = threading.Event()
    def observe_next():
        c.record('fresh_observation',c._batch_completed)
        if c._batch_completed == 0:
            return True
        waiting.set()
        c._abort.wait(2.)
        return False
    c._wait_for_targets = observe_next
    thread,result = start_batch(c)
    try:
        c._on_continue_request(None,Trigger.Response())
        assert waiting.wait(2.)
        before = list(c.calls)
        time.sleep(.03)
        assert c.calls == before
        assert c._batch_completed == 1 and not c._batch_progress.in_progress
        assert not c._on_continue_request(None,Trigger.Response()).success
    finally:
        c.emergency_stop('operator',terminate=True)
        thread.join(2.)
    assert result == [False]
    assert c.calls == before  # no home, close, or other movement on stop
    assert [s for s,_ in c.calls].count('Zero Pose') == 1


def test_continuous_feed_respects_existing_slots_and_starts_with_home(config,tmp_path):
    config['operator_feed'] = True
    c = CycleHarness(config,tmp_path)
    c.continuous_feed = True
    c._batch_progress.reserve()
    c._batch_progress.finish()
    c._batch_completed = 1
    thread,result = start_batch(c)
    try:
        assert [s for s,_ in c.calls] == ['bridge','obstacle']
        c._on_continue_request(None,Trigger.Response())
        wait_until(lambda: any('모두 사용' in s for s in c.logs))
        assert [v for s,v in c.calls if s=='fresh_observation'] == [1]
        assert [s for s,_ in c.calls].count('Zero Pose') == 1
        assert c._batch_completed == 2
        assert len({v for s,v in c.calls if s=='obstacle'}) == 2
    finally:
        c._terminate.set()
        thread.join(2.)
    assert result == [True]


def test_continuous_feed_stop_before_c_never_moves(config,tmp_path):
    config['operator_feed'] = True
    c = CycleHarness(config,tmp_path)
    c.continuous_feed = True
    thread,result = start_batch(c)
    c.emergency_stop('operator',terminate=True)
    thread.join(2.)
    assert result == [False] and c.calls == [('bridge',None)]


def test_feed_stop_while_waiting_for_refill_prevents_next_cycle(config,tmp_path):
    config['operator_feed'] = True
    c = CycleHarness(config,tmp_path)
    thread,result = start_batch(c)
    c._on_continue_request(None,Trigger.Response())
    wait_until(lambda: c._batch_completed == 1 and c._waiting_confirmation)
    before = list(c.calls)
    c.emergency_stop('operator',terminate=True)
    thread.join(2.)
    assert result == [False]
    assert c.calls == before
    assert not c._on_continue_request(None,Trigger.Response()).success
    assert BatchProgress(tmp_path/'config.json',c._batch_workspace).completed == 1


@pytest.mark.parametrize('failure',['Pick 타겟까지 직선 하강','close','Place 타겟까지 직선 하강'])
def test_failed_cycle_stops_without_retry_or_home(config,tmp_path,failure):
    c = CycleHarness(config,tmp_path,fail_at=failure)
    thread,result = start_batch(c)
    c._on_continue_request(None,Trigger.Response())
    thread.join(2.)
    assert not thread.is_alive() and result == [False]
    assert c._batch_completed == 0
    assert c.stops
    assert [s for s,_ in c.calls].count('Zero Pose') == 1  # startup only; no error recovery motion
    assert [s for s,_ in c.calls].count('fresh_observation') == 1
    with pytest.raises(ValueError): BatchProgress(tmp_path/'config.json',c._batch_workspace)


def test_stop_during_initial_approval_never_moves_and_r_cannot_resume(config,tmp_path):
    c = CycleHarness(config,tmp_path)
    thread,result = start_batch(c)
    c.emergency_stop('operator',terminate=True)
    thread.join(2.)
    assert result == [False] and c.calls == [('bridge',None)]
    assert not Controller.resume(c)
    assert c.aborted


@pytest.mark.parametrize('color,digest,expected', [('any','match',True),
    ('yellow','match',False),('any','wrong',False)])
def test_mismatched_bridge_cannot_start(config,tmp_path,color,digest,expected):
    c = CycleHarness(config,tmp_path)
    c._cb_group = None
    c.create_client = lambda *a,**k: SimpleNamespace(
        wait_for_service=lambda **k: True, call_async=lambda r: r)
    c.destroy_client = lambda client: None
    c._await_control_future = lambda *a: SimpleNamespace(values=[
        SimpleNamespace(type=4,string_value=c._batch_workspace.digest if digest=='match' else digest),
        SimpleNamespace(type=4,string_value=color),
        SimpleNamespace(type=8,double_array_value=c._batch_workspace.dims)])
    assert BatchControlMixin._check_batch_bridge(c) is expected
