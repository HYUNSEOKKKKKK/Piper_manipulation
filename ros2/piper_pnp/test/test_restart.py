"""No hardware: confirmed reset, persistent interruption, stop races, late goals."""
from concurrent.futures import Future
from pathlib import Path
import json
import threading
import time
from types import SimpleNamespace

import pytest
from std_srvs.srv import SetBool

from piper_pnp.batch_workspace import BatchWorkspace, BatchProgress
from piper_pnp.console_input import ConsoleKeyState
from piper_pnp.control_guard import CommandGuard
from piper_pnp.moveit_client import MoveItClient
from piper_pnp.restart_control import RestartControlMixin
import piper_pnp.restart_control as restart_module
from test_integration import Log


def workspace():
    return BatchWorkspace(dict(object_dims_m=[.077,.035,.030],
        pick_bounds_xy=[.2,.6,0.,.3], place_positions_m=[[.35,-.12,.02],[.47,-.12,.02]],
        place_exclusion_radius_m=.08, placed_box_envelope_m=[.1,.1,.1],table_z_m=0.))


def test_interrupted_progress_can_load_for_reset_but_cannot_reserve(tmp_path):
    p = tmp_path/'workspace.json'
    w = workspace()
    old = BatchProgress(p,w)
    old.reserve()
    before = old.path.read_bytes()
    with pytest.raises(ValueError): BatchProgress(p,w)
    restored = BatchProgress(p,w,allow_interrupted=True)
    assert restored.in_progress
    assert restored.path.read_bytes() == before
    with pytest.raises(ValueError): restored.reserve()


@pytest.mark.parametrize('bad', [None,1,'true'])
def test_reset_loading_still_rejects_malformed_state(tmp_path,bad):
    p,w = tmp_path/'workspace.json',workspace()
    Path = type(p)
    Path(str(p)+'.state.json').write_text(json.dumps(dict(
        workspace_digest=w.digest,completed=0,in_progress=bad)))
    with pytest.raises(ValueError): BatchProgress(p,w,allow_interrupted=True)


def test_reset_backs_up_and_clears_all_slots(tmp_path):
    p,w = tmp_path/'workspace.json',workspace()
    record = BatchProgress(p,w)
    record.reserve(); record.finish(); record.reserve()
    before = record.path.read_bytes()
    record.reset_after_home()
    restored = BatchProgress(p,w)
    assert restored.completed == 0 and not restored.in_progress
    backups = list(tmp_path.glob('*.before-reset-*'))
    assert len(backups)==1 and backups[0].read_bytes()==before


def test_external_record_change_is_not_overwritten(tmp_path):
    p,w = tmp_path/'workspace.json',workspace()
    record = BatchProgress(p,w)
    record.reserve()
    record.path.write_text('{}')
    with pytest.raises(ValueError): record.reset_after_home()
    assert record.in_progress and record.path.read_text()=='{}'


@pytest.mark.parametrize('keys,expected', [
    ('hy',['reset_prompt','reset_confirm']),
    ('ㅗㅛ',['reset_prompt','reset_confirm']),
    ('hh',['reset_prompt','reset_prompt']),
    ('y',['stop']), ('hcy',['reset_prompt','stop','stop']),
    ('h y',['reset_prompt','stop','stop']),
    ('h\ny',['reset_prompt','stop','stop']),
    ('hry',['reset_prompt','stop','stop']),
])
def test_only_explicit_h_then_y_confirms_reset(keys,expected):
    state = ConsoleKeyState()
    assert [state.action(k) for k in keys]==expected


class Harness(RestartControlMixin):
    @property
    def aborted(self): return self._abort.is_set()

    def __init__(self,tmp_path):
        self._restart_lock=threading.RLock()
        self._restart_event=threading.Event()
        self._restart_waiting=self._restart_pending=False
        self._restart_supported=True
        self._terminate=threading.Event()
        self._abort=threading.Event()
        self._start_requested=threading.Event()
        self._continue_event=threading.Event()
        self._waiting_confirmation=False
        self._batch_workspace=workspace()
        self._batch_progress=BatchProgress(tmp_path/'workspace.json',self._batch_workspace)
        self._batch_progress.reserve(); self._batch_progress.finish()
        self._batch_progress.reserve()
        self._batch_completed=1
        self.calls=[]
        self.home_ok=self.scene_ok=self.cancel_ok=True
        self.home_hook=None
        self.zero_pose=[0.]*6
        self._command_guard=CommandGuard(['joint1'])
        self._command_guard.record('feedback',['joint1'],[.2],time.monotonic())
        self._control_enable_client=object()
        self.moveit=SimpleNamespace(
            wait_for_servers=lambda **k:True,
            clear_collision_boxes=lambda ids:self.record('clear_goal',ids) and self.scene_ok,
            check_state_validity=lambda s:True)

    def get_logger(self): return Log()
    def _show_restart_state(self,msg): self.calls.append(('status',msg))
    def record(self,name,data=None): self.calls.append((name,data)); return True
    def print_report(self): pass
    def _restart_arm_ready(self): return True
    def _cancel_previous_motion(self): return self.cancel_ok
    def _setup_planning_scene(self): return self.record('table')
    def _reset_cycle_state(self): self.record('discard_targets')
    def _prepare_real_control(self): return self.record('prepare')
    def _set_driver_gate(self,client,value): return self.record('gate',value)
    def _goto_joints(self,label,pose):
        self.record('home')
        if self.home_hook: self.home_hook()
        return self.home_ok
    def emergency_stop(self,reason,terminate=False):
        self._cancel_restart_request()
        self._command_guard.close()
        self._continue_event.set()
        if terminate:self._terminate.set()
    def run(self):
        # After home, emulate waiting for the user's next c, not auto motion.
        self._waiting_confirmation=True
        while not self.aborted and not self._terminate.is_set():
            if self._continue_event.wait(.02):
                self._continue_event.clear()
                if not self.aborted:self.record('cycle')
        return False


def test_reset_service_requires_confirmed_idle_and_never_moves_in_callback(tmp_path):
    c=Harness(tmp_path)
    assert not c._on_reset_request(SetBool.Request(data=True),SetBool.Response()).success
    c._restart_waiting=True
    assert not c._on_reset_request(SetBool.Request(data=False),SetBool.Response()).success
    assert c._on_reset_request(SetBool.Request(data=True),SetBool.Response()).success
    assert not c._on_reset_request(SetBool.Request(data=True),SetBool.Response()).success
    assert not c.calls and c._batch_progress.in_progress
    c.emergency_stop('stop arrived after confirmation')
    assert not c._restart_pending and c.aborted


@pytest.mark.parametrize('failure',['home','stop','scene','cancel'])
def test_home_failure_or_stop_keeps_progress_and_never_opens_gripper(tmp_path,failure):
    c=Harness(tmp_path)
    before=c._batch_progress.path.read_bytes()
    if failure=='home':c.home_ok=False
    if failure=='stop':c.home_hook=lambda:c.emergency_stop('during home')
    if failure=='scene':c.scene_ok=False
    if failure=='cancel':c.cancel_ok=False
    assert not c._home_for_restart()
    assert c._batch_progress.path.read_bytes()==before
    assert c._batch_progress.in_progress and c._batch_completed==1
    if failure in ('scene','cancel'):
        assert not any(name in ('home','prepare') for name,_ in c.calls)


def test_home_success_clears_goal_records_and_stays_closed(tmp_path):
    c=Harness(tmp_path)
    assert c._home_for_restart()
    assert c._batch_completed==0 and not c._batch_progress.in_progress
    names=[name for name,_ in c.calls]
    assert names.index('clear_goal') < names.index('prepare') < names.index('home')
    assert ('gate',False) in c.calls
    assert not c._command_guard.enabled


@pytest.mark.parametrize('detach_ok',[True,False])
@pytest.mark.parametrize('mode',['sweep','feed_auto'])
def test_sweep_reset_clears_held_geometry_before_home(tmp_path,detach_ok,mode):
    from test_sweep import workspace as sweep_workspace
    c=Harness(tmp_path)
    c._batch_workspace=(sweep_workspace() if mode=='sweep' else
                        BatchWorkspace.load(str(Path(__file__).resolve().parents[3] / 'config/feed_auto_workspace.json')))
    c.ee_link='tcp_link'
    c.moveit.clear_attached_box=lambda *a:c.record('detach') and detach_ok
    assert c._home_for_restart() is detach_ok
    names=[name for name,_ in c.calls]
    if detach_ok:assert names.index('detach')<names.index('home')
    else:assert 'home' not in names and c._batch_completed==1


def wait_for(predicate):
    deadline=time.monotonic()+2.
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.005)
    assert predicate()


def test_interrupted_start_waits_for_reset_then_new_c(tmp_path,monkeypatch):
    monkeypatch.setattr(restart_module.rclpy,'ok',lambda:True)
    c=Harness(tmp_path)
    thread=threading.Thread(target=c.run_restartable)
    thread.start()
    try:
        wait_for(lambda:c._restart_waiting)
        assert not any(name in ('home','prepare','cycle') for name,_ in c.calls)
        assert c._on_reset_request(SetBool.Request(data=True),SetBool.Response()).success
        wait_for(lambda:c._waiting_confirmation)
        assert c._batch_completed==0
        assert ('home',None) in c.calls and ('cycle',None) not in c.calls
        c._continue_event.set()
        wait_for(lambda:('cycle',None) in c.calls)
        c.emergency_stop('new experiment error')
        wait_for(lambda:c._restart_waiting)
        assert thread.is_alive()  # services can keep serving another reset
    finally:
        c.emergency_stop('shutdown',terminate=True)
        thread.join(2.)
        assert not thread.is_alive()


def moveit():
    c=MoveItClient.__new__(MoveItClient)
    c._log=Log();c._goal_lock=threading.Lock()
    c._active_goal=None;c._unsettled_goals=set()
    return c


def test_late_goal_is_canceled_and_blocks_reset_until_terminal_result():
    c=moveit()
    sent,done=Future(),Future()
    cancels=[]
    handle=SimpleNamespace(accepted=True,get_result_async=lambda:done,
                           cancel_goal_async=lambda:cancels.append(True))
    client=SimpleNamespace(send_goal_async=lambda goal:sent)
    assert c._send_goal(client,object(),.001,'test') is None
    assert not c.motion_requests_settled()
    sent.set_result(handle)
    assert cancels and not c.motion_requests_settled()
    done.set_result(SimpleNamespace(result='canceled'))
    assert c.motion_requests_settled()


def test_execution_timeout_cancels_without_forgetting_unfinished_goal():
    c=moveit()
    sent,done=Future(),Future();cancels=[]
    sent.set_result(SimpleNamespace(accepted=True,get_result_async=lambda:done,
                    cancel_goal_async=lambda:cancels.append(True)))
    assert c._send_goal(SimpleNamespace(send_goal_async=lambda g:sent),None,.001,'test') is None
    assert cancels and not c.motion_requests_settled()
    done.set_result(SimpleNamespace(result='finished'))
    assert c.motion_requests_settled()


@pytest.mark.parametrize('present',[[],['table'],['table','box']])
def test_clear_goal_models_is_idempotent_and_keeps_table(present):
    c=moveit();objects=set(present);removed=[]
    def get_scene(request):
        future=Future()
        future.set_result(SimpleNamespace(scene=SimpleNamespace(world=SimpleNamespace(
            collision_objects=[SimpleNamespace(id=name) for name in objects]))))
        return future
    def remove(name,timeout):objects.remove(name);removed.append(name);return True
    c._get_scene=SimpleNamespace(wait_for_service=lambda **k:True,call_async=get_scene)
    c.remove_collision_box=remove
    assert c.clear_collision_boxes(['box','absent'])
    assert removed==(['box'] if 'box' in present else [])
    assert ('table' in objects)==('table' in present)
