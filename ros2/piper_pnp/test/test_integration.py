"""실제 서비스 호출 없이 제어기 메서드와 명령 중계의 경계 조건 검증."""
from collections import deque
import math
import threading
import time
from types import SimpleNamespace, MethodType

import pytest
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_pnp.piper_pnp_controller import PiperPnpController as Controller
from piper_pnp.control_guard import CommandGuard
from piper_pnp.real_control import RealControlMixin


class Log:
    def info(self, *a, **kw): pass
    warn = error = info


def target_controller():
    c = SimpleNamespace(
        now_ns=10_100_000_000, target_max_age=.5, _target_not_before_ns=9_000_000_000,
        _last_target_stamp={}, _target_stamps={}, _targets={},
        _target_samples={'pick':deque(),'place':deque()}, _targets_lock=threading.Lock(),
        _targets_ready=threading.Event(), _frozen=False, marker_stable_window=3.,
        marker_stable_samples=8, marker_stable_spread=.02, marker_stable_yaw=math.radians(15),
        base_frame='base_link', aborted=False, marker_wait_timeout=.02,
    )
    c.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=c.now_ns))
    c.get_logger=lambda:Log()
    c._to_base_frame=lambda p,h:((.35,0.,.03),(0.,0.,0.,1.))
    c._target_is_sane=lambda key,pos:True
    c._stable_estimate=Controller._stable_estimate
    for name in ('_store_target','_expire_targets_locked','_expire_targets','_wait_for_targets'):
        setattr(c,name,MethodType(getattr(Controller,name),c))
    return c


def observe(c, key, stamp_ns):
    p=Pose()
    p.orientation.w=1.
    h=Header(stamp=Time(nanoseconds=stamp_ns).to_msg(),frame_id='camera')
    c._store_target(key,p,h)


def test_repeated_image_cannot_fill_quorum():
    c=target_controller()
    for _ in range(10): observe(c,'pick',10_000_000_000)
    assert len(c._target_samples['pick'])==1
    assert not c._targets


def test_fresh_quorum_confirms_and_expiry_removes_targets():
    c=target_controller()
    for key in ('pick','place'):
        for i in range(8): observe(c,key,10_000_000_000+i*1_000_000)
    assert c._targets_ready.is_set()
    c.now_ns+=1_000_000_000
    c._expire_targets()
    assert not c._targets_ready.is_set()
    assert not c._targets


def test_old_and_pre_camera_ready_frames_are_ignored():
    c=target_controller()
    observe(c,'pick',9_000_000_000)
    c._target_not_before_ns=10_050_000_000
    observe(c,'pick',10_000_000_000)
    assert not c._target_samples['pick']


def test_reset_during_tf_conversion_cannot_reintroduce_old_sample():
    c=target_controller()
    def convert(p,h):
        c._target_not_before_ns=c.now_ns
        return (.35,0.,.03),(0.,0.,0.,1.)
    c._to_base_frame=convert
    observe(c,'pick',10_000_000_000)
    assert not c._target_samples['pick']


def test_camera_ready_wait_discards_preexisting_ready_event():
    c=target_controller()
    c._targets_ready.set()
    c._targets={'pick':{},'place':{}}
    assert not c._wait_for_targets()
    assert not c._frozen
    assert not c._targets_ready.is_set()


@pytest.mark.parametrize('tilt',[0.,.3,.7])
@pytest.mark.parametrize('axial_error',[-.005,.004])
def test_pick_approach_error_does_not_change_insertion_depth(tilt,axial_error):
    from piper_pnp.geometry import quat_from_rpy,quat_rotate_vector
    from piper_pnp.sweep_control import pose_at
    q=quat_from_rpy(math.pi-tilt,0.,.4)
    axis=quat_rotate_vector(q,(0.,0.,1.))
    contact=(.4,.1,.006)
    pre=tuple(v-.06*a for v,a in zip(contact,axis))
    actual=tuple(v+axial_error*a for v,a in zip(pre,axis))
    c=SimpleNamespace(_achieved_ee_pose=lambda:pose_at(actual,q),
                      grasp_ik_mode='candidates',get_logger=lambda:Log())
    _,result=Controller._align_grasp_to_achieved(c,pose_at(pre,q),pose_at(contact,q),anchor_contact=True)
    assert (result.position.x,result.position.y,result.position.z)==pytest.approx(contact)


def test_pick_actual_orientation_keeps_contact_plane_without_flat_table_assumption():
    from piper_pnp.geometry import quat_from_rpy,quat_rotate_vector
    from piper_pnp.sweep_control import pose_at
    q=quat_from_rpy(math.pi-.4,0.,.4)
    actual_q=quat_from_rpy(math.pi-.41,.006,.403)
    normal=quat_rotate_vector(q,(0.,0.,1.))
    axis=quat_rotate_vector(actual_q,(0.,0.,1.))
    contact=(.4,.1,.07)
    pre=tuple(v-.06*n for v,n in zip(contact,normal))
    actual=(pre[0]+.002,pre[1]-.001,pre[2]-.004)
    c=SimpleNamespace(_achieved_ee_pose=lambda:pose_at(actual,actual_q),
                      grasp_ik_mode='candidates',get_logger=lambda:Log())
    _,result=Controller._align_grasp_to_achieved(c,pose_at(pre,q),pose_at(contact,q),anchor_contact=True)
    end=(result.position.x,result.position.y,result.position.z)
    assert sum((e-t)*n for e,t,n in zip(end,contact,normal))==pytest.approx(0.,abs=1e-12)
    delta=tuple(e-s for e,s in zip(end,actual));distance=math.dist(end,actual)
    assert tuple(d/distance for d in delta)==pytest.approx(axis)


@pytest.mark.parametrize('velocity',[.1,.5])
@pytest.mark.parametrize('continuous',[False,True])
@pytest.mark.parametrize('feed_drop',[False,True])
def test_feed_candidate_approach_preserves_legacy_sixty_mm_descent(velocity,continuous,feed_drop):
    from piper_pnp.geometry import quat_from_rpy,quat_rotate_vector
    from piper_pnp.sweep_control import pose_at
    q=quat_from_rpy(math.pi-.25,0.,.4)
    axis=quat_rotate_vector(q,(0.,0.,1.))
    marker=(.426,.142,.034)
    pre,contact=Controller._poses_along_axis(marker,q,.03,-.03)
    p=(pre.position.x,pre.position.y,pre.position.z)
    # This approach landed high. The experimental plane correction would
    # lengthen the descent to 63.4 mm; legacy feed must remain exactly 60 mm.
    actual=tuple(v-.0034*a for v,a in zip(p,axis))
    calls=[]
    c=SimpleNamespace(grasp_ik_mode='candidates',velocity_scaling=velocity,continuous_feed=continuous,
                      _batch_workspace=SimpleNamespace(sweep=False,feed_drop=feed_drop),place_match_pick_tilt=True,
                      _choose_grasp_candidate=lambda *a:q,
                      _poses_along_axis=Controller._poses_along_axis,
                      _goto_pose=lambda *a:calls.append('legacy_approach') or True,
                      _achieved_ee_pose=lambda:pose_at(actual,q),get_logger=lambda:Log())
    c._align_grasp_to_achieved=MethodType(Controller._align_grasp_to_achieved,c)
    ready,end=Controller._approach_ready_impl(c,'pick','pick',marker,q,-.03,.03)
    end_xyz=(end.position.x,end.position.y,end.position.z)
    assert calls==['legacy_approach']
    assert math.dist(actual,end_xyz)==pytest.approx(.060)
    assert end_xyz==pytest.approx(tuple(v+.060*a for v,a in zip(actual,axis)))


def test_same_yaw_with_flipped_normal_is_unstable():
    samples=[(0.,(.35,0.,.03),(0.,0.,0.,1.)),
             (1.,(.35,0.,.03),(1.,0.,0.,0.))]
    assert Controller._stable_estimate(samples,.02,math.radians(15))[0] is None


def test_quaternion_sign_does_not_change_orientation_stability():
    samples=[(0.,(.35,0.,.03),(0.,0.,0.,1.)),
             (1.,(.35,0.,.03),(0.,0.,0.,-1.))]
    assert Controller._stable_estimate(samples,.02,math.radians(15))[0] is not None


def relay():
    r=SimpleNamespace(_command_guard=CommandGuard(['joint1','joint2']),
                      _abort=threading.Event(), messages=[], stops=[])
    r._hardware_pub=SimpleNamespace(publish=r.messages.append)
    r.emergency_stop=r.stops.append
    r._command_guard.record('feedback',['joint1','joint2'],[.4,-.2],time.monotonic())
    return r


def command():
    m=JointState()
    m.name=['joint1','joint2','gripper']
    m.position=[.4,-.2,0.]
    return m


def test_closed_relay_drops_commands_and_dummy_gripper_is_filtered():
    r=relay()
    RealControlMixin._on_mock_command(r,command())
    assert not r.messages
    r._command_guard.open(time.monotonic())
    RealControlMixin._on_mock_command(r,command())
    assert r.messages[-1].name==['joint1','joint2']
    assert list(r.messages[-1].position)==[.4,-.2]


def test_feedback_loss_closes_gate_before_next_publish():
    r=relay()
    RealControlMixin._on_mock_command(r,command())
    r._command_guard.open(time.monotonic())
    r._command_guard.feedback_at-=1.
    RealControlMixin._on_mock_command(r,command())
    assert not r._command_guard.enabled
    assert not r.messages
    assert r.stops


def test_watchdog_stops_even_without_new_command_callbacks():
    r=relay()
    RealControlMixin._on_mock_command(r,command())
    r._command_guard.open(time.monotonic())
    r._command_guard.command_at-=1.
    RealControlMixin._command_watchdog(r)
    assert not r._command_guard.enabled
    assert r.stops


def test_prepare_seeds_while_gate_closed_then_enables():
    r=relay()
    r.guard_real_commands=True
    r.preview_only=False
    r.use_real_gripper=False
    r.aborted=False
    r.arm_joints=['joint1','joint2']
    r._control_enable_client='control'
    r._arm_enable_client='arm'
    r._arm_seed_client='seed'
    r.count_publishers=lambda topic:1
    r.get_logger=lambda:Log()
    calls=[]
    def driver(client,value):
        assert not r._command_guard.enabled
        calls.append((client,value))
        return True
    def seed(client,names,positions):
        assert not r._command_guard.enabled
        r._command_guard.record('command',names,positions,time.monotonic())
        calls.append(('seed',positions))
        return True
    r._set_driver_gate=driver
    r._seed_mock_controller=seed
    assert RealControlMixin._prepare_real_control(r)
    assert calls==[('control',False),('seed',[.4,-.2]),('arm',True),('control',True)]
    assert r._command_guard.enabled


def arrival_controller(monkeypatch, actual, *, ok=True, trajectory=None, stale=False):
    """MoveIt応答のみを代替し、実際の _goto_joints → 到達検査を通す。"""
    clock = SimpleNamespace(now=100.)
    def advance(dt):
        clock.now += dt
    monkeypatch.setattr('piper_pnp.real_control.time', SimpleNamespace(
        monotonic=lambda: clock.now, sleep=advance))
    names = [f'joint{i}' for i in range(1, 7)]
    guard = CommandGuard(names)
    guard.record('feedback', names, actual, clock.now)
    guard.record('command', names, actual, clock.now)
    guard.open(clock.now)
    if stale:
        guard.feedback_at -= 1.
    c = SimpleNamespace(
        arm_joints=names, guard_real_commands=True, preview_only=False,
        _preview_state=None, _command_guard=guard, _report=[], stops=[],
        feedback_arrival_timeout=.2, feedback_joint_tolerance=.03, aborted=False,
        moveit=SimpleNamespace(move_to_joints=lambda *a, **kw: (ok, trajectory)),
        get_logger=lambda: Log(), _prefix=lambda: '')
    c.emergency_stop=c.stops.append
    c._after_motion=MethodType(Controller._after_motion,c)
    c._verify_real_arrival=MethodType(RealControlMixin._verify_real_arrival,c)
    return c, clock


def test_already_at_joint_goal_with_empty_success_checks_actual_feedback(monkeypatch):
    c,clock=arrival_controller(monkeypatch,[.0008]*6,trajectory=RobotTrajectory())
    assert Controller._goto_joints(c,'Zero Pose',[0.]*6)
    assert clock.now >= 100.15  # 기존 0.15초 실제 도달 안정 조건 유지
    assert c._report==[('Zero Pose',True)]
    assert not c.stops


@pytest.mark.parametrize('actual,stale,ok',[
    ([.2]*6,False,True),  # MoveIt 성공이어도 실제 팔이 목표에서 멀면 차단
    ([0.]*6,True,True),  # 영점 값이어도 오래된 피드백이면 차단
    ([0.]*6,False,False),  # MoveIt 실패를 현재 관절값으로 덮어쓰지 않음
])
def test_empty_joint_trajectory_cannot_hide_failure(monkeypatch,actual,stale,ok):
    c,_=arrival_controller(monkeypatch,actual,trajectory=RobotTrajectory(),stale=stale,ok=ok)
    assert not Controller._goto_joints(c,'Zero Pose',[0.]*6)
    assert c._report==[('Zero Pose',False)]
    assert c.stops


def test_missing_trajectory_is_not_already_at_goal(monkeypatch):
    c,_=arrival_controller(monkeypatch,[0.]*6)
    assert not Controller._goto_joints(c,'Zero Pose',[0.]*6)


def test_empty_pose_trajectory_without_joint_target_still_fails(monkeypatch):
    c,_=arrival_controller(monkeypatch,[0.]*6,trajectory=RobotTrajectory())
    assert not c._after_motion('Pose',True,RobotTrajectory())
    assert c.stops


@pytest.mark.parametrize('target',[[0.]*5,[float('nan')]*6])
def test_empty_joint_trajectory_requires_complete_finite_goal(monkeypatch,target):
    c,_=arrival_controller(monkeypatch,[0.]*6,trajectory=RobotTrajectory())
    assert not Controller._goto_joints(c,'Zero Pose',target)


def test_nonempty_trajectory_checks_planned_endpoint(monkeypatch):
    trajectory=RobotTrajectory()
    trajectory.joint_trajectory.joint_names=[f'joint{i}' for i in range(1,7)]
    trajectory.joint_trajectory.points=[JointTrajectoryPoint(positions=[.1]*6)]
    c,_=arrival_controller(monkeypatch,[0.]*6,trajectory=trajectory)
    assert not Controller._goto_joints(c,'Zero Pose',[0.]*6)


def test_model_and_track_changes_restart_eight_frame_quorum():
    c=target_controller();c._object_catalog=object()
    p=Pose();p.orientation.w=1.
    def send(i,model,track):
        h=Header(stamp=Time(nanoseconds=10_000_000_000+i*1_000_000).to_msg(),frame_id='camera')
        c._store_target('pick',p,h,metadata=dict(model_id=model,track_id=track,dims=(.08,.08,.0415)))
    for i in range(7):send(i,'small',1)
    send(7,'square',1)
    assert len(c._target_samples['pick'])==1 and 'pick' not in c._targets
    for i in range(8,15):send(i,'square',1)
    assert c._targets['pick']['model_id']=='square'
    send(15,'square',2)
    assert len(c._target_samples['pick'])==1 and 'pick' not in c._targets


def test_multimodel_controller_rejects_legacy_marker_bypass():
    c=target_controller();c._object_catalog=object()
    for i in range(8):observe(c,'pick',10_000_000_000+i*1_000_000)
    assert not c._target_samples['pick'] and not c._targets


def test_frozen_cycle_keeps_model_and_geometry():
    c=target_controller();c._object_catalog=object()
    p=Pose();p.orientation.w=1.
    for i in range(8):
        h=Header(stamp=Time(nanoseconds=10_000_000_000+i*1_000_000).to_msg(),frame_id='camera')
        c._store_target('pick',p,h,metadata=dict(model_id='square',track_id=1,dims=(.08,.08,.0415)))
    c._frozen=True
    h.stamp=Time(nanoseconds=10_020_000_000).to_msg()
    c._store_target('pick',p,h,metadata=dict(model_id='small',track_id=2,dims=(.077,.035,.03)))
    assert c._targets['pick']['model_id']=='square' and c._targets['pick']['dims']==(.08,.08,.0415)
