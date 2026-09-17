import json
import math
from pathlib import Path
from types import SimpleNamespace, MethodType
import pytest

from piper_pnp.batch_workspace import BatchWorkspace, BatchProgress
from piper_pnp.sweep_control import SweepControlMixin, validate_scene, drop_geometry
from piper_pnp.object_manipulation import held_transform, span
from piper_pnp.geometry import quat_from_rpy, quat_multiply, quat_rotate_vector
from test_object_models import pick
from test_batch import batch_targets, observe
from test_batch import CycleHarness, wait_until
import threading


ROOT = Path(__file__).resolve().parents[3]


def workspace():
    return BatchWorkspace.load(ROOT/'config'/'sweep_workspace.json')


def observation(w, height=.08):
    return dict(workspace_digest=w.digest, frame_id='base_link',
                drop=dict(coverage=1.,height_m=height),
                source=dict(coverage=1.,height_m=.09))


def test_sweep_reuses_one_destination_with_bounded_persistent_capacity(tmp_path):
    w=workspace()
    assert len(w.places)==10 and len(set(w.places))==1
    assert not w.operator_feed and w.sweep
    assert not w.allows_pick(w.places[0])
    progress=BatchProgress(tmp_path/'state',w)
    for _ in range(10):
        progress.reserve();progress.finish()
    assert BatchProgress(tmp_path/'state',w).completed==10
    with pytest.raises(ValueError):progress.reserve()


@pytest.mark.parametrize('field,value',[
    ('max_transfers',11),('max_transfers',True),('drop_max_height_m',float('nan')),
    ('drop_clearance_m',0.),('operator_feed',True),('drop_zone_size_xy_m',[.24,.3]),
])
def test_sweep_configuration_rejects_unbounded_or_overlapping_zone(field,value):
    config=json.loads((ROOT/'config'/'sweep_workspace.json').read_text())
    config[field]=value
    with pytest.raises(ValueError):BatchWorkspace(config)


@pytest.mark.parametrize('key,value', [('coverage',.74),('coverage',float('nan')),
                                     ('height_m',None),('height_m',float('nan')),('height_m',.3)])
def test_unknown_goal_depth_is_not_assumed_empty(key,value):
    w=workspace();obs=observation(w);obs['drop'][key]=value
    with pytest.raises(ValueError):validate_scene(obs,w)


@pytest.mark.parametrize('height', [0.,.08,.16])
@pytest.mark.parametrize('tilt', [0.,.3,.7])
def test_drop_preserves_metric_center_and_held_box_clearance(height,tilt):
    w=workspace();p=pick();pq=quat_from_rpy(math.pi-tilt,0.,.4)
    q=quat_from_rpy(math.pi,0.,0.)
    pose,floor=drop_geometry(p,pq,q,w,height,.09,-.03)
    off,rel=held_transform(p,pq,-.03)
    world=quat_rotate_vector(q,off)
    center=[pose.position.x+world[0],pose.position.y+world[1],pose.position.z+world[2]]
    assert center[:2]==pytest.approx(w.places[0][:2])
    bottom=center[2]-span(p['dims'],quat_multiply(q,rel),(0.,0.,1.))/2
    assert bottom>=height+w.drop_clearance-1e-9
    assert pose.position.z>=floor
    assert floor>height+math.sqrt(sum(x*x for x in p['dims']))


def test_scene_requires_eight_valid_observations_and_invalid_frame_resets():
    w=workspace();logs=[]
    c=SimpleNamespace(_batch_workspace=w,marker_stable_samples=8,
                      get_logger=lambda:SimpleNamespace(warn=lambda *a,**kw:logs.append(a)))
    f=MethodType(SweepControlMixin._accept_sweep_scene,c)
    data=dict(workspace_observation=observation(w))
    for _ in range(7):assert f(data) is False
    assert f(data)['drop_top']==pytest.approx(.095)
    assert f({}) is None
    for _ in range(7):assert f(data) is False
    assert f(data)['drop_top']==pytest.approx(.095)


def test_varying_depth_extrema_use_highest_value_without_starving_pose_quorum():
    c=SimpleNamespace(_batch_workspace=workspace(),marker_stable_samples=8)
    f=MethodType(SweepControlMixin._accept_sweep_scene,c)
    for h in (.01,.08,.02,.06,.02,.04,.07):
        assert f(dict(workspace_observation=observation(c._batch_workspace,h))) is False
    result=f(dict(workspace_observation=observation(c._batch_workspace,.01)))
    assert result['drop_top']==pytest.approx(.095)
    assert result['source_top']==pytest.approx(.105)


def test_controller_cannot_freeze_pick_without_same_frame_goal_observation():
    config=json.loads((ROOT/'config'/'sweep_workspace.json').read_text())
    c=batch_targets(config)
    c._accept_sweep_scene=MethodType(SweepControlMixin._accept_sweep_scene,c)
    # Old/legacy target streams cannot trigger a sweep pick.
    for i in range(12):observe(c,'pick',10_000_000_000+i*1_000_000)
    assert not c._targets_ready.is_set()
    assert not c._target_samples['pick']


def test_atomic_pose_and_goal_height_quorum_then_new_track_resets_both():
    from geometry_msgs.msg import Pose
    from std_msgs.msg import Header
    from rclpy.time import Time
    config=json.loads((ROOT/'config'/'sweep_workspace.json').read_text())
    c=batch_targets(config)
    c._accept_sweep_scene=MethodType(SweepControlMixin._accept_sweep_scene,c)
    meta=pick()
    meta['workspace_observation']=observation(c._batch_workspace)
    pose=Pose();pose.orientation.w=1.
    def send(i):
        header=Header(stamp=Time(nanoseconds=10_000_000_000+i*1_000_000).to_msg(),frame_id='camera')
        c._store_target('pick',pose,header,metadata=meta)
    for i in range(7):send(i)
    assert not c._targets_ready.is_set()
    send(7)
    assert c._targets_ready.is_set()
    assert c._targets['pick']['sweep_scene']['drop_top']==pytest.approx(.095)
    meta['track_id']+=1
    send(8)
    assert not c._targets_ready.is_set()
    assert len(c._sweep_scene_samples)==1


def test_contact_speed_restores_even_when_motion_raises():
    c=SimpleNamespace(moveit=SimpleNamespace(velocity_scaling=.25,acceleration_scaling=.2),
                      velocity_scaling=.50,acceleration_scaling=.25)
    with pytest.raises(RuntimeError):
        with SweepControlMixin._sweep_speed(c,contact=True):
            assert c.moveit.velocity_scaling==.50
            assert c.moveit.acceleration_scaling==.25
            raise RuntimeError('motion interrupted')
    assert (c.moveit.velocity_scaling,c.moveit.acceleration_scaling)==(.25,.2)


class SweepHarness(SweepControlMixin, CycleHarness):
    def __init__(self,tmp_path,fail_at=None):
        config=json.loads((ROOT/'config'/'sweep_workspace.json').read_text())
        super().__init__(config,tmp_path,fail_at=fail_at)
        self._targets={'pick':dict(dims=(.08,.08,.0415),
                                  sweep_scene=dict(drop_top=.015,source_top=.08))}
        self._pick_grasp_q=quat_from_rpy(math.pi,0.,0.)
        self._pick_contact_position=(.4,.1,.015)
        self.arm_joints=[f'joint{i}' for i in range(1,7)]
        self.velocity_scaling=.25;self.acceleration_scaling=.2
        self._sweep_times={}
        self._sweep_details={}
        self.moveit.velocity_scaling=.25;self.moveit.acceleration_scaling=.2
        self.moveit.clear_collision_boxes=lambda ids:self.record('clear_source')
        self.moveit.move_to_joints=lambda *a,**kw:(self.record('return'),object())
        self.moveit._execute_trajectory=lambda *a,**kw:self.record('return')
        self._sweep_return_path=object()
        self.moveit.add_collision_box=lambda *a,**kw:self.record('obstacle')
        self.create_publisher=lambda *a,**kw:SimpleNamespace(publish=lambda m:self.record('timing'))
        finish=self._batch_progress.finish
        def counted_finish():
            finish()
            if self._batch_progress.completed==2:self._terminate.set()
        self._batch_progress.finish=counted_finish
    def _choose_sweep_drop(self,*a):
        from piper_pnp.sweep_control import pose_at
        return pose_at((.35,-.12,.25),self._pick_grasp_q),.22
    def _sweep_transfer_release(self,*a):return self.record('release')
    def _sweep_attach_pick(self):return self.record('attach')
    def _after_motion(self,label,ok,path,**kwargs):return ok


def test_one_approval_runs_two_sweeps_without_home_or_reseeding(tmp_path):
    c=SweepHarness(tmp_path)
    thread=threading.Thread(target=c._run_sweep);thread.start()
    try:
        wait_until(lambda:c._waiting_confirmation)
        c._continue_event.set()
        thread.join(3.)
        assert not thread.is_alive()
        labels=[x[0] for x in c.calls]
        assert labels.count('prepare')==1
        assert labels.count('fresh_observation')==2
        assert labels.count('release')==2
        assert labels.count('return')==2
        assert labels.count('sweep 파지 후 촬영 자세')==2
        assert labels.count('attach')==2
        assert 'sweep 수직 인양' not in labels
        assert 'Zero Pose' not in labels
        assert c._batch_progress.completed==2
        assert not c._batch_progress.in_progress
    finally:
        c._terminate.set();c._continue_event.set();thread.join(2.)


def test_failed_transfer_does_not_finish_or_automatically_retry(tmp_path):
    c=SweepHarness(tmp_path,fail_at='release')
    assert not c._run_sweep_cycle()
    assert c._batch_progress.completed==0 and c._batch_progress.in_progress
    assert [x[0] for x in c.calls].count('release')==1
    assert not any(x[0]=='return' for x in c.calls)


def test_attachment_failure_prevents_transit_and_drop(tmp_path):
    c=SweepHarness(tmp_path,fail_at='attach')
    assert not c._run_sweep_cycle()
    labels=[x[0] for x in c.calls]
    assert 'sweep 파지 후 촬영 자세' not in labels and 'release' not in labels
    assert c._batch_progress.in_progress and c._batch_progress.completed==0


@pytest.mark.parametrize('released', [True,False])
def test_transfer_uses_joint_planning_and_detaches_only_after_measured_open(released):
    from piper_pnp.moveit_client import make_start_state
    from piper_pnp.sweep_control import pose_at,HELD_BOX_ID
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    calls=[]
    release=pose_at((.35,-.12,.25),quat_from_rpy(math.pi,0.,0.))
    path=RobotTrajectory()
    path.joint_trajectory.joint_names=['joint1']
    path.joint_trajectory.points=[JointTrajectoryPoint(positions=[.1])]
    state=make_start_state(['joint1','gripper'],[.1,.04])
    def record(name,value=True):calls.append(name);return value
    c=SimpleNamespace(
        _targets={'pick':dict(dims=(.08,.08,.0415),sweep_scene=dict(drop_top=.015))},
        _sweep_details={},aborted=False,arm_joints=['joint1'],camera_ready_pose=[.2],
        gripper_joint='gripper',gripper_open_position=.095,ee_link='tcp_link',
        _sweep_obstacle=lambda *a:True,_release_current_state=lambda:state,
        _gripper_sweep_clear=lambda s:True,_after_motion=lambda *a:True,
        _achieved_ee_pose=lambda:release,open_gripper=lambda:record('open'),
        _sweep_verify_open=lambda:record('measured',released),
        moveit=SimpleNamespace(
            move_to_pose=lambda *a,**kw:(record('joint_plan'),path),
            move_to_joints=lambda *a,**kw:(record('return_plan'),path),
            _execute_trajectory=lambda *a:record('execute'),
            clear_attached_box=lambda *a:record('detach')))
    assert SweepControlMixin._sweep_transfer_release(c,release,.22) is released
    assert calls[:3]==['joint_plan','return_plan','execute']
    assert ('detach' in calls) is released
    if released:assert calls[-3:]==['open','measured','detach']


def test_detach_is_applied_before_removing_the_resulting_world_object():
    from concurrent.futures import Future
    from piper_pnp.moveit_client import MoveItClient
    from moveit_msgs.msg import CollisionObject
    calls=[]
    def apply(request):
        assert request.scene.robot_state.is_diff
        attached=request.scene.robot_state.attached_collision_objects[0]
        assert attached.object.operation==CollisionObject.REMOVE
        calls.append('detach')
        f=Future();f.set_result(SimpleNamespace(success=True));return f
    c=MoveItClient.__new__(MoveItClient)
    c._apply_scene=SimpleNamespace(call_async=apply)
    c.clear_collision_boxes=lambda ids,timeout:calls.append('remove_world') or True
    assert c.clear_attached_box('held','tcp_link')
    assert calls==['detach','remove_world']
