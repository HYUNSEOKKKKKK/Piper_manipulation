"""No hardware: XYZ-only airborne release, continuous feed, and stop boundaries."""
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from piper_pnp.batch_workspace import BatchWorkspace, BatchProgress
from piper_pnp.feed_drop import FeedDropMixin
from piper_pnp.geometry import quat_from_rpy
from piper_pnp.moveit_client import MoveItClient, make_start_state
from piper_pnp.piper_pnp_controller import PiperPnpController as Controller
from piper_pnp.sweep_control import pose_at, validate_scene
from test_batch import CycleHarness, start_batch, wait_until
from test_place_alignment import harness

ROOT = Path(__file__).resolve().parents[3]


def config():
    return json.loads((ROOT/'config'/'feed_auto_workspace.json').read_text())


def test_fixed_point_is_center_of_old_four_places_and_not_sweep():
    w = BatchWorkspace(config())
    old = BatchWorkspace.load(ROOT/'config'/'feed_workspace.json')
    center = tuple(sum(p[i] for p in old.places)/len(old.places) for i in (0,1))
    assert w.place(0)[:2] == pytest.approx(center)
    assert w.place(0)[2] == .15
    assert w.feed_drop and w.drop and not w.sweep
    assert len(w.places) == 10 and len(set(w.places)) == 1
    assert not w.allows_pick(w.place(0))


@pytest.mark.parametrize('value', [.02,float('nan'),.5])
def test_drop_cannot_reuse_near_table_precise_height(value):
    data=config();data['place_positions_m'][0][2]=value
    with pytest.raises(ValueError): BatchWorkspace(data)


@pytest.mark.parametrize('continuous,catalog,valid', [(True,True,True),(False,True,False),(True,False,False)])
def test_deployed_drop_mode_requires_continuous_and_metric_parameters(tmp_path,continuous,catalog,valid):
    path=tmp_path/'workspace.json';path.write_text(json.dumps(config()))
    params={}
    c=SimpleNamespace(declare_parameter=lambda name,default,*a:params.setdefault(name,default),
                      get_parameter=lambda name:SimpleNamespace(value=params[name]),
                      _target_is_sane=lambda *a:True)
    Controller._declare_parameters(c)
    params.update(batch_config_file=str(path),continuous_feed=continuous,
                  object_models_file=str(ROOT/'config'/'object_models.json') if catalog else '',
                  auto_start=False,loop_mode=True,preview_only=False,
                  guard_real_commands=True,use_real_gripper=True)
    if not valid:
        with pytest.raises(ValueError,match='feed_drop'):Controller._read_parameters(c)
        return
    Controller._read_parameters(c)
    assert not params['batch_operator_feed'] and c._batch_workspace.feed_drop
    assert not c._batch_progress.in_progress and c._batch_completed==0


@pytest.mark.parametrize('completed,busy,blocked', [(0,False,False),(1,False,True),(0,True,True)])
def test_new_mode_preserves_previous_record_and_requires_reset_if_occupied(tmp_path,completed,busy,blocked):
    data=config();data['previous_state_file']='old_workspace.json.state.json';w=BatchWorkspace(data)
    old=tmp_path/data['previous_state_file']
    old.write_text(json.dumps(dict(completed=completed,in_progress=busy)))
    before=old.read_bytes()
    p=BatchProgress(tmp_path/'new.json',w,allow_interrupted=True)
    assert p.in_progress is blocked and not p.path.exists()
    assert old.read_bytes()==before
    if blocked:
        with pytest.raises(ValueError):p.reserve()
    p.reset_after_home()
    assert not BatchProgress(tmp_path/'new.json',w).in_progress
    assert old.read_bytes()==before


def test_drop_depth_ignores_source_but_rejects_unknown_goal():
    w=BatchWorkspace(config())
    obs=dict(workspace_digest=w.digest,frame_id='base_link',
             drop=dict(coverage=1.,height_m=.04),source=None)
    assert validate_scene(obs,w)==(.04,w.table_z)
    obs['drop']['coverage']=.5
    with pytest.raises(ValueError):validate_scene(obs,w)


def test_pick_no_longer_rejects_grasp_for_place_alignment():
    c=harness();c._batch_workspace=BatchWorkspace(config())
    c.moveit.compute_ik=lambda *a,**kw:True
    c.grasp_yaw_candidates=[0.,math.pi/2]
    c.grasp_tilt_candidates=[0.]
    c._choose_place_orientation=lambda *a,**kw:pytest.fail('drop has no alignment constraint')
    c._plan_sweep_pick=lambda *a:pytest.fail('must retain legacy feed pick')
    assert Controller._choose_grasp_candidate(c,(.4,.1,.04),(0.,0.,0.,1.),.03,-.03) is not None


def test_moveit_position_only_goal_has_no_orientation_constraints():
    c=MoveItClient.__new__(MoveItClient)
    c.base_frame='base_link';c.ee_link='tcp_link';c.goal_position_tolerance=.005
    c._plan_and_execute=lambda constraints,*a:constraints
    result=c.move_to_pose(pose_at((.41,-.18,.10),(1.,0.,0.,0.)),position_only=True)
    assert len(result.position_constraints)==1 and not result.orientation_constraints
    p=result.position_constraints[0].constraint_region.primitive_poses[0].position
    assert (p.x,p.y,p.z)==(.41,-.18,.10)


def release_harness(fail=None,q=(0.,0.,0.,1.),actual_xyz=None,drop_z=.10):
    # Keep the low-height geometry regression cases explicit, independent of
    # the deployed 150 mm target. The full cycle below uses deployed config.
    data=config();data['place_positions_m'][0][2]=drop_z
    target_xyz=(.41,-.18,drop_z)
    if actual_xyz is None:actual_xyz=target_xyz
    calls=[]
    messages=[]
    def record(name):calls.append(name);return name!=fail
    trajectory=RobotTrajectory()
    trajectory.joint_trajectory.joint_names=['joint1']
    trajectory.joint_trajectory.points=[JointTrajectoryPoint(positions=[.1])]
    state=make_start_state(['joint1','gripper'],[.1,.035])
    c=SimpleNamespace(
        _batch_workspace=BatchWorkspace(data),_batch_completed=0,
        _targets={'pick':dict(dims=(.078,.053,.03),sweep_scene=dict(drop_top=.015),
                             position=(.4,.1,.035),orientation=(0.,0.,0.,1.),
                             box_in_grasp_position=(0.,0.,-.020),
                             box_in_grasp_orientation=(0.,0.,0.,1.))},
        _pick_grasp_q=(1.,0.,0.,0.),_pick_contact_position=None,grasp_offset=-.03,
        base_frame='base_link',_sweep_obstacle_id=lambda *a:'goal',
        aborted=False,arm_joints=['joint1'],camera_ready_pose=[.2],_report=[],_messages=messages,
        gripper_joint='gripper',gripper_open_position=.095,ee_link='tcp_link',
        _begin_step=lambda n,s:record('step'+str(n)),
        _sweep_obstacle=lambda *a:record('pile'),_release_current_state=lambda:state,
        _gripper_sweep_clear=lambda s:record('open_clear'),
        _achieved_ee_pose=lambda:pose_at(actual_xyz,q),
        _after_motion=lambda label,ok,path,**kw:record(label) and ok,
        open_gripper=lambda:record('open'),_sweep_verify_open=lambda:record('measured_open'),
        get_logger=lambda:SimpleNamespace(
            error=lambda msg:messages.append(msg) or record('error'),warn=messages.append))
    def plan(pose,**kwargs):
        assert kwargs['plan_only'] and kwargs['position_only']
        assert (pose.position.x,pose.position.y,pose.position.z)==target_xyz
        return record('plan_xyz'),trajectory
    c.moveit=SimpleNamespace(
        move_to_pose=plan,
        compute_fk=lambda *a:pose_at(target_xyz,q),
        move_to_joints=lambda *a,**kw:(record('plan_camera'),trajectory),
        _execute_trajectory=lambda *a:record('execute'),
        clear_attached_box=lambda *a:record('detach'),
        clear_collision_boxes=lambda *a:record('clear_landing'),
        add_collision_box=lambda *a:record('pile'),
        move_cartesian=lambda *a,**kw:pytest.fail('place must not use Cartesian moves'))
    for name in ('_drop_box_extent','_drop_landing_geometry','_drop_landing_fits','_drop_return_scene','_plan_feed_drop'):
        setattr(c,name,MethodType(getattr(FeedDropMixin,name),c))
    return c,calls


@pytest.mark.parametrize('q', [(0.,0.,0.,1.),quat_from_rpy(.2,.1,-1.1),quat_from_rpy(math.pi,0.,.4)])
def test_arrived_orientation_is_accepted_and_no_place_cartesian_is_called(q):
    c,calls=release_harness(q=q)
    assert FeedDropMixin._run_feed_drop(c)
    assert calls==['step6','pile','plan_xyz','open_clear','plan_camera','execute',
                   '중앙 낙하 위치 이동','open_clear','step7','open','measured_open','detach',
                   'pile','step8','execute','낙하 후 Camera-Ready Pose']


@pytest.mark.parametrize('xyz', [(.44,-.18,.10),(.41,-.18,float('nan'))])
def test_actual_xyz_error_prevents_open(xyz):
    c,calls=release_harness(actual_xyz=xyz)
    assert not FeedDropMixin._run_feed_drop(c)
    assert 'open' not in calls and 'detach' not in calls


@pytest.mark.parametrize('failure',['pile','plan_xyz','open_clear','plan_camera','execute'])
def test_failed_motion_or_clearance_does_not_open(failure):
    c,calls=release_harness(fail=failure)
    assert not FeedDropMixin._run_feed_drop(c)
    assert 'open' not in calls and 'detach' not in calls


@pytest.mark.parametrize('reason,failure,summary', [
    ('height',None,'높이 여유=3, 개방 검사=0, 복귀 계획=0'),
    ('opening','open_clear','높이 여유=0, 개방 검사=3, 복귀 계획=0'),
    ('return','plan_camera','높이 여유=0, 개방 검사=0, 복귀 계획=3')])
def test_successful_outbound_plans_report_the_actual_rejection(reason,failure,summary):
    c,calls=release_harness(fail=failure)
    if reason=='height':c._targets['pick']['sweep_scene']['drop_top']=.050
    assert not FeedDropMixin._run_feed_drop(c)
    assert calls.count('plan_xyz')==3
    assert 'execute' not in calls and 'open' not in calls
    assert c._report==[('중앙 낙하 사전 검사',False)]
    assert summary in c._messages[-1]
    if reason=='height':
        assert '간격=25.0mm < 필요 40.0mm' in c._messages[0]
        assert 'plan_camera' not in calls and 'open_clear' not in calls


@pytest.mark.parametrize('failure',['open','measured_open','detach'])
def test_return_requires_successful_measured_release(failure):
    c,calls=release_harness(fail=failure)
    assert not FeedDropMixin._run_feed_drop(c)
    assert 'step8' not in calls and calls.count('execute')==1
    if failure!='detach':assert 'detach' not in calls


def test_high_pile_is_rejected_before_pick_without_raising_fixed_z():
    c,calls=release_harness()
    c._targets['pick']['sweep_scene']['drop_top']=.14
    assert not FeedDropMixin._prepare_feed_drop(c)
    assert calls==['error']
    assert c._batch_workspace.place(0)[2]==.10


def test_low_goal_uses_box_bottom_clearance_instead_of_diagonal_bound():
    c,calls=release_harness()
    assert FeedDropMixin._prepare_feed_drop(c)  # empty goal is valid at 100 mm
    bottom,height=c._drop_box_extent(pose_at((.41,-.18,.10),(0.,0.,0.,1.)))
    assert bottom==pytest.approx(.075) and height==pytest.approx(.03)
    c,calls=release_harness(q=quat_from_rpy(.6,.9,-1.1))
    assert not FeedDropMixin._run_feed_drop(c)  # orientation is free, clearance is not
    assert 'open' not in calls


def test_deployed_height_passes_reported_62mm_goal_without_lowering_clearance():
    c,calls=release_harness(drop_z=config()['place_positions_m'][0][2])
    c._targets['pick']['sweep_scene']['drop_top']=.062
    assert c._batch_workspace.drop_clearance==.04
    assert FeedDropMixin._prepare_feed_drop(c)
    assert FeedDropMixin._run_feed_drop(c)
    assert 'open' in calls and c._batch_workspace.place(0)[2]==.15


def test_return_scene_removes_only_hypothetical_held_box_and_bounds_new_pile():
    from moveit_msgs.msg import CollisionObject
    from piper_pnp.sweep_control import HELD_BOX_ID
    c,_=release_harness()
    scene=c._drop_return_scene(pose_at((.41,-.18,.10),(0.,0.,0.,1.)))
    assert scene.is_diff and scene.robot_state.is_diff
    removal=scene.robot_state.attached_collision_objects[0]
    assert removal.object.id==HELD_BOX_ID and removal.object.operation==CollisionObject.REMOVE
    assert scene.world.collision_objects[0].id==HELD_BOX_ID
    assert scene.world.collision_objects[0].operation==CollisionObject.REMOVE
    box=scene.world.collision_objects[-1]
    assert box.id=='goal_landing' and box.primitives[0].dimensions[2]==pytest.approx(.040)


def test_tilted_landing_uses_oriented_cuboid_and_measured_corners():
    q=quat_from_rpy(.2,.1,.8)
    c,_=release_harness(q=q)
    pose=pose_at((.41,-.18,.10),q)
    box=c._drop_return_scene(pose).world.collision_objects[-1]
    assert list(box.primitives[0].dimensions)==pytest.approx([.088,.063,.040])
    assert abs(box.primitive_poses[0].orientation.y)>.1
    assert c._drop_landing_fits(pose,box)
    assert not c._drop_landing_fits(pose_at((.43,-.18,.10),q),box)
    assert not c._drop_landing_fits(pose_at((.41,-.18,.10),quat_from_rpy(.2,.1,1.8)),box)


def test_scene_diff_is_forwarded_only_for_planning():
    from moveit_msgs.msg import Constraints,PlanningScene,MoveItErrorCodes
    c=MoveItClient.__new__(MoveItClient)
    c.group_name='arm';c.planning_attempts=1;c.planning_time=1.
    c.velocity_scaling=.5;c.acceleration_scaling=.25;c.base_frame='base_link'
    c._move_group=object();goals=[]
    c._send_goal=lambda client,goal,*a:goals.append(goal) or SimpleNamespace(
        error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS),planned_trajectory=RobotTrajectory())
    scene=PlanningScene(name='return_after_release',is_diff=True)
    assert c._plan_and_execute(Constraints(),1.,plan_only=True,scene_diff=scene)[0]
    assert goals[0].planning_options.planning_scene_diff.name==scene.name
    with pytest.raises(ValueError,match='planning only'):
        c._plan_and_execute(Constraints(),1.,plan_only=False,scene_diff=scene)
    assert len(goals)==1


@pytest.mark.parametrize('busy',[True,False])
@pytest.mark.parametrize('previous_digest',['1'*64, '2'*64])
def test_height_change_keeps_old_progress_until_confirmed_reset(tmp_path,busy,previous_digest):
    data=config();data['previous_workspace_digests']=['1'*64, '2'*64]
    w=BatchWorkspace(data);path=tmp_path/'workspace.json'
    state_path=Path(str(path)+'.state.json')
    state_path.write_text(json.dumps(dict(workspace_digest=previous_digest,completed=2,in_progress=busy)))
    before=state_path.read_bytes()
    with pytest.raises(ValueError):BatchProgress(path,w)
    p=BatchProgress(path,w,allow_interrupted=True)
    assert p.completed==2 and p.in_progress
    assert state_path.read_bytes()==before
    with pytest.raises(ValueError):p.reserve()
    p.reset_after_home()
    current=BatchProgress(path,w)
    assert current.completed==0 and not current.in_progress
    assert json.loads(state_path.read_text())['workspace_digest']==w.digest
    assert list(tmp_path.glob('*.before-reset-*'))[0].read_bytes()==before


class DropCycleHarness(CycleHarness):
    def __init__(self,tmp_path,fail_at=None):
        super().__init__(config(),tmp_path,fail_at)
        self.continuous_feed=True
    def _sweep_obstacle(self,*a):return self.record('pile')
    def _prepare_feed_drop(self):return self.record('observe_goal')
    def _sweep_attach_pick(self):return self.record('attach')
    def _run_feed_drop(self):
        return (self.record('drop_xyz',self._batch_workspace.place(self._batch_completed))
                and self.open_gripper() and self._goto_joints('Camera-Ready Pose',self.camera_ready_pose))


def test_one_c_runs_ten_same_point_drops_with_only_pick_descent(tmp_path):
    c=DropCycleHarness(tmp_path)
    thread,result=start_batch(c)
    try:
        assert [s for s,_ in c.calls]==['bridge','pile']
        assert c._on_continue_request(None,Trigger.Response()).success
        wait_until(lambda:any('최대 이송 횟수' in s for s in c.logs))
        labels=[s for s,_ in c.calls]
        assert c._batch_completed==10 and labels.count('Zero Pose')==1
        assert labels.count('Pick 타겟까지 직선 하강')==10
        assert labels.count('직선 상승')==10  # pick retreat only
        assert 'Place-Ready Pose' not in labels and 'Place 타겟까지 직선 하강' not in labels
        assert 'alignment_check' not in labels and 'release_check' not in labels
        assert [v for s,v in c.calls if s=='drop_xyz']==[(.41,-.18,.15)]*10
        assert labels.count('fresh_observation')==10 and labels.count('attach')==10
        assert labels.count('Camera-Ready Pose')==11  # initial + after each release only
        for i,s in enumerate(labels):
            if s=='attach': assert labels[i+1]=='drop_xyz'
        assert labels.count('open')==11 and labels.count('close')==10
        assert not c._on_continue_request(None,Trigger.Response()).success
    finally:
        c._terminate.set();thread.join(2.)
    assert result==[True]


def test_goal_check_failure_does_not_reserve_or_pick(tmp_path):
    c=DropCycleHarness(tmp_path,fail_at='observe_goal')
    thread,result=start_batch(c)
    c._on_continue_request(None,Trigger.Response())
    thread.join(2.)
    assert result==[False] and c._batch_completed==0
    assert not c._batch_progress.in_progress
    assert 'Pick-Ready Pose' not in [s for s,_ in c.calls]
