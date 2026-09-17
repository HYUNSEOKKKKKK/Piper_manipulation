import json
import math
from pathlib import Path
import pytest
from piper_pnp.object_models import ObjectCatalog
from piper_pnp.object_manipulation import box_pose, held_transform, level_place_candidates, place_positions, check_level_release, grip_width
from piper_pnp.geometry import quat_from_rpy, quat_multiply, quat_rotate_vector


def catalog():
    return ObjectCatalog({'version':1,'models':[
        {'id':'small','dims_m':[.077,.035,.030],'place_mode':'legacy'},
        {'id':'square','dims_m':[.08,.08,.0415]}]})


def packet(c):
    return dict(version=1,catalog_digest=c.digest,model_id='square',dims_m=[.08,.08,.0415],track_id=1,
                stamp_ns=10_000_000_000,frame_id='camera',position=[.3,.1,.0465],orientation=[0,0,0,1],
                box_in_grasp_position=[0,0,-.02575],box_in_grasp_orientation=[0,0,0,1])


@pytest.mark.parametrize('key,value',[('catalog_digest','wrong'),('dims_m',[.08,.08,.04]),
                                    ('stamp_ns',0),('track_id',False),('orientation',[0,0,0,0]),
                                    ('box_in_grasp_position',[1.,0.,0.])])
def test_inconsistent_atomic_target_is_rejected(key,value):
    c=catalog();p=packet(c);p[key]=value
    with pytest.raises(ValueError):c.decode(json.dumps(p))


def test_catalog_rejects_duplicate_shape_under_axis_permutation():
    with pytest.raises(ValueError):
        ObjectCatalog({'version':1,'models':[{'id':'a','dims_m':[.08,.08,.0415]},
                                             {'id':'b','dims_m':[.0415,.08,.08]}]})


def pick():
    c=catalog();p=packet(c)
    _,_,pos,q,metadata=c.decode(json.dumps(p))
    return dict(position=pos,orientation=q,**metadata)


@pytest.mark.parametrize('tilt,yaw',[(0,0),(.3,.4),(1.2,.7)])
def test_level_place_bottom_uses_held_transform_and_real_dimensions(tilt,yaw):
    p=pick();p['orientation']=quat_from_rpy(tilt,0.,yaw)
    tool=quat_multiply(p['orientation'],quat_from_rpy(math.pi,0.,0.))
    candidates=level_place_candidates(p,tool,(0.,0.,0.,1.))
    assert len(candidates)==8
    offset,relative=held_transform(p,tool,-.030)
    for q in candidates:
        ready,contact=place_positions(p,tool,q,(.35,-.12),0.,-.030,.018)
        center=tuple(a+b for a,b in zip(contact,quat_rotate_vector(q,offset)))
        assert center[2]-.0415/2==pytest.approx(.003)
        normal=quat_rotate_vector(quat_multiply(q,relative),(0.,0.,1.))
        assert abs(normal[2])==pytest.approx(1.)
        assert math.dist(ready,contact)==pytest.approx(.018)
        assert check_level_release(p,tool,q,contact,(0.,0.,0.,1.),0.,-.030,.1)
        below=(contact[0],contact[1],contact[2]-.015)
        assert not check_level_release(p,tool,q,below,(0.,0.,0.,1.),0.,-.030,.1)


def test_square_box_jaw_width_is_checked_at_actual_grasp_orientation():
    p=pick();down=quat_from_rpy(math.pi,0.,0.)
    assert grip_width(p,down)==pytest.approx(.08)
    diagonal=quat_multiply(quat_from_rpy(0.,0.,math.pi/4),down)
    assert grip_width(p,diagonal)>.11


def test_equivalent_model_axis_sign_cannot_force_upward_tool_at_place():
    for relative in ((0.,0.,0.,1.),quat_from_rpy(math.pi,0.,0.)):
        p=pick();p['box_in_grasp_orientation']=relative
        tool=quat_from_rpy(math.pi,0.,0.)
        candidates=level_place_candidates(p,tool,(0.,0.,0.,1.))
        downward=[q for q in candidates if quat_rotate_vector(q,(0.,0.,1.))[2]<-.1]
        assert len(downward)==4


def test_controller_uses_same_metric_release_pose_in_selection_and_execution():
    from types import MethodType
    pytest.importorskip("geometry_msgs", reason="requires the ROS integration environment")
    from test_place_alignment import harness
    from piper_pnp.piper_pnp_controller import PiperPnpController as Controller
    c=harness();c._targets['pick']=pick()
    c.object_table_z=0.;c.grasp_offset=-.030
    c.preview_only=False;c.guard_real_commands=False
    c._metric_place_poses=MethodType(Controller._metric_place_poses,c)
    c.moveit.compute_ik=lambda *a,**kw:True
    c._align_grasp_to_achieved=lambda ready,contact:(ready,contact)
    q=c._choose_place_orientation((.35,-.12,.02),(0.,0.,0.,1.),.05,.032)
    assert q is not None
    ready,contact=c._approach_ready_impl('Place','place',(.35,-.12,.02),(0.,0.,0.,1.),.032,.05)
    assert contact.position.z==pytest.approx(.0195)
    assert ready.position.z==pytest.approx(.0375)
    assert c._check_place_alignment(contact)
