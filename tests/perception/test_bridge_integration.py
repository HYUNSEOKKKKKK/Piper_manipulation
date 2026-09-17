"""Regression checks for target loss, camera motion and sensor encoding."""
import time
import cv2
import numpy as np
import pytest
import ros_pose_bridge as module
from bridge_probe import Probe


class Segmenter:
    def __init__(self, masks):
        self.masks = masks

    def detect_and_segment(self, image):
        return self.masks, []


@pytest.fixture
def scene(monkeypatch):
    mask = np.zeros((100,100), np.uint8)
    cv2.fillPoly(mask, [np.array([[30,40],[50,25],[70,40],[70,60],[50,75],[30,60]])], 1)
    probe = Probe(Segmenter([mask.astype(bool)]), [.077,.035,.030], 0.)
    probe.K = np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
    probe.dbg_pub = None
    c = probe.bridge.cv2_to_imgmsg(np.zeros((100,100,3),np.uint8),'bgr8')
    c.header.frame_id = 'camera'
    d = probe.bridge.cv2_to_imgmsg(np.full((100,100),400,np.uint16),'16UC1')
    fit = dict(R=np.eye(3), t=np.array([0.,0.,.4]), combined=1., sdf_err_mm=1., contour_err_px=1.)
    monkeypatch.setattr(module, 'fit_pose_rgb_then_depth', lambda *a,**kw: fit)
    return probe, c, d, fit


def test_no_detections_eventually_release_lock(scene):
    p,c,d,fit = scene
    p._process(c,d)
    p.seg.masks=[]
    for _ in range(16):
        p._process(c,d)
    assert p._lock_pos is None
    assert p._prev_x is None


def test_unrelated_object_cannot_steal_lock(scene):
    p,c,d,fit = scene
    p._process(c,d)
    fit['t'] = np.array([.4,0.,.4])
    p.pub.messages.clear()
    for _ in range(15):
        p._process(c,d)
    assert not p.pub.messages
    p._process(c,d)  # expires old lock
    assert not p.pub.messages
    p._process(c,d)  # next observation can establish a new target
    assert len(p.pub.messages)==1


def test_no_frames_for_timeout_allows_new_lock(scene):
    p,c,d,fit = scene
    p._process(c,d)
    p._lock_seen = time.monotonic()-3.
    fit['t'] = np.array([.4,0.,.4])
    p.pub.messages.clear()
    p._process(c,d)
    assert len(p.pub.messages)==1


def test_camera_motion_compensated_in_base_frame(scene):
    p,c,d,fit = scene
    p.use_tf_up = True
    p._base_from_camera = lambda *a: (np.eye(3),np.zeros(3))
    p._process(c,d)
    p.pub.messages.clear()
    # Camera moves 20cm in base X; stationary object's camera X changes by -20cm.
    p._base_from_camera = lambda *a: (np.eye(3),np.array([.2,0.,0.]))
    fit['t'] = np.array([-.2,0.,.4])
    p._process(c,d)
    assert len(p.pub.messages)==1
    np.testing.assert_allclose(p._lock_pos,[0.,0.,.4])
    assert '[locked]' in p.logs[-1]


def test_missing_tf_suppresses_pose(scene):
    p,c,d,fit = scene
    p.use_tf_up=True
    p._process(c,d)
    assert not p.pub.messages


def test_stale_after_inference_suppresses_pose(scene):
    p,c,d,fit = scene
    checks=iter([False,True])
    p._pose_is_stale=lambda stamp: next(checks)
    p._process(c,d)
    assert not p.pub.messages


def test_float_depth_uses_metres(scene):
    p,c,d,fit = scene
    d = p.bridge.cv2_to_imgmsg(np.full((100,100),.4,np.float32),'32FC1')
    p._process(c,d)
    assert len(p.pub.messages)==1


def test_nonfinite_fit_is_rejected(scene):
    p,c,d,fit=scene
    fit['combined']=float('nan')
    p._process(c,d)
    assert not p.pub.messages


def test_yellow_target_rejects_better_fitting_orange_distractor(scene, monkeypatch):
    p,c,d,fit = scene
    yellow = np.zeros((100,100), np.uint8)
    cv2.fillPoly(yellow, [np.array([[10,40],[25,25],[40,40],[40,60],[25,75],[10,60]])], 1)
    yellow = yellow.astype(bool)
    orange = np.roll(yellow, 50, axis=1)
    bgr = np.zeros((100,100,3), np.uint8)
    bgr[yellow] = (0,220,220)
    bgr[orange] = (0,90,220)
    c = p.bridge.cv2_to_imgmsg(bgr, 'bgr8')
    c.header.frame_id = 'camera'
    p.seg.masks = [orange, yellow]
    orange_fit = dict(fit, t=np.array([.2,0.,.4]), combined=0.)
    monkeypatch.setattr(module, 'fit_pose_rgb_then_depth',
                        lambda m,*a,**kw: orange_fit if np.array_equal(m,orange) else fit)
    # Without the color constraint, both silhouettes pass and best_fit picks
    # the distractor. This verifies that the new constraint makes the difference.
    p._process(c,d)
    np.testing.assert_allclose(p._lock_pos, orange_fit['t'])
    p._reset_lock()
    p.pub.messages.clear()
    p.target_color = 'yellow'
    p._process(c,d)
    assert len(p.pub.messages) == 1
    np.testing.assert_allclose(p._lock_pos, fit['t'])


def test_yellow_target_loss_never_falls_back_to_other_color(scene):
    p,c,d,fit = scene
    p.target_color = 'yellow'
    bgr = np.full((100,100,3), (0,220,220), np.uint8)
    c = p.bridge.cv2_to_imgmsg(bgr, 'bgr8')
    p._process(c,d)
    assert p.pub.messages
    p.pub.messages.clear()
    c = p.bridge.cv2_to_imgmsg(np.full_like(bgr,(0,90,220)), 'bgr8')
    for _ in range(20):
        p._process(c,d)
    assert not p.pub.messages
    assert p._lock_pos is None


def test_any_color_keeps_original_detection_behavior(scene):
    p,c,d,fit = scene
    c = p.bridge.cv2_to_imgmsg(np.full((100,100,3),(0,90,220),np.uint8), 'bgr8')
    p._process(c,d)
    assert len(p.pub.messages) == 1


def batch_workspace():
    from piper_pnp.batch_workspace import BatchWorkspace
    return BatchWorkspace(dict(object_dims_m=[.077,.035,.030],
        pick_bounds_xy=[.2,.6,0.,.3], place_positions_m=[[.35,-.12,.02],[.47,-.12,.02]],
        place_exclusion_radius_m=.08, placed_box_envelope_m=[.10,.10,.10], table_z_m=0.))


def test_batch_selects_second_color_after_first_is_placed(scene,monkeypatch):
    p,c,d,fit = scene
    left = np.zeros((100,100), np.uint8)
    cv2.fillPoly(left,[np.array([[10,40],[25,25],[40,40],[40,60],[25,75],[10,60]])],1)
    left = left.astype(bool)
    right = np.roll(left,50,axis=1)
    bgr = np.zeros((100,100,3),np.uint8)
    bgr[left], bgr[right] = (0,220,220), (220,80,0)  # yellow and blue
    c = p.bridge.cv2_to_imgmsg(bgr,'bgr8')
    c.header.frame_id = 'camera'
    p.seg.masks = [left,right]
    first = dict(fit,t=np.array([.35,.10,.04]),combined=0.)
    second = dict(fit,t=np.array([.50,.15,.04]),combined=1.)
    monkeypatch.setattr(module,'fit_pose_rgb_then_depth',
                        lambda m,*a,**k: first if np.array_equal(m,left) else second)
    p._batch_workspace = batch_workspace()
    p.use_tf_up = True
    p._base_from_camera = lambda *a: (np.eye(3),np.zeros(3))
    p._process(c,d)
    np.testing.assert_allclose(p._lock_pos,first['t'])
    # First box now sits at the place target and still has the better fit.
    first['t'] = np.array([.35,-.12,.04])
    p.pub.messages.clear()
    p._lock_seen = time.monotonic()-3.  # transfer took longer than lock timeout
    p._process(c,d)
    assert len(p.pub.messages) == 1
    np.testing.assert_allclose(p._lock_pos,second['t'])
    # All candidates in the destination region: no pickup output at all.
    second['t'] = np.array([.47,-.12,.04])
    p.pub.messages.clear()
    for _ in range(20): p._process(c,d)
    assert not p.pub.messages


def test_batch_releases_lock_that_moved_outside_source(scene):
    p,c,d,fit = scene
    p._batch_workspace = batch_workspace()
    p._base_from_camera = lambda *a: (np.eye(3),np.zeros(3))
    p._lock_frame = p.base_frame
    p._lock_pos = np.array([.35,-.12,.04])
    p._lock_seen = time.monotonic()
    fit['t'] = np.array([.5,.15,.04])
    p._process(c,d)
    assert len(p.pub.messages) == 1
    np.testing.assert_allclose(p._lock_pos,fit['t'])


def test_batch_never_uses_camera_coordinates_as_workspace(scene):
    p,c,d,fit = scene
    p._batch_workspace = batch_workspace()
    fit['t'] = np.array([.4,.1,.04])
    p._process(c,d)  # TF absent even though offline Probe has use_tf_up=false
    assert not p.pub.messages


def test_rgbd_backend_has_no_six_vertex_admission_gate(scene,monkeypatch):
    p,c,d,fit=scene
    p.pose_solver='rgbd'
    m=np.zeros((100,100),np.uint8)
    cv2.fillPoly(m,[np.array([[30,25],[62,25],[73,45],[58,71],[30,64]])],1)
    p.seg.masks=[m.astype(bool)]
    seen=[]
    result=dict(fit,accepted=True,reason='ok',vertex_count=5,timing_ms={'total':1.})
    monkeypatch.setattr(module,'fit_cuboid_rgbd',lambda mask,*a,**k:(seen.append(mask) or result))
    p._process(c,d)
    assert len(seen)==1 and len(p.pub.messages)==1
    assert p._last_timing.counts['vertices_5']==1


def test_ambiguous_rgbd_pose_is_not_published(scene,monkeypatch):
    p,c,d,fit=scene;p.pose_solver='rgbd'
    result=dict(fit,accepted=False,reason='ambiguous',vertex_count=4,timing_ms={'total':1.})
    monkeypatch.setattr(module,'fit_cuboid_rgbd',lambda *a,**kw:result)
    p._process(c,d)
    assert not p.pub.messages
    assert p._last_timing.counts['pose_ambiguous']==1
    assert p._last_selected_fit is None


def test_multi_model_packet_contains_selected_geometry_and_square_symmetry(scene,monkeypatch):
    import json
    from piper_pnp.object_models import ObjectCatalog
    p,c,d,fit=scene;p.pose_solver='rgbd'
    p.object_catalog=ObjectCatalog({'version':1,'models':[
        {'id':'small','dims_m':[.077,.035,.03]}, {'id':'square','dims_m':[.08,.08,.0415]}]})
    p.use_tf_up=True
    p._base_from_camera=lambda *a:(np.eye(3),np.zeros(3))
    result=dict(fit,accepted=True,reason='ok',model_id='square',dims=(.08,.08,.0415),
                vertex_count=6,timing_ms={'total':1.})
    monkeypatch.setattr(module,'fit_cuboid_models',lambda *a,**kw:result)
    p._process(c,d)
    first=json.loads(p.target_pub.messages[-1].data)
    assert first['model_id']=='square' and first['dims_m']==[.08,.08,.0415]
    assert not p.pub.messages and len(p.model_markers_pub.messages)==1
    assert abs(first['box_in_grasp_position'][2])==pytest.approx(.0415/2)
    result['R']=cv2.Rodrigues(np.array([0.,0.,np.pi/2]))[0]
    p._process(c,d)
    second=json.loads(p.target_pub.messages[-1].data)
    np.testing.assert_allclose(second['orientation'],first['orientation'],atol=1e-8)
    assert first['track_id']==second['track_id']


def test_ambiguous_model_is_never_published(scene,monkeypatch):
    from piper_pnp.object_models import ObjectCatalog
    p,c,d,fit=scene;p.pose_solver='rgbd'
    p.object_catalog=ObjectCatalog({'version':1,'models':[{'id':'square','dims_m':[.08,.08,.0415]}]})
    result=dict(fit,accepted=False,reason='ambiguous_model',timing_ms={'total':1.})
    monkeypatch.setattr(module,'fit_cuboid_models',lambda *a,**kw:result)
    p._process(c,d)
    assert not p.pub.messages and not p.target_pub.messages
