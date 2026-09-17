"""Pose tests with independent six-face rendering and known SE(3) ground truth."""
import cv2
import numpy as np
import pytest
from cuboid_rgbd import FitConfig, Observation, fit_cuboid_rgbd, symmetry_angle

DIMS=np.array([.077,.035,.030])
K=np.array([[420.,0,160.],[0,420.,120.],[0,0,1.]])


def render(R,t,shape=(240,320),K=K,dims=DIMS,noise=.0005,seed=1):
    """Intersect each of the six finite rectangles independently; nearest wins."""
    yy,xx=np.indices(shape);rays=np.stack(((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones(shape)),axis=-1)
    depth=np.full(shape,np.inf)
    for axis in range(3):
        for sign in (-1,1):
            normal=sign*R[:,axis];center=t+normal*dims[axis]/2
            denominator=rays@normal
            with np.errstate(divide='ignore',invalid='ignore'):
                z=(center@normal)/denominator
                local=(rays*z[...,None]-t)@R
            valid=(z>0)&np.isfinite(z)
            for a in range(3):
                if a!=axis:valid &= np.abs(local[...,a])<=dims[a]/2+1e-9
            depth=np.where(valid& (z<depth),z,depth)
    mask=np.isfinite(depth)
    rng=np.random.default_rng(seed)
    depth[mask]+=rng.normal(0,noise,int(mask.sum()))
    depth[~mask]=.9
    return mask,depth.astype(np.float32)


def rotation(degrees):
    return cv2.Rodrigues(np.radians(degrees).astype(float))[0]


@pytest.mark.parametrize('angles',[(0,0,0),(25,35,15),(65,-20,50),(95,10,0),(130,45,25),(-20,70,-50)])
def test_arbitrary_orientation_no_table_constraint(angles):
    R=rotation(angles);t=np.array([.01,-.012,.44]);mask,d=render(R,t)
    fit=fit_cuboid_rgbd(mask,d,DIMS,K)
    assert fit['accepted'],fit
    assert np.linalg.norm(fit['t']-t)<.004,fit
    assert symmetry_angle(R,fit['R'])<8,fit
    assert fit['sampled_points']<=192 and fit['n_refined']<=4


def test_corrupt_depth_boundary_is_excluded():
    R=rotation([30,-20,20]);t=np.array([0.,0.,.42]);m,d=render(R,t)
    interior=cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    edge=m&~interior;bad=d.copy();bad[edge]=np.where(np.indices(m.shape)[0][edge]%2,0,.75)
    fit=fit_cuboid_rgbd(m,bad,DIMS,K)
    assert fit['accepted'],fit
    assert np.linalg.norm(fit['t']-t)<.004
    assert symmetry_angle(R,fit['R'])<8


def test_no_depth_cannot_become_a_robot_target():
    m,d=render(np.eye(3),np.array([0.,0.,.4]));d[:]=0
    fit=fit_cuboid_rgbd(m,d,DIMS,K)
    assert not fit['accepted'] and fit['reason']=='insufficient_clean_depth'


def test_noncuboid_background_is_rejected():
    m=np.zeros((240,320),np.uint8);cv2.circle(m,(160,120),45,1,-1)
    fit=fit_cuboid_rgbd(m,np.full(m.shape,.4),DIMS,K)
    assert not fit['accepted'],fit


def test_exact_symmetry_does_not_equate_30_and_35_mm_axes():
    assert symmetry_angle(np.eye(3),np.diag([1.,-1.,-1.]))<1e-6
    assert symmetry_angle(np.eye(3),rotation([90,0,0]))>89


def test_native_matches_reference_for_perturbed_hypotheses():
    R=rotation([30,20,10]);t=np.array([.01,.02,.45]);m,d=render(R,t)
    obs=Observation(m,d,K,FitConfig())
    if obs.native is None:pytest.skip('optional native module not built')
    for angles,shift in [([0,0,0],[0,0,0]),([2,-3,1],[.003,-.002,.004]),([20,10,-15],[.015,0,-.01])]:
        a=rotation(angles)@R;b=t+shift
        actual=obs.residual(a,b,DIMS);reference=obs.residual_numpy(a,b,DIMS)
        np.testing.assert_allclose(actual,reference,atol=2e-5,rtol=2e-5)


def test_optimizer_accepts_only_cost_decreasing_steps():
    R=rotation([20,-40,15]);t=np.array([0.,0.,.45]);m,d=render(R,t)
    fit=fit_cuboid_rgbd(m,d,DIMS,K)
    assert fit['cost']<=fit['initial_cost']
    np.testing.assert_allclose(fit['R'].T@fit['R'],np.eye(3),atol=1e-8)
    assert np.linalg.det(fit['R'])>.99999


def test_reduced_pnp_keeps_every_physical_correspondence():
    from cuboid_rgbd import SYMMETRIES
    from cuboid_pnp_correspondence import enumerate_hexagon_hypotheses
    hypotheses=enumerate_hexagon_hypotheses(DIMS)
    assert len(hypotheses)==48
    assert all(any(np.allclose(h@S,b,rtol=0,atol=1e-12)
                   for b in hypotheses[:12] for S in SYMMETRIES) for h in hypotheses)


@pytest.mark.parametrize('kernel,expected_vertices',[(1,4),(7,8)])
def test_frontal_and_rounded_non_six_contours(kernel,expected_vertices):
    R=np.eye(3);t=np.array([0.,0.,.4]);m,d=render(R,t)
    if kernel>1:
        m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel))).astype(bool)
    f=fit_cuboid_rgbd(m,d,DIMS,K)
    assert f['vertex_count']==expected_vertices and f['accepted'],f
    assert np.linalg.norm(f['t']-t)<.003
    assert symmetry_angle(f['R'],R)<5


def test_tiny_mask_holes_do_not_destroy_depth_support():
    R=rotation([45,30,12]);t=np.array([0.,0.,.4]);m,d=render(R,t)
    yy,xx=np.indices(m.shape);m[m&((xx+3*yy)%17==0)]=False
    f=fit_cuboid_rgbd(m,d,DIMS,K)
    assert f['accepted'],f
    assert np.linalg.norm(f['t']-t)<.004 and symmetry_angle(R,f['R'])<8


def test_fully_occluded_extent_is_not_an_arbitrary_pose():
    m,d=render(np.eye(3),np.array([0.,0.,.4]))
    hidden=np.ones(m.shape,bool);hidden[110:130,145:175]=False
    m[hidden]=False;d[hidden]=.25
    f=fit_cuboid_rgbd(m,d,DIMS,K)
    assert not f['accepted'] and f['reason']=='insufficient_visible_contour'


def test_foreground_corner_occlusion_with_five_or_seven_vertices():
    rng=np.random.default_rng(20);seen=set()
    for i in range(32):
        R=rotation(rng.uniform(-140,140,3));t=np.r_[rng.uniform(-.02,.02,2),rng.uniform(.36,.55)]
        if i%4!=2:continue
        m,d=render(R,t,seed=i);ys,xs=np.where(m);yy,xx=np.indices(m.shape)
        hidden=(xx>np.quantile(xs,.7))&(yy<np.quantile(ys,.5));d[hidden]=.25;m[hidden]=False
        f=fit_cuboid_rgbd(m,d,DIMS,K)
        assert f['accepted'],f
        assert np.linalg.norm(f['t']-t)<.005 and symmetry_angle(R,f['R'])<12,f
        seen.add(f['vertex_count'])
    assert {5,7}<=seen


def test_partially_offscreen_box_uses_visible_depth_and_edges():
    R=rotation([25,35,15]);t=np.array([.13,0.,.4]);m,d=render(R,t)
    f=fit_cuboid_rgbd(m,d,DIMS,K)
    assert f['accepted'] and f['partial_contour_fraction']>.15,f
    assert np.linalg.norm(f['t']-t)<.004 and symmetry_angle(R,f['R'])<8


def test_numpy_fallback_without_compiled_library(monkeypatch):
    import cuboid_rgbd
    monkeypatch.setattr(cuboid_rgbd, 'LIB', None)
    R=rotation([25,35,15]);t=np.array([.01,-.01,.4]);m,d=render(R,t)
    f=fit_cuboid_rgbd(m,d,DIMS,K)
    assert f['accepted'],f
    assert np.linalg.norm(f['t']-t)<.004 and symmetry_angle(R,f['R'])<8
