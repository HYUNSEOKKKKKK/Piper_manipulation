"""Metric model identity, partial visibility, square symmetries and bounded cost."""
import numpy as np
import pytest
import cuboid_rgbd as engine
from test_cuboid_rgbd import render, rotation, K

MODELS=[dict(id='small',dims=(.077,.035,.030)),
        dict(id='square',dims=(.080,.080,.0415))]


@pytest.mark.parametrize('model',MODELS,ids=lambda m:m['id'])
@pytest.mark.parametrize('angles',[(0,0,0),(25,35,15),(65,-20,50),(95,10,0),(130,45,25),(-20,70,-50)])
def test_model_and_pose_with_arbitrary_orientation(model,angles):
    R=rotation(angles);t=np.array([.01,-.012,.44])
    m,d=render(R,t,dims=np.array(model['dims']))
    f=engine.fit_cuboid_models(m,d,MODELS,K)
    assert f['accepted'] and f['model_id']==model['id'],f
    assert np.linalg.norm(f['t']-t)<.004,f
    assert engine.symmetry_angle(R,f['R'],model['dims'])<8,f
    assert f['n_refined']<=4 and f['sampled_points']<=192


def test_square_90_degree_symmetry_is_model_specific():
    R=rotation((0,0,90))
    assert len(engine.cuboid_symmetries(MODELS[0]['dims']))==4
    assert len(engine.cuboid_symmetries(MODELS[1]['dims']))==8
    assert len(engine.cuboid_symmetries((.08,.08,.08)))==24
    assert engine.symmetry_angle(np.eye(3),R,MODELS[1]['dims'])<1e-6
    assert engine.symmetry_angle(np.eye(3),R,MODELS[0]['dims'])>89
    np.testing.assert_allclose(engine.align_symmetric_rotation(R,np.eye(3),MODELS[1]['dims']),np.eye(3),atol=1e-8)


@pytest.mark.parametrize('dims,count',[(MODELS[0]['dims'],12),(MODELS[1]['dims'],6),((.08,.08,.08),2)])
def test_correspondence_reduction_preserves_every_model_geometry(dims,count):
    from cuboid_pnp_correspondence import model_hexagon_hypotheses,enumerate_hexagon_hypotheses
    group=engine.cuboid_symmetries(dims)
    reduced=model_hexagon_hypotheses(dims,tuple(group.ravel()))
    assert len(reduced)==count
    assert all(any(np.allclose(h@S,b,rtol=0,atol=1e-12) for b in reduced for S in group)
               for h in enumerate_hexagon_hypotheses(dims))


def test_observation_and_planes_shared_between_models(monkeypatch):
    called={'observation':0,'planes':0}
    original_observation,original_planes=engine.Observation,engine._planes
    def observation(*a,**kw):
        called['observation']+=1
        return original_observation(*a,**kw)
    def planes(*a,**kw):
        called['planes']+=1
        return original_planes(*a,**kw)
    monkeypatch.setattr(engine,'Observation',observation)
    monkeypatch.setattr(engine,'_planes',planes)
    m,d=render(rotation((25,35,15)),np.array([0.,0.,.44]))
    f=engine.fit_cuboid_models(m,d,MODELS,K)
    assert f['accepted'] and called=={'observation':1,'planes':1}


def test_unseen_thickness_cannot_be_classified_from_identical_front_face():
    models=[dict(id='thin',dims=(.08,.08,.0415)),dict(id='thick',dims=(.08,.08,.065))]
    m,d=render(np.eye(3),np.array([0.,0.,.44]),dims=np.array(models[0]['dims']),noise=0)
    f=engine.fit_cuboid_models(m,d,models,K)
    assert not f['accepted'] and f['reason']=='ambiguous_model',f


def test_rgb_scale_ambiguity_resolved_by_depth():
    models=[MODELS[0],dict(id='double',dims=tuple(2*x for x in MODELS[0]['dims']))]
    R=rotation((30,20,10));t=np.array([0.,0.,.44])
    m,d=render(R,t)
    f=engine.fit_cuboid_models(m,d,models,K)
    assert f['accepted'] and f['model_id']=='small',f
