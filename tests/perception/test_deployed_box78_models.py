"""The current two physical boxes, including genuinely ambiguous shared faces."""
from pathlib import Path
import numpy as np
import pytest
from piper_pnp.object_models import ObjectCatalog
from cuboid_rgbd import fit_cuboid_models,symmetry_angle
from test_cuboid_rgbd import render,rotation,K

CATALOG=ObjectCatalog.load(Path(__file__).resolve().parents[2] / 'config/object_models.json')


@pytest.mark.parametrize('model',CATALOG.models,ids=lambda m:m['id'])
@pytest.mark.parametrize('angles',[(0,0,0),(25,35,15),(65,-20,50)])
def test_current_models_are_selected_with_bounded_refinement(model,angles):
    R=rotation(angles);t=np.array([.01,-.012,.44])
    mask,depth=render(R,t,dims=np.array(model['dims']))
    fit=fit_cuboid_models(mask,depth,CATALOG.models,K)
    assert fit['accepted'] and fit['model_id']==model['id'],fit
    assert np.linalg.norm(fit['t']-t)<.004
    assert symmetry_angle(R,fit['R'],model['dims'])<8
    assert fit['n_refined']<=4 and fit['sampled_points']<=192


@pytest.mark.parametrize('model',CATALOG.models,ids=lambda m:m['id'])
def test_shared_78_by_30_face_is_not_forced_into_a_model(model):
    mask,depth=render(rotation((95,10,0)),np.array([.01,-.012,.44]),dims=np.array(model['dims']))
    fit=fit_cuboid_models(mask,depth,CATALOG.models,K)
    assert not fit['accepted'] and fit['reason']=='ambiguous_model',fit


def test_deployment_contains_only_the_two_requested_shapes():
    assert {tuple(sorted(m['dims'])) for m in CATALOG.models}=={
        (.030,.053,.078),(.030,.035,.078)}
