from pathlib import Path
import time
import numpy as np
import pytest
from piper_pnp.batch_workspace import BatchWorkspace
from sweep_depth import observe_workspace


def scene():
    w=BatchWorkspace.load(Path(__file__).resolve().parents[2] / 'config/sweep_workspace.json')
    # Camera looking vertically down; visible world rectangle covers both zones.
    K=np.array([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]])
    R=np.diag([1.,-1.,-1.]);t=np.array([.4,.02,.6])
    depth=np.full((480,640),.6,np.float32)
    return w,K,R,t,depth


def test_flat_table_has_full_coverage_and_zero_height():
    w,K,R,t,d=scene()
    result=observe_workspace(d,K,R,t,w)
    assert result['drop']['coverage']==1.
    assert result['source']['coverage']==1.
    assert result['drop']['height_m']==pytest.approx(0.,abs=1e-6)


def test_a_small_high_patch_is_not_erased_by_percentile_ranking():
    w,K,R,t,d=scene()
    # Pixel ~ (280,352) is inside goal at z=.10.
    d[346:362,274:290]=.5
    result=observe_workspace(d,K,R,t,w)
    assert result['drop']['height_m']==pytest.approx(.1,abs=1e-6)


def test_missing_depth_cannot_look_like_empty_goal():
    w,K,R,t,d=scene();d[:]=0.
    result=observe_workspace(d,K,R,t,w)
    assert result['drop']['coverage']==0.
    assert result['drop']['height_m'] is None


def test_single_depth_outlier_does_not_invent_a_pile():
    w,K,R,t,d=scene();d[350,280]=.2
    assert observe_workspace(d,K,R,t,w)['drop']['height_m']<.001


def test_feed_drop_observes_only_shared_goal_without_source_depth_requirement():
    _,K,R,t,d=scene()
    w=BatchWorkspace.load(Path(__file__).resolve().parents[2] / 'config/feed_auto_workspace.json')
    result=observe_workspace(d,K,R,t,w)
    assert result['drop']['coverage']==1.
    assert result['drop']['height_m']==pytest.approx(0.,abs=1e-6)
    assert result['source'] is None
