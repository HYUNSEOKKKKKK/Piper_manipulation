"""The detector reuses preparation, never detections or input frames."""
from types import SimpleNamespace
import ros_pose_bridge  # selects this workspace's vendored MobileSAMv2
from mobilesamv2.promt_mobilesamv2 import model as module


def test_backend_prepared_once_but_new_frames_and_thresholds_are_used(monkeypatch):
    setup=[];inputs=[]
    class Predictor:
        def __init__(self,overrides):self.args=SimpleNamespace(**overrides)
        def setup_model(self,model,verbose):setup.append(model)
        def __call__(self,source,stream):
            inputs.append(source)
            return (source,self.args.conf)
    monkeypatch.setattr(module,'PromptModelPredictor',Predictor)
    monkeypatch.setattr(module,'get_cfg',lambda old,new:SimpleNamespace(**{**vars(old),**new}))
    detector=module.ObjectAwareModel.__new__(module.ObjectAwareModel)
    detector.overrides={};detector.model=object();detector.predictor=None
    assert detector.predict('frame-a',device='cpu',conf=.4)==('frame-a',.4)
    assert detector.predict('frame-b',device='cpu',conf=.7)==('frame-b',.7)
    assert len(setup)==1 and inputs==['frame-a','frame-b']
    detector.predict('frame-c',device='cuda',conf=.7)
    assert len(setup)==2  # changing the backend device must rebuild it
