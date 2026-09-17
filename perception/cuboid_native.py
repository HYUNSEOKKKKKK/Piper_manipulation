"""Optional native residual acceleration; the NumPy implementation remains usable."""
import ctypes as C
import hashlib
from pathlib import Path
import numpy as np

DP=C.POINTER(C.c_double);FP=C.POINTER(C.c_float);BP=C.POINTER(C.c_ubyte)

class Data(C.Structure):
    _fields_=[(k,C.c_int) for k in ('n','c','m','w','h','full_w','full_h','x0','y0')]+[
        (k,DP) for k in ('rays','z','points','contour')]+[(k,FP) for k in ('depth','dt')]+[
        ('mask',BP)]+[(k,C.c_double) for k in ('fx','fy','cx','cy','jump','depth_sigma','contour_sigma')]

ROOT=Path(__file__).parent/'native'
LIB=None
try:
    if (ROOT/'build/source.sha256').read_text().strip()==hashlib.sha256((ROOT/'cuboid_residual.cpp').read_bytes()).hexdigest():
        LIB=C.CDLL(str(ROOT/'build/libcuboid_residual.so'))
        LIB.cuboid_residual_v1.argtypes=[C.POINTER(Data),DP,DP,DP,DP,DP]
        LIB.cuboid_residual_v1.restype=C.c_int
except (OSError,AttributeError):
    LIB=None


class NativeObservation:
    def __init__(self,obs):
        self.refs={k:np.ascontiguousarray(getattr(obs,k),dtype=np.float64) for k in ('rays','z','points','contour')}
        self.refs.update({k:np.ascontiguousarray(getattr(obs,k),dtype=np.float32) for k in ('depth','dt')})
        self.refs['mask']=np.ascontiguousarray(obs.mask,dtype=np.uint8)
        h,w=obs.mask.shape
        self.data=Data(len(obs.points),len(obs.contour),obs.cfg.contour_points,w,h,obs.full_shape[1],obs.full_shape[0],*map(int,obs.offset),
                       *[self.refs[k].ctypes.data_as(DP) for k in ('rays','z','points','contour')],
                       *[self.refs[k].ctypes.data_as(FP) for k in ('depth','dt')],self.refs['mask'].ctypes.data_as(BP),
                       obs.K[0,0],obs.K[1,1],obs.K[0,2],obs.K[1,2],obs.cfg.discontinuity_m,obs.cfg.depth_sigma_m,obs.cfg.contour_sigma_px)

    def residual(self,R,t,dims,details):
        R,t,dims=[np.ascontiguousarray(a,dtype=np.float64) for a in (R,t,dims)]
        residual=np.empty(self.data.n+self.data.c+self.data.m,dtype=np.float64)
        quality=np.empty(5,dtype=np.float64) if details else None
        ok=LIB.cuboid_residual_v1(C.byref(self.data),R.ctypes.data_as(DP),t.ctypes.data_as(DP),dims.ctypes.data_as(DP),residual.ctypes.data_as(DP),quality.ctypes.data_as(DP) if details else None)
        if not ok:return None
        if details:return residual,dict(zip(('depth_error_m','depth_p90_m','ray_hit_fraction','contour_err_px','visible_model_fraction'),map(float,quality)))
        return residual
