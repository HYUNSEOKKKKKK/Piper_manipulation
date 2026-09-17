"""Shared metric model catalog and atomic perception packet; no ROS imports."""
import hashlib
import json
import math
from pathlib import Path


def vector(value, n):
    if (not isinstance(value,(list,tuple)) or len(value)!=n
            or not all(type(x) in (int,float) and math.isfinite(x) for x in value)):
        raise ValueError('Invalid finite vector')
    return tuple(float(x) for x in value)


def quaternion(value):
    q=vector(value,4);norm=math.sqrt(sum(x*x for x in q))
    if abs(norm-1.)>.01:raise ValueError('Invalid unit quaternion')
    return tuple(x/norm for x in q)


class ObjectCatalog:
    def __init__(self, config):
        if config.get('version')!=1:raise ValueError('Unsupported object catalog version')
        self.models=[];self.by_id={};shapes=set()
        for item in config['models']:
            key=item['id'];dims=vector(item['dims_m'],3)
            if (not isinstance(key,str) or not key or len(key)>64 or key in self.by_id
                    or min(dims)<=0 or max(dims)>.5 or tuple(sorted(dims)) in shapes):
                raise ValueError('Invalid or duplicate metric model')
            mode=item.get('place_mode','geometry')
            clearance=float(item.get('release_clearance_m',.003))
            if mode not in ('legacy','geometry') or not 0<=clearance<=.02:
                raise ValueError('Invalid placement policy')
            model=dict(id=key,dims=dims,place_mode=mode,release_clearance_m=clearance)
            self.models.append(model);self.by_id[key]=model;shapes.add(tuple(sorted(dims)))
        if not 1<=len(self.models)<=2:raise ValueError('This deployment supports one or two models')
        self.digest=hashlib.sha256(json.dumps(self.models,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    def validate_workspace(self,workspace):
        for m in self.models:
            if m['place_mode']=='geometry':
                envelope=workspace.obstacle_size
                if (m['dims'][0]+.01>envelope[0] or m['dims'][1]+.01>envelope[1]
                        or m['dims'][2]+m['release_clearance_m']+.005>envelope[2]):
                    raise ValueError('Placed-box collision envelope is too small for '+m['id'])

    def decode(self, text):
        if not isinstance(text,str) or len(text)>8192:raise ValueError('Invalid target packet')
        p=json.loads(text)
        if p.get('version')!=1 or p.get('catalog_digest')!=self.digest:
            raise ValueError('Perception/controller object catalogs do not match')
        model=self.by_id[p['model_id']]
        if vector(p['dims_m'],3)!=model['dims']:raise ValueError('Target dimensions do not match model')
        stamp=p['stamp_ns'];frame=p['frame_id'];track=p['track_id']
        if (type(stamp) is not int or stamp<=0 or not isinstance(frame,str) or not frame
                or type(track) is not int or track<1):raise ValueError('Invalid acquisition identity')
        position=vector(p['position'],3);orientation=quaternion(p['orientation'])
        relative_position=vector(p['box_in_grasp_position'],3)
        relative_orientation=quaternion(p['box_in_grasp_orientation'])
        if math.sqrt(sum(x*x for x in relative_position))>math.sqrt(sum(x*x for x in model['dims'])):
            raise ValueError('Box center is inconsistent with grasp frame')
        metadata=dict(model_id=model['id'],track_id=track,dims=model['dims'],
                      place_mode=model['place_mode'],release_clearance_m=model['release_clearance_m'],
                      box_in_grasp_position=relative_position,box_in_grasp_orientation=relative_orientation)
        if 'workspace_observation' in p:
            # Workspace-specific validation is performed by the controller;
            # this observation has exactly the target's acquisition stamp.
            metadata['workspace_observation'] = p['workspace_observation']
        return stamp,frame,position,orientation,metadata
