"""Bounded, table-independent RGB-D fitting of a known cuboid.

Silhouette vertex count is a source of optional initial guesses, never an
admission test. Depth planes provide general SE(3) guesses. A small set of
visible depth rays and contour samples refine them; unresolved hypotheses
return an explicit rejection instead of an arbitrary robot target.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
import time
import cv2
import numpy as np

from cuboid_pnp_correspondence import cuboid_corners_3d, solve_pnp_hexagon
from cuboid_native import LIB, NativeObservation

SYMMETRIES = np.array([np.diag(s) for s in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1))])


@dataclass(frozen=True)
class FitConfig:
    max_points: int = 192
    contour_points: int = 64
    refine_candidates: int = 4
    iterations: int = 5
    erosion_px: int = 2
    discontinuity_m: float = .008
    plane_tolerance_m: float = .0025
    depth_sigma_m: float = .002
    contour_sigma_px: float = 1.5
    max_depth_error_m: float = .005
    max_contour_error_px: float = 3.5
    ambiguity_cost: float = .15
    ambiguity_rotation_deg: float = 12.
    ambiguity_translation_m: float = .006

    def __post_init__(self):
        if not (32 <= self.max_points <= 512 and 16 <= self.contour_points <= 128
                and 1 <= self.refine_candidates <= 8 and 1 <= self.iterations <= 12
                and 1 <= self.erosion_px <= 5):
            raise ValueError('Invalid bounded RGB-D fitting budget')
        for name in ('discontinuity_m','plane_tolerance_m','depth_sigma_m','contour_sigma_px',
                     'max_depth_error_m','max_contour_error_px','ambiguity_cost',
                     'ambiguity_rotation_deg','ambiguity_translation_m'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError('Invalid fit threshold: '+name)


@lru_cache(maxsize=32)
def cuboid_symmetries(dims):
    """Proper rotations preserving metric dimensions: 4 / 8 / 24 elements.

    Equality is numerical equality, not a tolerance for similar physical sizes.
    In particular, 30 and 35 mm axes are never interchangeable.
    """
    d = np.asarray(dims)
    result = []
    for perm in permutations(range(3)):
        if not np.allclose(d[list(perm)], d, rtol=0, atol=1e-9):
            continue
        for signs in product((-1, 1), repeat=3):
            S = np.eye(3)[:, perm] @ np.diag(signs)
            if np.linalg.det(S) > .5:
                result.append(S)
    return np.asarray(result)


def symmetry_angle(R1, R2, dims=None):
    """Angular discrepancy modulo the given model's exact proper symmetries."""
    rel = R1.T @ R2
    group = SYMMETRIES if dims is None else cuboid_symmetries(tuple(dims))
    trace = np.einsum('sij,ji->s', group, rel)
    return float(np.degrees(np.arccos(np.clip((trace.max()-1.)/2.,-1.,1.))))


def align_symmetric_rotation(R, reference, dims):
    variants = R @ cuboid_symmetries(tuple(dims))
    score = np.einsum('ij,sij->s', reference, variants)
    return variants[int(np.argmax(score))]


def _rays(xy, K):
    return np.column_stack(((xy[:,0]-K[0,2])/K[0,0], (xy[:,1]-K[1,2])/K[1,1], np.ones(len(xy))))


def _project(points, K):
    return points[:,:2]/points[:,2,None]*[K[0,0],K[1,1]] + [K[0,2],K[1,2]]


def _uniform_rows(a, count):
    return a[np.linspace(0,len(a)-1,min(len(a),count)).astype(int)]


def _perimeter(poly, count):
    nxt=np.roll(poly,-1,axis=0);length=np.linalg.norm(nxt-poly,axis=1)
    cumulative=np.r_[0.,np.cumsum(length)]
    s=np.arange(count)*cumulative[-1]/count
    i=np.minimum(np.searchsorted(cumulative,s,side='right')-1,len(poly)-1)
    return poly[i]+((s-cumulative[i])/np.maximum(length[i],1e-8))[:,None]*(nxt[i]-poly[i])


def _distance_to_polygon(points, poly):
    v=np.roll(poly,-1,axis=0)-poly
    d=points[:,None,:]-poly[None,:,:]
    u=np.clip(np.einsum('nki,ki->nk',d,v)/np.maximum(np.sum(v*v,axis=1),1e-8),0,1)
    return np.sqrt(np.min(np.sum((d-u[:,:,None]*v)**2,axis=2),axis=1))


def _ray_box(rays, R, t, half):
    direction=rays@R
    origin=-t@R
    safe=np.where(np.abs(direction)>1e-10,direction,np.where(direction>=0,1e-10,-1e-10))
    a=(-half-origin)/safe;b=(half-origin)/safe
    near=np.minimum(a,b).max(axis=1);far=np.maximum(a,b).min(axis=1)
    return near,(far>=near-1e-6)&(near>0)


def _box_distance(points, R, t, half):
    q=np.abs((points-t)@R)-half
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)


class Observation:
    def __init__(self, mask, depth, K, cfg):
        self.K,self.cfg,self.full_shape=K,cfg,mask.shape
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours: raise ValueError('empty_mask')
        contour=max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour)<40: raise ValueError('insufficient_area')
        x,y,w,h=cv2.boundingRect(contour);pad=12
        x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(mask.shape[1],x+w+pad),min(mask.shape[0],y+h+pad)
        self.offset=np.array([x0,y0]);self.depth=np.asarray(depth[y0:y1,x0:x1],np.float32)
        self.mask=mask[y0:y1,x0:x1].astype(bool)
        # Keep the largest component; separate SAM fragments must not invent a cuboid.
        component=np.zeros(self.mask.shape,np.uint8)
        local=contour.reshape(-1,2)-self.offset
        cv2.fillPoly(component,[local],1);self.mask &= component.astype(bool)
        valid=np.isfinite(self.depth)&(self.depth>.05)&(self.depth<3.)
        kernel=np.ones((3,3),np.uint8)
        lo=cv2.erode(np.where(valid,self.depth,0).astype(np.float32),kernel)
        hi=cv2.dilate(np.where(valid,self.depth,10).astype(np.float32),kernel)
        closed=cv2.morphologyEx(self.mask.astype(np.uint8),cv2.MORPH_CLOSE,kernel)
        interior=cv2.erode(closed,np.ones((2*cfg.erosion_px+1,)*2,np.uint8)).astype(bool)
        good=interior&valid&(lo>0)&((hi-lo)<cfg.discontinuity_m)
        ys,xs=np.where(good)
        self.support_count=len(xs)
        self.raw_valid_count=int((self.mask&valid).sum())
        if len(xs)<24: raise ValueError('insufficient_clean_depth')
        xy=_uniform_rows(np.column_stack((xs,ys)),cfg.max_points)
        self.rays=_rays(xy+self.offset,K)
        self.z=self.depth[xy[:,1],xy[:,0]].astype(float)
        self.points=self.rays*self.z[:,None]
        # Depth only identifies occluding foreground outside this instance.
        outside_min=cv2.erode(np.where(valid&~self.mask,self.depth,10).astype(np.float32),np.ones((7,7),np.uint8))
        inside_max=cv2.dilate(np.where(valid&self.mask,self.depth,0).astype(np.float32),np.ones((7,7),np.uint8))
        xy_full=contour.reshape(-1,2)
        clipped=(xy_full[:,0]<=1)|(xy_full[:,0]>=mask.shape[1]-2)|(xy_full[:,1]<=1)|(xy_full[:,1]>=mask.shape[0]-2)
        occluded=outside_min[local[:,1],local[:,0]]+cfg.discontinuity_m<inside_max[local[:,1],local[:,0]]
        usable=~clipped&~occluded
        self.partial_fraction=float(1-usable.mean())
        self.partial=self.partial_fraction>.08
        if usable.sum()<12: raise ValueError('insufficient_visible_contour')
        self.contour=_uniform_rows(xy_full[usable],cfg.contour_points).astype(float)
        boundary=np.full(self.mask.shape,255,np.uint8)
        good_boundary=local[usable];boundary[good_boundary[:,1],good_boundary[:,0]]=0
        self.dt=cv2.distanceTransform(boundary,cv2.DIST_L2,5)
        hull=cv2.convexHull(contour)
        # Natural polygon; no epsilon sweep forcing a chosen vertex count.
        eps=max(1.0,cv2.arcLength(hull,True)*.008)
        self.polygon=cv2.approxPolyDP(hull,eps,True).reshape(-1,2)
        self.native=NativeObservation(self) if LIB is not None else None

    def residual(self, R, t, dims, details=False):
        if self.native is not None:
            return self.native.residual(R,t,dims,details)
        return self.residual_numpy(R,t,dims,details)

    def residual_numpy(self, R, t, dims, details=False):
        half=np.asarray(dims)/2
        corners=cuboid_corners_3d(dims)@R.T+t
        if not np.isfinite(corners).all() or np.min(corners[:,2])<=.03:
            return None
        hull=cv2.convexHull(_project(corners,self.K).astype(np.float32)).reshape(-1,2).astype(float)
        if len(hull)<3: return None
        hull=np.roll(hull,-np.lexsort((hull[:,1],hull[:,0]))[0],axis=0)
        edge=_perimeter(hull,self.cfg.contour_points)
        predicted,hit=_ray_box(self.rays,R,t,half)
        depth_res=np.where(hit,predicted-self.z,_box_distance(self.points,R,t,half))
        # A miss needs a nonzero geometric residual even close to a grazing ray.
        depth_res=np.where(hit,depth_res,np.abs(depth_res)+.002)
        observed_dist=_distance_to_polygon(self.contour,hull)
        loc=edge-self.offset
        h,w=self.mask.shape
        cx=np.clip(loc[:,0],0,w-1.001);cy=np.clip(loc[:,1],0,h-1.001)
        ix=cx.astype(int);iy=cy.astype(int);wx=cx-ix;wy=cy-iy
        rendered_dist=(1-wy)*((1-wx)*self.dt[iy,ix]+wx*self.dt[iy,ix+1])+wy*((1-wx)*self.dt[iy+1,ix]+wx*self.dt[iy+1,ix+1])
        rendered_dist += np.linalg.norm(loc-np.column_stack((cx,cy)),axis=1)
        pix=np.rint(loc).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
        pix[:,0]=np.clip(pix[:,0],0,w-1);pix[:,1]=np.clip(pix[:,1],0,h-1)
        meas=self.depth[pix[:,1],pix[:,0]]
        pred_edge,_=_ray_box(_rays(edge,self.K),R,t,half)
        hidden=inside&~self.mask[pix[:,1],pix[:,0]]&np.isfinite(meas)&(meas>.05)&(meas+self.cfg.discontinuity_m<pred_edge)
        out_of_image=(edge[:,0]<1)|(edge[:,0]>=self.full_shape[1]-2)|(edge[:,1]<1)|(edge[:,1]>=self.full_shape[0]-2)
        visible=~hidden&~out_of_image
        rendered_dist=np.where(visible,rendered_dist,0.)
        r=np.r_[depth_res/self.cfg.depth_sigma_m/np.sqrt(len(depth_res)),
                observed_dist/self.cfg.contour_sigma_px/np.sqrt(len(observed_dist)),
                rendered_dist/self.cfg.contour_sigma_px/np.sqrt(len(rendered_dist))]
        if details:
            return r,{'depth_error_m':float(np.median(np.abs(depth_res))),
                       'depth_p90_m':float(np.percentile(np.abs(depth_res),90)),
                       'ray_hit_fraction':float(hit.mean()),
                       'contour_err_px':float((observed_dist.mean()+rendered_dist[visible].mean())/2) if visible.any() else 1e6,
                       'visible_model_fraction':float(visible.mean())}
        return r


def _planes(points, tolerance):
    """Small deterministic RANSAC + SVD, no world-up or support-plane prior."""
    remain=points.copy();result=[];rng=np.random.default_rng(7)
    for _ in range(3):
        if len(remain)<18: break
        triples=rng.integers(0,len(remain),size=(48,3))
        a,b,c=remain[triples[:,0]],remain[triples[:,1]],remain[triples[:,2]]
        normals=np.cross(b-a,c-a);length=np.linalg.norm(normals,axis=1)
        good=length>1e-7
        if not good.any(): break
        normals=normals[good]/length[good,None];a=a[good]
        distances=np.abs(remain@normals.T-np.sum(a*normals,axis=1))
        idx=int(np.argmax((distances<tolerance).sum(axis=0)))
        inliers=distances[:,idx]<tolerance
        if inliers.sum()<max(18,int(len(points)*.12)): break
        p=remain[inliers];center=p.mean(axis=0)
        _,singular,vh=np.linalg.svd(p-center,full_matrices=False)
        if singular[1]<.004: break
        normal=vh[-1]
        if normal@center>0:normal=-normal
        result.append((normal,center,p))
        remain=remain[~inliers]
    return result


def _orientations(basis, dims=None):
    seen=set()
    for perm in permutations(range(3)):
        if dims is not None:
            key=tuple(dims[perm.index(axis)] for axis in range(3))
            if key in seen:continue
            seen.add(key)
        R=basis[:,perm].copy()
        if np.linalg.det(R)<0:R[:,0]*=-1
        yield R


def _depth_candidates(obs, dims):
    if not hasattr(obs, '_depth_planes'):
        obs._depth_planes = _planes(obs.points,obs.cfg.plane_tolerance_m)
    planes=obs._depth_planes
    if not planes: return []
    n,center,pts=planes[0]
    tangent=np.eye(3)[np.argmin(np.abs(n))];tangent-=n*(n@tangent);tangent/=np.linalg.norm(tangent)
    bitangent=np.cross(n,tangent)
    xy=np.column_stack((pts@tangent,pts@bitangent)).astype(np.float32)
    corners=cv2.boxPoints(cv2.minAreaRect(xy)).astype(float)
    v2=corners[1]-corners[0];v2/=max(np.linalg.norm(v2),1e-9)
    u=tangent*v2[0]+bitangent*v2[1]
    bases=[np.column_stack((u,np.cross(n,u),n))]
    for n2,_,_ in planes[1:]:
        if abs(n@n2)<.3:
            n2=n2-n*(n@n2);n2/=np.linalg.norm(n2)
            bases.append(np.column_stack((n,n2,np.cross(n,n2))))
            break
    candidates=[];half=np.asarray(dims)/2
    if not hasattr(obs,'_depth_geometry_cache'):obs._depth_geometry_cache={}
    for basis in bases:
        for R in _orientations(basis,dims):
            key=R.tobytes()
            if key not in obs._depth_geometry_cache:
                lo,hi=np.percentile(obs.points@R,[1,99],axis=0)
                faces=[]
                for normal,c,p in planes:
                    axis=int(np.argmax(np.abs(normal@R)))
                    if abs(normal@R[:,axis])>.96:
                        sign=1 if normal@R[:,axis]>0 else -1
                        faces.append((axis,sign,np.median(p@R[:,axis])))
                obs._depth_geometry_cache[key]=(lo,hi,faces)
            lo,hi,faces=obs._depth_geometry_cache[key]
            midpoint=(lo+hi)/2
            constrained=[]
            for axis,sign,coordinate in faces:
                midpoint[axis]=coordinate-sign*half[axis]
                constrained.append(axis)
            candidates.append({'R':R,'t':R@midpoint,'source':'depth'})
            # For partial observations, retain alternative hidden extents instead of
            # silently assuming the observed fragment is centered in the full box.
            if obs.partial:
                for axis in range(3):
                    if axis in constrained:continue
                    for x in (lo[axis]+half[axis],hi[axis]-half[axis]):
                        c=midpoint.copy();c[axis]=x
                        candidates.append({'R':R,'t':R@c,'source':'depth_partial'})
    return candidates


def _ippe_candidates(obs, dims):
    poly=obs.polygon
    if len(poly)!=4 or obs.partial: return []
    result=[]
    seen_faces=set()
    for normal_axis in range(3):
        axes=[i for i in range(3) if i!=normal_axis]
        face_key=(dims[normal_axis], tuple(sorted(np.array(dims)[axes])))
        if face_key in seen_faces:continue
        seen_faces.add(face_key)
        a,b=np.array(dims)[axes]/2
        obj=np.array([[-a,-b,0.],[a,-b,0.],[a,b,0.],[-a,b,0.]])
        for shift in range(4):
            image=np.roll(poly,shift,axis=0).astype(float)
            try:sol=cv2.solvePnPGeneric(obj,image,obs.K,np.zeros(4),flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:continue
            if not sol[0]:continue
            for rv,tv in zip(sol[1],sol[2]):
                face=cv2.Rodrigues(rv)[0];tc=tv.ravel()
                normal=face[:,2].copy()
                if normal@tc>0:normal=-normal
                R=np.empty((3,3));R[:,axes[0]]=face[:,0];R[:,axes[1]]=face[:,1];R[:,normal_axis]=normal
                if np.linalg.det(R)<0:R[:,axes[1]]*=-1
                result.append({'R':R,'t':tc-normal*dims[normal_axis]/2,'source':'ippe'})
    return result


def _cost(r):
    # Scales already incorporate the bounded sample counts. Huber clips the
    # influence of isolated residuals without concealing the raw acceptance metrics.
    a=np.abs(r);delta=.35
    return float(np.sum(np.where(a<=delta,.5*a*a,delta*(a-.5*delta))))


def _step(R,t,delta):
    return cv2.Rodrigues(delta[:3]*.15)[0]@R,t+delta[3:]*.005


def _refine(obs,candidate,dims):
    R,t=candidate['R'].copy(),candidate['t'].copy()
    r=obs.residual(R,t,dims)
    if r is None:return None
    start_cost=cost=_cost(r);accepted_steps=0;J=None
    for _ in range(obs.cfg.iterations):
        eps=.01
        columns=[]
        for j in range(6):
            d=np.zeros(6);d[j]=eps;rp,tp=_step(R,t,d)
            perturbed=obs.residual(rp,tp,dims)
            if perturbed is None:return None
            columns.append((perturbed-r)/eps)
        J=np.column_stack(columns)
        weights=np.minimum(1.,.35/np.maximum(np.abs(r),1e-9))
        A=J.T@(weights[:,None]*J);g=J.T@(weights*r)
        improved=False
        for damping in (.003,.03,.3):
            try:delta=-np.linalg.solve(A+damping*np.eye(6),g)
            except np.linalg.LinAlgError:continue
            # Bound each iteration: <= ~6deg and <= 5mm per vector update.
            delta[:3]*=min(1.,.7/max(np.linalg.norm(delta[:3]),1e-9))
            delta[3:]*=min(1.,1./max(np.linalg.norm(delta[3:]),1e-9))
            nr,nt=_step(R,t,delta);new_r=obs.residual(nr,nt,dims)
            new_cost=_cost(new_r) if new_r is not None else np.inf
            if new_cost<cost-1e-7:
                R,t,r,cost=nr,nt,new_r,new_cost;improved=True;accepted_steps+=1;break
        if not improved:break
    residual,quality=obs.residual(R,t,dims,details=True)
    out=dict(candidate,R=R,t=t,cost=_cost(residual),initial_cost=start_cost,refine_steps=accepted_steps,**quality)
    if J is not None:
        singular=np.linalg.svd(J,compute_uv=False)
        out['jacobian_condition']=float(singular[0]/max(singular[-1],1e-12))
    return out


def fit_cuboid_rgbd(mask, depth_m, dims, camera_matrix, *, config=None):
    """Compatible single-model entry point."""
    return fit_cuboid_models(mask, depth_m, [{'id':'single', 'dims':dims}],
                             camera_matrix, config=config)


def fit_cuboid_models(mask, depth_m, models, camera_matrix, *, config=None):
    """Joint discrete model / continuous pose fit with one shared observation.

    The refinement budget is TOTAL, not per model. Reserve one candidate per
    model before filling by cost, so model uncertainty cannot disappear merely
    because the leading model monopolizes refinement. No cached pose propagation.
    """
    cfg=config or FitConfig();timings={};started=time.perf_counter()
    def finish(result):
        result['timing_ms']=dict(timings,total=(time.perf_counter()-started)*1000.)
        return result
    K=np.asarray(camera_matrix,dtype=float)
    mask=np.asarray(mask,dtype=bool);depth=np.asarray(depth_m)
    if mask.ndim!=2 or depth.shape!=mask.shape:
        return finish({'accepted':False,'reason':'invalid_input'})
    try:
        models=[dict(id=m['id'], dims=tuple(float(x) for x in m['dims'])) for m in models]
        valid=(0<len(models)<=min(8,cfg.refine_candidates)
               and len({m['id'] for m in models})==len(models)
               and all(isinstance(m['id'],str) and m['id'] and len(m['dims'])==3
                       and all(np.isfinite(x) and x>0 for x in m['dims']) for m in models))
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:return finish({'accepted':False,'reason':'invalid_models'})
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        return finish({'accepted':False,'reason':'invalid_intrinsics'})
    try:obs=Observation(mask,depth,K,cfg)
    except ValueError as exc:return finish({'accepted':False,'reason':str(exc)})
    timings['prepare']=(time.perf_counter()-started)*1000.;mark=time.perf_counter()
    candidates=[]
    for model in models:
        dims=model['dims']
        seeds=_depth_candidates(obs,dims)+_ippe_candidates(obs,dims)
        if len(obs.polygon)==6 and not obs.partial:
            seeds += [dict(c,source='pnp6') for c in solve_pnp_hexagon(
                obs.polygon,dims,K,symmetry_reduced=True,model_symmetries=cuboid_symmetries(tuple(dims)))]
        candidates.extend(dict(c,model_id=model['id'],dims=dims) for c in seeds)
    timings['initialize']=(time.perf_counter()-mark)*1000.;mark=time.perf_counter()
    ranked=[]
    for c in candidates:
        r=obs.residual(c['R'],c['t'],c['dims'])
        if r is not None:ranked.append(dict(c,cost=_cost(r)))
    ranked.sort(key=lambda c:c['cost']);chosen=[]
    # First compare a representative of EVERY proposed metric model.
    if len(models)>1:
        for model in models:
            representative=next((c for c in ranked if c['model_id']==model['id']),None)
            if representative is not None:chosen.append(representative)
    for c in ranked:
        if len(chosen)>=cfg.refine_candidates:break
        if any(c['model_id']==p['model_id'] and np.linalg.norm(c['t']-p['t'])<.004
               and symmetry_angle(c['R'],p['R'],c['dims'])<8 for p in chosen):continue
        chosen.append(c)
    timings['rank']=(time.perf_counter()-mark)*1000.;mark=time.perf_counter()
    refined=[r for c in chosen if (r:=_refine(obs,c,c['dims'])) is not None]
    refined.sort(key=lambda c:c['cost'])
    timings['refine']=(time.perf_counter()-mark)*1000.
    common={'vertex_count':len(obs.polygon),'clean_depth_points':obs.support_count,
            'sampled_points':len(obs.points),'raw_valid_points':obs.raw_valid_count,
            'partial_contour_fraction':obs.partial_fraction,'n_candidates':len(candidates),
            'n_refined':len(refined),'n_models':len(models)}
    if not refined:return finish(dict(common,accepted=False,reason='no_initialization'))
    best=refined[0];reason='ok'
    if best['ray_hit_fraction']<.85 or best['visible_model_fraction']<.25:
        reason='insufficient_model_coverage'
    elif best['depth_error_m']>cfg.max_depth_error_m or best['depth_p90_m']>cfg.max_depth_error_m*2:
        reason='depth_mismatch'
    elif best['contour_err_px']>cfg.max_contour_error_px:
        reason='contour_mismatch'
    elif best.get('jacobian_condition',np.inf)>1e5:
        reason='underconstrained'
    margin=np.inf;model_margin=np.inf
    for alternative in refined[1:]:
        other_model=best['model_id']!=alternative['model_id']
        if other_model:model_margin=min(model_margin,alternative['cost']-best['cost'])
        different=(other_model or symmetry_angle(best['R'],alternative['R'],best['dims'])>cfg.ambiguity_rotation_deg
                   or np.linalg.norm(best['t']-alternative['t'])>cfg.ambiguity_translation_m)
        if different:
            margin=min(margin,alternative['cost']-best['cost'])
    if reason=='ok' and margin<cfg.ambiguity_cost:
        reason='ambiguous_model' if model_margin<cfg.ambiguity_cost else 'ambiguous'
    model_scores={m['id']:min((c['cost'] for c in refined if c['model_id']==m['id']),default=None) for m in models}
    out=dict(common,**best,accepted=reason=='ok',reason=reason,score_margin=float(margin) if np.isfinite(margin) else None,
             model_margin=float(model_margin) if np.isfinite(model_margin) else None,model_scores=model_scores,
             sdf_err_mm=best['depth_error_m']*1000.,
             combined=best['contour_err_px']+best['depth_error_m']*1000.,score_total=best['cost'])
    return finish(out)
