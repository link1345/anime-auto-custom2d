"""Conservative parent deformation settings and hair-excluded depth approximation."""
import copy
import math
import numpy as np
from PIL import Image

DEFAULTS=dict(enabled=True,face=.15,eyes=.15,mouth=.25,hair=0,body=.25,lag=.2,light=.12,shadow=.1,camera=0,cursorMode=0,cameraDistance=.5,targetDepth=.5,cursorDepthFollow=.5,relief=1,faceTilt=0,yawLimit=20,pitchLimit=15,hairRelief=.5,gazeEnabled=1,gazeRangeX=.2,gazeRangeY=.15,neckRelief=.5,bodyRelief=.5)

def validate_settings(s):
    if type(s.get('enabled')) is not bool:raise ValueError('Invalid parent motion switch')
    for key in DEFAULTS:
        if key=='enabled':continue
        x=s.get(key,DEFAULTS[key])
        if key=='camera' and (type(x) not in (int,float) or x not in (0,1,2)):raise ValueError('Invalid camera mode')
        if key=='cursorMode' and (type(x) not in (int,float) or x not in (0,1)):raise ValueError('Invalid cursor mode')
        if type(x) not in (int,float) or not math.isfinite(x) or not ((x in (0,1,2)) if key=='camera' else (5<=x<=45) if key=='yawLimit' else (5<=x<=30) if key=='pitchLimit' else (0<=x<=3) if key in ('relief','hairRelief','neckRelief','bodyRelief') else (0<=x<=1)):raise ValueError('Invalid parent motion setting')
    surface=s.get('surface')
    for grid in [*s.get('hairSurfaces',{}).values(),*s.get('bodySurfaces',{}).values()]:
        if grid['cols']!=49 or grid['rows']!=73 or len(grid['values'])!=49*73:raise ValueError('Invalid hair surface')
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in grid['values']):raise ValueError('Invalid hair depth')
    if surface:
        g=surface.get('grid')
        if g:
            if g.get('cols')!=128 or g.get('rows')!=128:raise ValueError('Invalid face grid')
            for key in ('residual','confidence'):
                a=np.asarray(g.get(key,[]),dtype=float)
                if a.shape!=(128*128,) or not np.isfinite(a).all() or np.max(np.abs(a))>4:raise ValueError('Invalid face grid values')
            plane=np.asarray(g.get('plane',[]),dtype=float)
            if plane.shape!=(3,) or not np.isfinite(plane).all() or np.max(np.abs(plane))>20:raise ValueError('Invalid face plane')
        c=surface['coefficients']
        if len(c)!=6 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>20 for x in c):raise ValueError('Invalid face relief')

def configure(store,project,data):
    p=copy.deepcopy(project);s={**DEFAULTS,**p.get('parentMotion',{})}
    if 'cursorMode' not in p.get('parentMotion',{}):s['cursorMode']=1 if p.get('parentMotion',{}).get('camera')==1 else 0
    s.update({k:data[k] for k in DEFAULTS if k in data});validate_settings(s)
    if data.get('rebuild') or 'surface' not in s:
        s['surface']=build_surface(store,p)
    if data.get('rebuildHair'):
        s['hairSurfaces']=build_hair_surfaces(store,p)
    if data.get('rebuildBody'):
        s['bodySurfaces']=build_hair_surfaces(store,p,body=True)
        p=build_body_meshes(p,s['bodySurfaces'])
    p['parentMotion']=s
    return p

def build_hair_surfaces(store,p,body=False):
    record=p.get('depth',{});arrays=record.get('arrays',{})
    if not all(k in arrays for k in ('points.npy','mask.npy')):raise ValueError('先に原画のMoGe3深度を推定・取り込みしてください')
    folder=store.folder(p['id']);points=np.load(folder/arrays['points.npy'],allow_pickle=False);valid=np.load(folder/arrays['mask.npy'],allow_pickle=False).astype(bool)
    h,w=valid.shape
    if points.shape!=(h,w,3):raise ValueError('Invalid hair depth dimensions')
    cols,rows=49,73;yy,xx=np.mgrid[0:rows,0:cols];gx=xx/(cols-1);gy=yy/(rows-1)
    masks={}
    for part in p['parts']:
        if not part.get('visible',True):continue
        with Image.open(folder/part['image']) as im:a=np.asarray(im.convert('RGBA'))[:,:,3]
        r=part.get('rect',[0,0,1,1]);u=(gx-r[0])/r[2];v=(gy-r[1])/r[3]
        masks[part['id']]=((u>=0)&(u<=1)&(v>=0)&(v<=1)&(a[np.clip((v*(a.shape[0]-1)).astype(int),0,a.shape[0]-1),np.clip((u*(a.shape[1]-1)).astype(int),0,a.shape[1]-1)]>180))
    z=points[(gy*(h-1)).astype(int),(gx*(w-1)).astype(int),2]
    good=valid[(gy*(h-1)).astype(int),(gx*(w-1)).astype(int)]&np.isfinite(z)&(z>0)
    hairparts=[part for part in p['parts'] if (part['role'] in ('neck','body','chest','skirt','legs','feet','armLeft','armRight') if body else part['role'].startswith('hair')) and not part.get('locked') and part['id'] in masks]
    observations={}
    for part in hairparts:
        mask=masks[part['id']].copy()
        for front in p['parts']:
            if front['id'] in masks and front['depth']>part['depth']:mask&=~masks[front['id']]
        observations[part['id']]=mask&good
    union=np.zeros_like(good)
    for mask in observations.values():union|=mask
    if union.sum()<8:raise ValueError('対象マスク内の有効な深度点が不足しています。パーツの表示・配置・保護を確認してください')
    lo,hi=np.percentile(z[union],[5,95]);span=max(float(hi-lo),float(np.median(z[union]))*.01)
    result={}
    for part in hairparts:
        mask=observations[part['id']];weight=mask.astype(float);weighted=np.where(mask,np.clip((hi-z)/span,0,1),0)
        # Mask-normalized filtering never introduces skin/background depth.
        for _ in range(4):
            def blur(a):
                b=np.pad(a,1,mode='edge');return (4*b[1:-1,1:-1]+b[:-2,1:-1]+b[2:,1:-1]+b[1:-1,:-2]+b[1:-1,2:])/8
            weighted=blur(weighted);weight=blur(weight)
        values=np.divide(weighted,weight,out=np.full_like(weighted,.35),where=weight>1e-5)
        # Unobserved regions smoothly return to the neutral layer, not claimed reconstruction.
        confidence=np.clip(weight*4,0,1);values=.35+(values-.35)*confidence
        result[part['id']]={'cols':cols,'rows':rows,'values':values.ravel().tolist(),'samples':int(mask.sum()),'image':part['image'],'rect':part.get('rect',[0,0,1,1]),'source':'masked-moge3','sourceArray':arrays['points.npy']}
    return result

def build_surface(store,p):
    fallback={'coefficients':[.7,0,0,-1.2,0,-1.2],'source':'smooth-template','samples':0}
    record=p.get('faceDepths',{}).get('front');face=p.get('faceInput')
    if not record or not face or record['sourceAsset']!=face['image']:return fallback
    folder=store.folder(p['id']);points=np.load(folder/record['arrays']['points.npy'],allow_pickle=False);valid=np.load(folder/record['arrays']['mask.npy'],allow_pickle=False).astype(bool)
    h,w=valid.shape;r=face['alignment']['rect'];n=128
    yy,xx=np.mgrid[0:n,0:n];u=xx/(n-1);v=yy/(n-1);gx=r[0]+u*r[2];gy=r[1]+v*r[3]
    skin=np.zeros((n,n),bool);hair=np.zeros((n,n),bool)
    skin_depth=max((part['depth'] for part in p['parts'] if part.get('visible',True) and part.get('tag') in ('skin','face')),default=0)
    for part in p['parts']:
        is_hair=part['role'].startswith('hair')
        if is_hair and (part['depth']<skin_depth or part.get('tag')=='back hair'):continue
        if not part.get('visible',True) or not (is_hair or part.get('tag') in ('skin','face')):continue
        with Image.open(folder/part['image']) as im:a=np.asarray(im.convert('RGBA'))[:,:,3]
        pr=part.get('rect',[0,0,1,1]);px=(gx-pr[0])/pr[2];py=(gy-pr[1])/pr[3]
        inside=(px>=0)&(px<=1)&(py>=0)&(py<=1)
        alpha=a[np.clip((py*(a.shape[0]-1)).astype(int),0,a.shape[0]-1),np.clip((px*(a.shape[1]-1)).astype(int),0,a.shape[1]-1)]
        if is_hair:hair|=inside&(alpha>16)
        else:skin|=inside&(alpha>220)
    iz=(v*(h-1)).astype(int);ix=(u*(w-1)).astype(int);z=points[iz,ix,2]
    good=skin&~hair&valid[iz,ix]&np.isfinite(z)&(z>0)&(v>.3)&(v<.85)
    if good.sum()<60:return fallback
    lo,hi=np.percentile(z[good],[5,95])
    if hi-lo<1e-6:return fallback
    x=u-.5;y=v-.55;A=np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],axis=-1)
    target=np.clip((hi-z)/(hi-lo),0,1);a=A[good];b=target[good]
    prior=np.array(fallback['coefficients']);c=np.linalg.solve(a.T@a+np.eye(6)*10,a.T@b+prior*10)
    # Fit only the global plane, retaining local depth as a separate masked residual.
    plane_matrix=np.stack([np.ones_like(x),x,y],axis=-1)
    raw=(hi-z)/(hi-lo)
    plane=np.linalg.lstsq(plane_matrix[good],raw[good],rcond=None)[0]
    residual=np.where(good,np.clip(raw-plane_matrix@plane,-2,2),0)
    weight=good.astype(float)
    def blur(a):
        b=np.pad(a,1,mode='edge')
        return (4*b[1:-1,1:-1]+b[:-2,1:-1]+b[2:,1:-1]+b[1:-1,:-2]+b[1:-1,2:])/8
    for _ in range(2):residual=blur(residual);weight=blur(weight)
    residual=np.divide(residual,weight,out=np.zeros_like(residual),where=weight>1e-6)
    confidence=np.clip(weight,0,1)
    return {'coefficients':np.clip(c,-3,3).tolist(),'source':'skin-only-moge3-depth-grid','samples':int(good.sum()),
            'grid':{'cols':n,'rows':n,'plane':plane.tolist(),'residual':residual.ravel().tolist(),'confidence':confidence.ravel().tolist(),
                    'sourceAsset':face['image'],'sourceArray':record['arrays']['points.npy'],'rect':r},
            'note':'Masked local depth minus fitted global plane; unobserved residual fades to zero, not reconstructed geometry.'}


def build_body_meshes(p,surfaces):
    from .character import build_rig
    changed=set()
    for part in p['parts']:
        if part['id'] not in surfaces:continue
        r=part.get('rect',[0,0,1,1]);cols,rows=25,33
        vertices=[[r[0]+r[2]*i/(cols-1),r[1]+r[3]*j/(rows-1),part['depth']] for j in range(rows) for i in range(cols)]
        triangles=[]
        for j in range(rows-1):
            for i in range(cols-1):
                n=j*cols+i;triangles.extend([[n,n+1,n+cols],[n+1,n+cols+1,n+cols]])
        part['mesh']={'vertices':vertices,'triangles':triangles,'version':part['mesh'].get('version',0)+1,'source':'body-relief'}
        changed.add(part['id'])
    for expression in p.get('expressions',[]):
        if expression.get('fit',{}).get('partId') in changed:expression['stale']=True
    if p.get('rig') and not p['rig'].get('stale'):p=build_rig(p)
    return p
