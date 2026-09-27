"""Character construction: editable landmarks, weighted rig, correspondence morphs."""
import copy
import math
import numpy as np
from PIL import Image
from .project import uid

JOINTS=('root','neck','head','shoulderLeft','elbowLeft','wristLeft','handLeft',
        'shoulderRight','elbowRight','wristRight','handRight','hipLeft','kneeLeft','ankleLeft','hipRight','kneeRight','ankleRight')
PARENTS=(-1,0,1,1,3,4,5,1,7,8,9,0,11,12,0,14,15)
POSE_MAP={'shoulderLeft':12,'elbowLeft':14,'wristLeft':16,'handLeft':20,'hipLeft':24,'kneeLeft':26,'ankleLeft':28,
          'shoulderRight':11,'elbowRight':13,'wristRight':15,'handRight':19,'hipRight':23,'kneeRight':25,'ankleRight':27}


def check_points(points,count=None):
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or not 3<=len(a)<=512 or not np.isfinite(a).all() or np.min(a)<0 or np.max(a)>1 or (count and len(a)!=count):
        raise ValueError('対応点は3〜512個の0〜1座標です。同じ順序・個数で指定してください')
    return a


def landmarks_from_detection(project,metadata):
    updated=copy.deepcopy(project);old=updated.get('landmarks',{});joints=old.get('joints',{})
    pose=metadata.get('pose',[])
    for name,index in POSE_MAP.items():
        if index<len(pose) and len(pose[index])>=5:
            x,y,z,visibility,presence=pose[index][:5]
            if np.isfinite([x,y,visibility,presence]).all() and 0<=x<=1 and 0<=y<=1 and min(visibility,presence)>=.5 and not joints.get(name,{}).get('locked'):
                joints[name]={'position':[x,y],'confidence':min(visibility,presence),'source':'mediapipe','locked':False}
    for name,a,b in [('neck','shoulderLeft','shoulderRight'),('root','hipLeft','hipRight')]:
        if a in joints and b in joints and not joints.get(name,{}).get('locked'):
            joints[name]={'position':((np.array(joints[a]['position'])+joints[b]['position'])/2).tolist(),'confidence':min(joints[a]['confidence'],joints[b]['confidence']),'source':'derived','locked':False}
    face=metadata.get('face',[])
    if len(face)>=468:
        centers=sorted([np.mean([face[i][:2] for i in group],axis=0).tolist() for group in ((33,133),(263,362))])
        for name,xy in [('leftEye',centers[0]),('rightEye',centers[1]),('mouth',np.mean([face[13][:2],face[14][:2]],axis=0).tolist())]:
            if all(0<=x<=1 for x in xy):updated['face'][name]=xy
        xy=face[1][:2]
        if all(0<=x<=1 for x in xy) and not joints.get('head',{}).get('locked'):
            joints['head']={'position':xy,'confidence':None,'source':'mediapipe','locked':False}
    updated['landmarks']={'joints':joints,'face':face,'hands':metadata.get('hands',[]),'warnings':metadata.get('warnings',[])}
    if updated.get('rig'):updated['rig']['stale']=True
    return updated


def manual_joint(project,name,position):
    if name not in JOINTS or len(position)!=2 or not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in position):raise ValueError('Invalid joint')
    p=copy.deepcopy(project);p.setdefault('landmarks',{}).setdefault('joints',{})[name]={'position':position,'confidence':1,'source':'manual','locked':True}
    if p.get('rig'):p['rig']['stale']=True
    return p


def segment_distance(points,a,b):
    delta=b-a;t=np.clip(((points-a)*delta).sum(axis=1)/max(float(delta@delta),1e-9),0,1)
    return np.linalg.norm(points-(a+t[:,None]*delta),axis=1)


def build_rig(project):
    p=copy.deepcopy(project);joints=p.get('landmarks',{}).get('joints',{})
    required=JOINTS[:11]
    missing=[k for k in required if k not in joints]
    if missing:raise ValueError('上半身リグに必要な関節を推定または手動登録してください: '+', '.join(missing))
    bones=[];lookup={}
    for i,name in enumerate(JOINTS):
        if name not in joints:continue
        parent_name=JOINTS[PARENTS[i]] if PARENTS[i]>=0 else None
        if parent_name and parent_name not in lookup:continue
        lookup[name]=len(bones);position=joints['neck']['position'] if name=='head' else joints[name]['position']
        bones.append({'name':name,'parent':lookup.get(parent_name,-1),'position':position+[0]})
    for part in p['parts']:
        vertices=np.asarray(part['mesh']['vertices'])[:,:2]
        role=part['role'];choices=['root']
        if role=='head' or role.startswith('hair') or part.get('scope')=='face':choices=['head']
        elif role=='neck':choices=['neck','head']
        elif role.startswith(('arm','forearm','hand')):
            side='Left' if 'Left' in role else 'Right' if 'Right' in role else None
            choices=[k+s for s in ([side] if side else ['Left','Right']) for k in ['shoulder','elbow','wrist']]
        elif role in ('legs','feet'):choices=[k+s for s in ['Left','Right'] for k in ['hip','knee','ankle'] if k+s in lookup]
        choices=choices or ['root']
        distances=[]
        for name in choices:
            index=lookup[name];a=np.asarray(bones[index]['position'][:2]);children=[b for b in bones if b['parent']==index]
            b=np.asarray(children[0]['position'][:2]) if children else a
            distances.append(segment_distance(vertices,a,b))
        distances=np.stack(distances,axis=1)
        order=np.argsort(distances,axis=1)[:,:min(4,len(choices))]
        weights=1/np.maximum(np.take_along_axis(distances,order,axis=1),.005)**4
        weights/=weights.sum(axis=1,keepdims=True)
        indices=np.array([lookup[k] for k in choices])[order]
        pad=4-indices.shape[1];indices=np.pad(indices,((0,0),(0,pad)));weights=np.pad(weights,((0,0),(0,pad)))
        part['mesh']['joints']=indices.tolist();part['mesh']['weights']=weights.tolist()
    head=joints['head']['position'];neck=joints['neck']['position'];s1=joints['shoulderLeft']['position'];s2=joints['shoulderRight']['position']
    width=abs(s2[0]-s1[0])
    p['rig']={'version':1,'bones':bones,'stale':False,'colliders':[
        {'name':'head','center':head,'radius':max(.025,width*.23),'bone':'head'},
        {'name':'shoulderLeft','center':s1,'radius':max(.015,width*.13),'bone':'neck'},
        {'name':'shoulderRight','center':s2,'radius':max(.015,width*.13),'bone':'neck'}]}
    return p


def split_arm(store,project,part_id,side):
    if side not in ('Left','Right'):raise ValueError('Invalid side')
    p=copy.deepcopy(project);part=next(x for x in p['parts'] if x['id']==part_id)
    if part['locked']:raise ValueError('保護を解除してから腕を分割してください')
    joints=p.get('landmarks',{}).get('joints',{})
    names=[k+side for k in ('shoulder','elbow','wrist','hand')]
    if any(k not in joints for k in names):raise ValueError('肩・肘・手首・手先を先に登録してください')
    im=Image.open(store.folder(p['id'])/part['image']).convert('RGBA');array=np.asarray(im).copy();h,w=array.shape[:2]
    rect=part.get('rect',[0,0,1,1]);yy,xx=np.mgrid[:h,:w];points=np.stack([rect[0]+xx.ravel()/w*rect[2],rect[1]+yy.ravel()/h*rect[3]],axis=1)
    js=[np.array(joints[k]['position']) for k in names]
    labels=np.argmin(np.stack([segment_distance(points,js[i],js[i+1]) for i in range(3)]),axis=0).reshape(h,w)
    children=[]
    from .pipeline import mesh_for
    for i,(role,name) in enumerate([('arm','上腕'),('forearm','前腕'),('hand','手')]):
        pixels=array.copy();pixels[:,:,3]=np.where(labels==i,array[:,:,3],0);out=Image.fromarray(pixels);bbox=out.getchannel('A').getbbox()
        if not bbox:raise ValueError('空の部位ができます。関節位置と対象パーツを確認してください')
        child=copy.deepcopy(part);child.update(id=uid(),name=name+('・画面左' if side=='Left' else '・画面右'),role=role+side,locked=True,visible=True,provenance='joint-split',derivedFrom=part_id)
        child['image']='assets/'+child['id']+'.png';out.save(store.folder(p['id'])/child['image']);child['pivot']=js[i].tolist()
        l,t,r,b=bbox;x,y,rw,rh=rect;child['polygon']=[[x+l/w*rw,y+t/h*rh],[x+r/w*rw,y+t/h*rh],[x+r/w*rw,y+b/h*rh],[x+l/w*rw,y+b/h*rh]]
        # Share the original tessellation across adjoining pieces so their common
        # cut edge follows exactly the same skin transform (and retains MoGe Z).
        child['mesh']=copy.deepcopy(part['mesh']);children.append(child)
    part['visible']=False;part['locked']=True;p['parts']+=children
    return build_rig(p)


def contour(image,count=48):
    """Ordered alpha silhouette samples, not inferred facial anatomical landmarks."""
    a=np.asarray(image.convert('RGBA'))[:,:,3];ys,xs=np.where(a>127)
    if len(xs)<10:raise ValueError('輪郭を取得できません。透過した顔/部位画像を使ってください')
    if np.all(a>127):raise ValueError('全面不透明です。対応点を手動指定するか、透過レイヤーを使用してください')
    cx,cy=xs.mean(),ys.mean();angle=(np.arctan2(ys-cy,xs-cx)+2*np.pi)%(2*np.pi);radius=(xs-cx)**2+(ys-cy)**2
    result=[]
    for theta in np.arange(count)*2*np.pi/count:
        delta=np.abs((angle-theta+np.pi)%(2*np.pi)-np.pi);indices=np.where(delta<np.pi/count)[0]
        if not len(indices):indices=np.array([np.argmin(delta)])
        index=indices[np.argmax(radius[indices])];result.append([float(xs[index]/image.width),float(ys[index]/image.height)])
    return result


def displacement(vertices,source,target):
    vertices=np.asarray(vertices);source=check_points(source);target=check_points(target,len(source))
    d=((vertices[:,None,:2]-source[None,:,:])**2).sum(axis=2)
    weights=1/np.maximum(d,1e-12)**2;weights/=weights.sum(axis=1,keepdims=True)
    return weights@(target-source)


def fit_expression(store,project,expression_id,part_id,base_points=None,target_points=None):
    p=copy.deepcopy(project);part=next(x for x in p['parts'] if x['id']==part_id);expr=next(e for e in p['expressions'] if e['id']==expression_id)
    folder=store.folder(p['id'])
    with Image.open(folder/part['image']) as im:base=im.convert('RGBA')
    with Image.open(folder/expr['image']) as im:target=im.convert('RGBA')
    method='manual' if base_points is not None else 'alpha-contour-idw'
    base_points=base_points if base_points is not None else contour(base)
    target_points=target_points if target_points is not None else contour(target)
    source=check_points(base_points);dest=check_points(target_points,len(source));r=part.get('rect',[0,0,1,1]);er=expr['rect']
    source=source*np.array(r[2:])+r[:2];dest=dest*np.array(er[2:])+er[:2]
    vertices=np.asarray(part['mesh']['vertices']);delta=displacement(vertices,source,dest)
    target_vertices=vertices.copy();target_vertices[:,:2]+=delta
    # Reject folded triangles instead of silently creating inverted faces.
    for tri in part['mesh']['triangles']:
        a,b,c=vertices[tri,:2];x,y,z=target_vertices[tri,:2]
        cross=lambda u,v:u[0]*v[1]-u[1]*v[0]
        if cross(b-a,c-a)*cross(y-x,z-x)<=0:raise ValueError('対応点でメッシュが反転します。位置と対応順を修正してください')
    morph=np.column_stack([delta,np.zeros(len(delta))]).tolist();part['mesh'].setdefault('morphs',{})[expression_id]=morph
    # Texture samples in the expression image correspond to target geometry.
    uv=((target_vertices[:,:2]-er[:2])/er[2:]).tolist()
    expr.update(fit={'partId':part_id,'sourcePoints':source.tolist(),'targetPoints':dest.tolist(),'uv':uv,'method':method},meshVersion=p['meshVersion'],stale=False)
    return p


def build_eye_masks(store,project,scale=None,synchronize=False):
    from .pipeline import mesh_for
    p=copy.deepcopy(project);folder=store.folder(p['id'])
    tags=('eyewhite','irides','eyelash')
    layers=[x for x in p['parts'] if x.get('scope')=='face' and x.get('visible',True) and x.get('tag') in tags]
    if not all(any(x['tag']==tag for x in layers) for tag in tags):raise ValueError('表示中の白目・瞳・まつげレイヤーが必要です。左右分割にも対応しています')
    scale=p.get('eyeMask',{}).get('scale',1) if scale is None else scale
    if type(scale) not in (int,float) or not math.isfinite(scale) or not .5<=scale<=1.5:raise ValueError('目の共通サイズは0.5〜1.5です')
    reference=next(x for x in layers if x['tag']=='eyewhite');rect=reference['rect']
    if not synchronize and any(x['rect']!=rect for x in layers):raise ValueError('先に「目の配置・変形を統一」を実行してください')
    with Image.open(folder/reference['image']) as reference_image:size=reference_image.size
    composites={tag:Image.new('RGBA',size) for tag in tags}
    for part in layers:
        im=Image.open(folder/part['image']).convert('RGBA').resize(size,Image.Resampling.LANCZOS)
        composites[part['tag']].alpha_composite(im)
    # Fill the pupil hole inside each row of the sclera silhouette. An oversized
    # iris must never enlarge the permitted aperture.
    alpha=np.array(composites['eyewhite'])[:,:,3].copy()
    for left,right in [(0,size[0]//2),(size[0]//2,size[0])]:
        region=alpha[:,left:right]
        # Ignore isolated inference specks outside the principal sclera island.
        thumb=Image.fromarray(region);thumb.thumbnail((256,512),Image.Resampling.BOX)
        occupied=np.asarray(thumb)>8;seen=np.zeros_like(occupied);best=[]
        for yy,xx in zip(*np.where(occupied)):
            if seen[yy,xx]:continue
            stack=[(int(yy),int(xx))];seen[yy,xx]=True;component=[]
            while stack:
                y,x=stack.pop();component.append((y,x))
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<occupied.shape[0] and 0<=nx<occupied.shape[1] and occupied[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
            if len(component)>len(best):best=component
        if best:
            keep=np.zeros_like(occupied,dtype=np.uint8)
            for y,x in best:keep[y,x]=255
            # Expand by one thumbnail pixel to retain antialiased native edges.
            from PIL import ImageFilter
            keep=Image.fromarray(keep).filter(ImageFilter.MaxFilter(3)).resize((right-left,size[1]),Image.Resampling.NEAREST)
            region[np.asarray(keep)==0]=0
        for row in region:
            xs=np.flatnonzero(row>8)
            if len(xs)>1:row[xs[0]+1:xs[-1]]=255
    rgba=np.zeros((*alpha.shape,4),dtype=np.uint8);rgba[:,:,:3]=255;rgba[:,:,3]=alpha
    path='assets/eye-mask-'+uid()+'.png';Image.fromarray(rgba).save(folder/path)
    iris_path='assets/eye-iris-'+uid()+'.png';composites['irides'].save(folder/iris_path)
    eyes=[]
    for left,right in [(0,size[0]//2),(size[0]//2,size[0])]:
        ys,xs=np.where(alpha[:,left:right]>8)
        if not len(xs):raise ValueError('片方の白目が空です。左右の白目を表示してください')
        eyes.append({'bbox':[float((left+xs.min())/size[0]),float(ys.min()/size[1]),float((left+xs.max()+1)/size[0]),float((ys.max()+1)/size[1])]})
    if synchronize:
        x,y,w,h=rect;polygon=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
        mesh=mesh_for(polygon,reference['depth'])
        for part in layers:
            part.update(rect=rect.copy(),pivot=[x+w/2,y+h/2],mesh=copy.deepcopy(mesh))
            part['physics']['enabled']=False
        for expression in p['expressions']:
            if expression.get('fit',{}).get('partId') in {x['id'] for x in layers}:expression['stale']=True
        if p.get('rig') and not p['rig'].get('stale'):p=build_rig(p)
    p['eyeMask']={'image':path,'irisImage':iris_path,'rect':rect,'eyes':eyes,'scale':scale,'source':'sclera-filled-aperture','layerIds':[x['id'] for x in layers]}
    return p


def extend_hidden(store,project,part_id,pixels=8):
    """Bounded colour propagation beneath existing occluders; no generative claim."""
    if type(pixels) is not int or not 1<=pixels<=32:raise ValueError('補完幅は1〜32ピクセルです')
    p=copy.deepcopy(project);part=next(x for x in p['parts'] if x['id']==part_id)
    if part['locked']:raise ValueError('保護を解除してください')
    folder=store.folder(p['id']);im=Image.open(folder/part['image']).convert('RGBA');a=np.array(im);known=a[:,:,3]>0
    coverage=Image.new('L',im.size);rect=part.get('rect',[0,0,1,1])
    for other in p['parts']:
        if other['id']==part_id or other['depth']<=part['depth'] or not other.get('visible',True):continue
        if p.get('display',{}).get('templatesVisible') is False and other.get('provenance')=='template':continue
        oi=Image.open(folder/other['image']).convert('RGBA');r=other.get('rect',[0,0,1,1])
        size=(max(1,round(r[2]/rect[2]*im.width)),max(1,round(r[3]/rect[3]*im.height)))
        oi=oi.resize(size);coverage.paste(oi.getchannel('A'),(round((r[0]-rect[0])/rect[2]*im.width),round((r[1]-rect[1])/rect[3]*im.height)),oi.getchannel('A'))
    allowed=np.asarray(coverage)>0;original=known.copy()
    for _ in range(pixels):
        source=a.copy();was=known.copy()
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            shifted=np.roll(source,(dy,dx),(0,1));valid=np.roll(was,(dy,dx),(0,1))
            if dy==-1:valid[-1,:]=False
            if dy==1:valid[0,:]=False
            if dx==-1:valid[:,-1]=False
            if dx==1:valid[:,0]=False
            fill=allowed&~known&valid;a[fill]=shifted[fill];known|=fill
    count=int((known&~original).sum())
    if not count:raise ValueError('奥側の補完可能領域がありません。重なりと奥行きを確認してください')
    path='assets/hidden-'+uid()+'.png';Image.fromarray(a).save(folder/path)
    part['hiddenOriginal']=part['image'];part['image']=path;part['completion']='boundary-extension'
    part['completionInfo']={'pixels':pixels,'filled':count,'generated':False}
    return p


def build_head_surface(store,project):
    """A shared camera-space surface. Missing/occluded geometry is not invented."""
    p=copy.deepcopy(project);folder=store.folder(p['id']);arrays=p.get('depth',{}).get('arrays',{})
    if not all(k in arrays for k in ('points.npy','mask.npy','intrinsics.npy')):raise ValueError('MoGe3の結果を取り込んでから顔面を構築してください')
    points=np.load(folder/arrays['points.npy'],allow_pickle=False);mask=np.load(folder/arrays['mask.npy'],allow_pickle=False).astype(bool)
    K=np.load(folder/arrays['intrinsics.npy'],allow_pickle=False)
    h,w=mask.shape
    if points.shape!=(h,w,3) or K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Invalid camera geometry')
    # MoGe returns intrinsics in normalized image coordinates.
    rect=p.get('faceInput',{}).get('alignment',{}).get('rect',[.25,.02,.5,.32]);x,y,rw,rh=rect
    good=mask&np.isfinite(points).all(axis=2)&(points[:,:,2]>0)
    crop=points[int(y*h):int((y+rh)*h),int(x*w):int((x+rw)*w),2]
    valid=good[int(y*h):int((y+rh)*h),int(x*w):int((x+rw)*w)]
    if valid.sum()<20:raise ValueError('顔の有効な3D点が不足しています')
    reference=float(np.median(crop[valid]));limit=reference*rw/float(K[0,0])*.22
    cols,rows=49,73;depth=np.full((rows,cols),reference,dtype=float);confidence=np.zeros((rows,cols))
    for j in range(rows):
        for i in range(cols):
            u,v=i/(cols-1),j/(rows-1);xx=min(w-1,round(u*(w-1)));yy=min(h-1,round(v*(h-1)))
            yy0,yy1=max(0,yy-3),min(h,yy+4);xx0,xx1=max(0,xx-3),min(w,xx+4)
            values=points[yy0:yy1,xx0:xx1,2][good[yy0:yy1,xx0:xx1]]
            if len(values):depth[j,i]=np.clip(np.median(values),reference-limit,reference+limit);confidence[j,i]=1
    # Shared smoothing suppresses isolated depth noise; no per-layer normalization.
    for _ in range(3):
        a=np.pad(depth,1,mode='edge');depth=(a[1:-1,1:-1]*4+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])/8
    pivot=p.get('landmarks',{}).get('joints',{}).get('neck',{}).get('position',[.5,.315])
    p['headSurface']={'enabled':True,'version':1,'cols':cols,'rows':rows,'depth':depth.ravel().tolist(),'intrinsics':K.tolist(),'pivot':[pivot[0],pivot[1],reference],
       'source':'moge3-points','sourceJob':p['depth'].get('provenance',{}).get('jobId'),'strength':1,'maxAngle':15,
       'validFraction':float(confidence.mean()),'depthRange':[float(depth.min()),float(depth.max())],'limit':limit}
    return p


def fit_direction(project,side,base_points,target_points,angle=18):
    if side not in ('left','right'):raise ValueError('Invalid direction')
    if type(angle) not in (int,float) or not math.isfinite(angle) or not 5<=angle<=30:raise ValueError('基準角度は5〜30度です')
    a=check_points(base_points);b=check_points(target_points,len(a))
    if len(a)<6:raise ValueError('目・鼻・口・輪郭に6点以上を指定してください')
    p=copy.deepcopy(project);entry=p.get('directionalFace',{}).get('views',{}).get(side)
    if not entry:raise ValueError('先に斜め顔画像を登録してください')
    if not p.get('faceInput'):raise ValueError('正面顔画像を登録してください')
    rect=p['faceInput']['alignment']['rect'];r=np.array(rect);src=a*r[2:]+r[:2];dst=b*r[2:]+r[:2]
    anchors=np.array([[0,0],[.5,0],[1,0],[0,.5],[1,.5],[0,1],[.5,1],[1,1]])
    src=np.vstack([src,anchors]);dst=np.vstack([dst,anchors]);n=len(src)
    if np.min(np.linalg.norm(src[:,None]-src[None,:]+np.eye(n)[:,:,None]*10,axis=2))<1e-5:raise ValueError('対応点が重複しています')
    def kernel(x,y):
        r2=((x[:,None]-y[None,:])**2).sum(axis=2);return r2*np.log(np.maximum(r2,1e-12))
    P=np.column_stack([np.ones(n),src]);system=np.block([[kernel(src,src)+np.eye(n)*1e-8,P],[P.T,np.zeros((3,3))]])
    try:weights=np.linalg.solve(system,np.vstack([dst-src,np.zeros((3,2))]))
    except np.linalg.LinAlgError:raise ValueError('対応点の配置が不安定です。顔全体へ点を分散してください')
    cols,rows=49,73;grid=np.array([[i/(cols-1),j/(rows-1)] for j in range(rows) for i in range(cols)])
    delta=np.column_stack([kernel(grid,src),np.ones(len(grid)),grid])@weights
    target=(grid+delta).reshape(rows,cols,2)
    for j in range(rows-1):
        for i in range(cols-1):
            for aa,bb,cc in ((target[j,i],target[j,i+1],target[j+1,i]),(target[j,i+1],target[j+1,i+1],target[j+1,i])):
                u,v=bb-aa,cc-aa
                if u[0]*v[1]-u[1]*v[0]<.02/((cols-1)*(rows-1)):raise ValueError('メッシュが反転します。対応点を修正してください')
    entry.update(basePoints=a.tolist(),targetPoints=b.tolist(),angle=angle,cols=cols,rows=rows,offsets=delta.tolist(),frontImage=p['faceInput']['image'],rect=rect.copy(),method='thin-plate-spline',fitted=True)
    p['directionalFace']['enabled']=True
    if p.get('multiViewFace'):p['multiViewFace']['enabled']=False
    return p
