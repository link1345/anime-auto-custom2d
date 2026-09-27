"""Fit one shallow face surface from independently inferred, registered views."""
import copy
import hashlib
import numpy as np


def similarity(source,target):
    if len(source)<6:raise ValueError('有効な3D対応点が6組以上必要です')
    weights=np.ones(len(source));rotation=np.eye(3);scale=1.;translation=np.zeros(3)
    for _ in range(5):
        weights/=weights.sum();a=(source*weights[:,None]).sum(0);b=(target*weights[:,None]).sum(0)
        x=source-a;y=target-b
        if np.linalg.matrix_rank(x,tol=1e-7)<2:raise ValueError('3D対応点の配置が退化しています')
        u,s,vt=np.linalg.svd((x*weights[:,None]).T@y);d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
        rotation=vt.T@d@u.T;scale=float((s*np.diag(d)).sum()/max((weights[:,None]*x*x).sum(),1e-12));translation=b-scale*rotation@a
        error=np.linalg.norm(source@rotation.T*scale+translation-target,axis=1)
        cutoff=max(float(np.median(error))*2,1e-5);weights=np.minimum(1,cutoff/np.maximum(error,1e-9))
    if not .1<scale<10:raise ValueError('推定の縮尺差が大きすぎます。画像と対応点を確認してください')
    return rotation,scale,translation,error


def sample(points,mask,uv):
    h,w=mask.shape;result=[];valid=[]
    for u,v in uv:
        x=min(w-1,max(0,round(u*(w-1))));y=min(h-1,max(0,round(v*(h-1))))
        a=points[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)];m=mask[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]
        m=m&np.isfinite(a).all(2)&(a[:,:,2]>0)
        valid.append(bool(m.any()));result.append(np.median(a[m],axis=0) if m.any() else [0,0,0])
    return np.array(result),np.array(valid)


def tps(points,delta,queries):
    anchors=np.array([[0,0],[.5,0],[1,0],[0,.5],[1,.5],[0,1],[.5,1],[1,1]])
    points=np.vstack([points,anchors]);delta=np.vstack([delta,np.zeros((8,delta.shape[1]))]);n=len(points)
    def kernel(a,b):
        r=((a[:,None]-b[None,:])**2).sum(2);return r*np.log(np.maximum(r,1e-12))
    p=np.column_stack([np.ones(n),points]);system=np.block([[kernel(points,points)+np.eye(n)*1e-7,p],[p.T,np.zeros((3,3))]])
    try:coef=np.linalg.solve(system,np.vstack([delta,np.zeros((3,delta.shape[1]))]))
    except np.linalg.LinAlgError:raise ValueError('対応点の補間が不安定です')
    return np.column_stack([kernel(queries,points),np.ones(len(queries)),queries])@coef


def project(points,K):
    if np.any(points[:,2]<=0):raise ValueError('顔面がカメラの後ろへ移動します')
    return points[:,:2]/points[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]


def unfold(uv,cols,rows):
    a=uv.reshape(rows,cols,2);u=a[:-1,1:]-a[:-1,:-1];v=a[1:,:-1]-a[:-1,:-1]
    u2=a[1:,1:]-a[:-1,1:];v2=a[1:,:-1]-a[:-1,1:]
    cross=lambda x,y:x[:,:,0]*y[:,:,1]-x[:,:,1]*y[:,:,0]
    return bool(min(cross(u,v).min(),cross(u2,v2).min())>1e-7)


def fit_multiview(store,project_data,strength=.35,correction=1):
    p=copy.deepcopy(project_data);folder=store.folder(p['id']);face=p.get('faceInput',{});rect=np.array(face.get('alignment',{}).get('rect',[]))
    if rect.shape!=(4,):raise ValueError('正面顔の配置が必要です')
    records=p.get('faceDepths',{})
    def load(scope,asset):
        rec=records.get(scope)
        if not rec or rec['sourceAsset']!=asset or hashlib.sha256((folder/asset).read_bytes()).hexdigest()!=rec['inputSha256']:raise ValueError(scope+' の顔専用MoGe3結果がありません、または画像変更後で古くなっています')
        arrays=rec['arrays'];points=np.load(folder/arrays['points.npy'],allow_pickle=False);mask=np.load(folder/arrays['mask.npy'],allow_pickle=False).astype(bool);K=np.load(folder/arrays['intrinsics.npy'],allow_pickle=False)
        if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError('Invalid camera intrinsics')
        return points,mask,K
    front,front_mask,K=load('front',face['image']);views={};all_src=[];all_delta=[];sources={'front':records['front']['inputSha256']}
    valid=front_mask&np.isfinite(front).all(2)&(front[:,:,2]>0)
    if valid.sum()<20:raise ValueError('正面顔の有効点が不足しています')
    reference=float(np.median(front[:,:,2][valid]));limit=reference/K[0,0]*.15
    for side,entry in p.get('directionalFace',{}).get('views',{}).items():
        if side not in records:continue
        if not entry.get('fitted') or entry.get('frontImage')!=face['image'] or entry.get('rect')!=rect.tolist():raise ValueError(side+' の対応点を再保存してください')
        other,other_mask,k=load(side,entry['image']);a=np.asarray(entry['basePoints']);b=np.asarray(entry['targetPoints'])
        aa,va=sample(front,front_mask,a);bb,vb=sample(other,other_mask,b);keep=va&vb;a=a[keep];b=b[keep];aa=aa[keep];bb=bb[keep]
        R,s,t,error=similarity(bb,aa);M=R.T/s;d=-M@t;aligned=bb@R.T*s+t
        rays=np.column_stack([(a[:,0]-K[0,2])/K[0,0],(a[:,1]-K[1,2])/K[1,1],np.ones(len(a))])
        q=(b-[k[0,2],k[1,2]])/[k[0,0],k[1,1]];mr=rays@M.T
        A=mr[:,:2]-q*mr[:,2,None];B=q*d[2]-d[:2]
        prior=(aa[:,2]+aligned[:,2])/2;reg=np.maximum((A*A).sum(1),.001)*2
        fitted=((A*B).sum(1)+reg*prior)/((A*A).sum(1)+reg)
        fitted=np.clip(fitted,reference-limit,reference+limit)
        all_src.extend(a*rect[2:]+rect[:2]);all_delta.extend((fitted-aa[:,2])[:,None])
        kg=k.copy();kg[0,0]*=rect[2];kg[1,1]*=rect[3];kg[0,2]=rect[0]+k[0,2]*rect[2];kg[1,2]=rect[1]+k[1,2]*rect[3]
        views[side]={'rotation':R.T.tolist(),'scale':1/s,'translation':d.tolist(),'intrinsics':kg.tolist(),'angle':entry['angle'],'basePoints':a.tolist(),'targetPoints':b.tolist(),'alignmentRms':float(np.sqrt(np.mean(error**2))),'pairs':len(a),'image':entry['image']}
        sources[side]=records[side]['inputSha256']
    if not views:raise ValueError('斜め顔のMoGe3結果と対応点が必要です')
    cols,rows=49,73;grid=np.array([[i/(cols-1),j/(rows-1)] for j in range(rows) for i in range(cols)])
    local=(grid-rect[:2])/rect[2:];xyz,ok=sample(front,front_mask,local);ok&=(local>=0).all(1)&(local<=1).all(1)
    depth=np.where(ok,xyz[:,2],reference)
    # Combine duplicate frontal controls when both side views are present.
    groups={}
    for uv,dz in zip(all_src,all_delta):groups.setdefault(tuple(uv),[]).append(dz)
    depth+=tps(np.array(list(groups)),np.array([np.mean(v,axis=0) for v in groups.values()]),grid)[:,0]
    depth=np.clip(depth,reference-limit,reference+limit).reshape(rows,cols)
    for _ in range(4):
        pad=np.pad(depth,1,mode='edge');depth=(4*depth+pad[:-2,1:-1]+pad[2:,1:-1]+pad[1:-1,:-2]+pad[1:-1,2:])/8
    depth=reference+(depth.ravel()-reference)*strength
    kg=K.copy();kg[0,0]*=rect[2];kg[1,1]*=rect[3];kg[0,2]=rect[0]+K[0,2]*rect[2];kg[1,2]=rect[1]+K[1,2]*rect[3]
    rays=np.column_stack([(grid[:,0]-kg[0,2])/kg[0,0],(grid[:,1]-kg[1,2])/kg[1,1],np.ones(len(grid))]);points=rays*depth[:,None]
    for side,v in views.items():
        M=np.array(v['rotation'])*v['scale'];d=np.array(v['translation']);vk=np.array(v['intrinsics']);a=np.array(v['basePoints'])*rect[2:]+rect[:2];b=np.array(v['targetPoints'])*rect[2:]+rect[:2]
        # Bilinear sample the same grid used by the renderer for diagnostics.
        def interp(uv):
            x=np.clip(uv[:,0]*(cols-1),0,cols-1-1e-8);y=np.clip(uv[:,1]*(rows-1),0,rows-1-1e-8);i=x.astype(int);j=y.astype(int);dx=x-i;dy=y-j;g=depth.reshape(rows,cols)
            return g[j,i]*(1-dx)*(1-dy)+g[j,i+1]*dx*(1-dy)+g[j+1,i]*(1-dx)*dy+g[j+1,i+1]*dx*dy
        z=interp(a);q=np.column_stack([(a[:,0]-kg[0,2])/kg[0,0],(a[:,1]-kg[1,2])/kg[1,1],np.ones(len(a))])*z[:,None]
        before=project(q@M.T+d,vk);residual=b-before
        # Cap artist correction to 8% of the face image, separately from depth.
        norm=np.linalg.norm(residual/rect[2:],axis=1);factor=np.minimum(1,.08/np.maximum(norm,1e-8));residual*=factor[:,None]
        offset=tps(a,residual,grid);endpoint=project(points@M.T+d,vk)
        rotation=np.array(v['rotation']);theta=float(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))
        if theta>np.pi/2:raise ValueError('推定回転が大きすぎます。浅い斜め画像と対応点を確認してください')
        axis=np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],rotation[1,0]-rotation[0,1]])
        axis/=max(np.linalg.norm(axis),1e-12);ax,ay,az=axis;cross=np.array([[0,-az,ay],[az,0,-ax],[-ay,ax,0]])
        frames=[]
        for amount in (.25,.5,.75,1):
            r=np.eye(3)+np.sin(theta*amount)*cross+(1-np.cos(theta*amount))*(cross@cross)
            camera=points@r.T*np.exp(np.log(v['scale'])*amount)+d*amount
            frames.append((project(camera,kg+(vk-kg)*amount),amount*amount*(3-2*amount)))
        gain=correction
        stable=lambda value:all(unfold(frame+offset*value*weight,cols,rows) for frame,weight in frames)
        while gain>.01 and not stable(gain):gain*=.5
        if not stable(gain):raise ValueError('位置合わせで顔面が反転します。対応点を確認してください')
        def raster_sample(vertices):
            x=np.clip(a[:,0]*(cols-1),0,cols-1-1e-8);y=np.clip(a[:,1]*(rows-1),0,rows-1-1e-8);ii=x.astype(int);jj=y.astype(int);dx=(x-ii)[:,None];dy=(y-jj)[:,None];g=vertices.reshape(rows,cols,2)
            upper=dx+dy<=1
            first=g[jj,ii]*(1-dx-dy)+g[jj,ii+1]*dx+g[jj+1,ii]*dy
            second=g[jj,ii+1]*(1-dy)+g[jj+1,ii+1]*(dx+dy-1)+g[jj+1,ii]*(1-dx)
            return np.where(upper,first,second)
        before_pixels=(raster_sample(endpoint)-b)/rect[2:]*2160
        after_pixels=(raster_sample(endpoint+offset*gain)-b)/rect[2:]*2160
        v.update(residualOffsets=offset.tolist(),correctionGain=gain,rmseBeforePixels=float(np.sqrt(np.mean(np.sum(before_pixels**2,axis=1)))),rmseAfterPixels=float(np.sqrt(np.mean(np.sum(after_pixels**2,axis=1)))),rotationDegrees=float(np.degrees(np.arccos(np.clip((np.trace(M/v['scale'])-1)/2,-1,1)))))
    p['multiViewFace']={'enabled':True,'version':1,'cols':cols,'rows':rows,'depth':depth.tolist(),'intrinsics':kg.tolist(),'pivot':[.5,.315,reference],'strength':strength,'correction':correction,'views':views,'sourceHashes':sources,'frontImage':face['image'],'rect':rect.tolist(),'maxAngle':15,'source':'registered-face-moge3','warning':'対応点の誤差は目測点に対する値で、輪郭や画質全体の一致率ではありません。髪で隠れた顔は復元しません。'}
    p.setdefault('directionalFace',{'views':{}})['enabled']=False
    return p
