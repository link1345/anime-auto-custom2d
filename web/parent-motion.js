// Shared attachment coordinates; shape attenuation never attenuates attachment motion.
export const defaults={enabled:true,face:.15,eyes:.15,mouth:.25,hair:0,body:.25,lag:.2,light:.12,shadow:.1,camera:0,cursorMode:0,cameraDistance:.5,targetDepth:.5,cursorDepthFollow:.5,relief:1,faceTilt:0,yawLimit:20,pitchLimit:15,hairRelief:.5,gazeEnabled:1,gazeRangeX:.2,gazeRangeY:.15,neckRelief:.5,bodyRelief:.5};
// Mesh evaluation resolves settings once per invocation; topology is shared across frames.
let evaluation=null;
function settingsFor(p){return evaluation?.project===p?evaluation.settings:{...defaults,...p.parentMotion};}
const attachmentUV=[],attachmentTriangles=[];
for(let j=0;j<73;j++)for(let i=0;i<49;i++){
 const n=j*49+i;attachmentUV.push([i/48,j/72]);
 if(i<48&&j<72)attachmentTriangles.push([n,n+1,n+49],[n+1,n+50,n+49]);
}
// Rotate the light once per frame instead of rotating 16,384 normals.
export function createLightingDelta(values){
 const rad=Math.PI/180,yaw=-(values.headX||0)*rad,pitch=(values.headY||0)*rad,roll=(values.headZ||0)*rad;
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),cr=Math.cos(roll),sr=Math.sin(roll);
 const lx=-.4*cr-.35*sr,ly=.4*sr-.35*cr;
 const py=ly*cp+.85*sp,pz=-ly*sp+.85*cp;
 const x=lx*cy-pz*sy,y=py,z=lx*sy+pz*cy;
 const clamp=v=>Math.max(0,Math.min(1,v));
 return (dx,dy)=>{const n=Math.hypot(dx,dy,1);return clamp((x*dx+y*dy+z)/n)-clamp((-.4*dx-.35*dy+.85)/n);};
}
export function lightingDelta(dx,dy,values){return createLightingDelta(values)(dx,dy);}
export function balancedLighting(deltas,weights){
 let sum=0,total=0;for(let i=0;i<deltas.length;i++){sum+=deltas[i]*weights[i];total+=weights[i];}
 if(!total)return deltas.map(()=>0);
 const mean=sum/total,centered=deltas.map(d=>d-mean);
 let peak=0;for(let i=0;i<centered.length;i++)if(weights[i])peak=Math.max(peak,Math.abs(centered[i]));
 const scale=Math.min(1,.12/Math.max(peak,1e-9));
 return centered.map((d,i)=>weights[i]?d*scale:0);
}
export function hairDepth(p,part,u,v,body=false){
 const s=p.parentMotion??{},g=(body?s.bodySurfaces:s.hairSurfaces)?.[part.id];
 if(!g||g.image!==part.image||!g.rect?.every((n,i)=>n===(part.rect??[0,0,1,1])[i])||g.sourceArray!==p.depth?.arrays?.['points.npy'])return .35;
 const x=Math.max(0,Math.min(48,u*48)),y=Math.max(0,Math.min(72,v*72)),i=Math.min(47,Math.floor(x)),j=Math.min(71,Math.floor(y)),a=x-i,b=y-j;
 const sample=(dx,dy)=>g.values[(j+dy)*49+i+dx];
 const depth=(sample(0,0)*(1-a)+sample(1,0)*a)*(1-b)+(sample(0,1)*(1-a)+sample(1,1)*a)*b;
 return .35+(depth-.35)*(body?1:(s.hairRelief??.5));
}
// Both deformation and shading sample the same retained depth field.
export function faceDepth(p,u,v){
 const s=settingsFor(p),r=p.faceInput?.alignment?.rect??[.25,.01,.5,.35],x=(u-r[0])/r[2]-.5,y=(v-r[1])/r[3]-.55;
 const surface=s.surface,g=surface?.grid;
 if(g&&g.sourceAsset===p.faceInput?.image&&g.sourceArray===p.faceDepths?.front?.arrays?.['points.npy']&&g.rect.every((n,i)=>n===r[i])){
  const gx=Math.max(0,Math.min(127,(x+.5)*127)),gy=Math.max(0,Math.min(127,(y+.55)*127)),i=Math.min(126,Math.floor(gx)),j=Math.min(126,Math.floor(gy)),a=gx-i,b=gy-j;
  const at=(dx,dy)=>{const n=(j+dy)*128+i+dx;return g.residual[n]*g.confidence[n];};
  const residual=(at(0,0)*(1-a)+at(1,0)*a)*(1-b)+(at(0,1)*(1-a)+at(1,1)*a)*b;
  return Math.max(-2,Math.min(3,.5+s.faceTilt*(g.plane[1]*x+g.plane[2]*y)+s.relief*residual));
 }
 const c=surface?.coefficients??[.7,0,0,-1.2,0,-1.2];
 return s.relief*Math.max(0,Math.min(1,c[0]+c[1]*x+c[2]*y+c[3]*x*x+c[4]*x*y+c[5]*y*y));
}
// Camera looks along +Z. A model-height unit is used, not measured metres.
export function cameraProject(p,part,point,values){
 const s=settingsFor(p),D=1+4*s.cameraDistance,aspect=p.size[0]/p.size[1];
 const pivot=p.landmarks?.joints?.neck?.position??[.5,.315],r=p.faceInput?.alignment?.rect??[.25,.01,.5,.35];
 const nx=(point[0]-r[0])/r[2]-.5,ny=(point[1]-r[1])/r[3]-.55,c=s.surface?.coefficients??[.7,0,0,-1.2,0,-1.2];
 const relief=part.role.startsWith('hair')?hairDepth(p,part,...point):faceDepth(p,...point);
 const z=D-.12*relief,px=(pivot[0]-.5)*aspect,py=pivot[1]-.5;
 const restScale=s.camera===2?1:z/D;
 let x=(point[0]-.5)*aspect*restScale-px,y=(point[1]-.5)*restScale-py,zz=z-D;
 const yaw=-Math.max(-s.yawLimit,Math.min(s.yawLimit,values.headX||0))*Math.PI/180,pitch=Math.max(-s.pitchLimit,Math.min(s.pitchLimit,values.headY||0))*Math.PI/180,roll=Math.max(-12,Math.min(12,values.headZ||0))*Math.PI/180;
 const xx=Math.cos(yaw)*x+Math.sin(yaw)*zz,zx=-Math.sin(yaw)*x+Math.cos(yaw)*zz;
 const yy=Math.cos(pitch)*y-Math.sin(pitch)*zx,zy=Math.sin(pitch)*y+Math.cos(pitch)*zx;
 x=Math.cos(roll)*xx-Math.sin(roll)*yy;y=Math.sin(roll)*xx+Math.cos(roll)*yy;
 const cameraZ=Math.max(.1,D+zy);
 if(s.camera===2)return [.5+(x+px)/aspect,.5+y+py,cameraZ];
 return [.5+D*(x+px)/(cameraZ*aspect),.5+D*(y+py)/cameraZ,cameraZ];
}
// Depth is bounded in front of the model: never send the look target behind the head.
export function cursorTargetDepth(p,point){
 const s=settingsFor(p),D=1+4*s.cameraDistance,aspect=p.size[0]/p.size[1];
 const f=p.face,center=f?[(f.leftEye[0]+f.rightEye[0])/2,(f.leftEye[1]+f.rightEye[1])/2]:[.5,.2];
 const distance=Math.hypot((point[0]-center[0])*aspect,point[1]-center[1]);
 const t=Math.min(1,distance/.5),smooth=t*t*(3-2*t),base=.2+.6*s.targetDepth;
 return D*(base+(.9-base)*s.cursorDepthFollow*smooth);
}
export function cursorAngles(p,point){
 const s=settingsFor(p);
 if(!s.enabled||!(p.parentMotion?.cursorMode??(s.camera===1?1:0)))return {headX:(point[0]-.5)*35*(s.enabled?s.yawLimit/20:1),headY:(point[1]-.3)*20*(s.enabled?s.pitchLimit/15:1)};
 const D=1+4*s.cameraDistance,z=cursorTargetDepth(p,point),aspect=p.size[0]/p.size[1];
 const face=p.face,center=face?[(face.leftEye[0]+face.rightEye[0])/2,(face.leftEye[1]+face.rightEye[1])/2]:[.5,.2];
 // Intersect the cursor ray with the distance-dependent target plane.
 const rayScale=s.camera===1?z/D:1;
 const tx=(point[0]-.5)*aspect*rayScale,ty=(point[1]-.5)*rayScale;
 const hx=(center[0]-.5)*aspect,hy=center[1]-.5,distance=D-z;
 return {headX:Math.max(-s.yawLimit,Math.min(s.yawLimit,Math.atan2(tx-hx,distance)*180/Math.PI)),headY:Math.max(-s.pitchLimit,Math.min(s.pitchLimit,Math.atan2(ty-hy,distance)*180/Math.PI))};
}
export function follow(previous,target,dt,lag){return previous+(target-previous)*(1-Math.exp(-Math.max(0,Math.min(.1,dt))/Math.max(.02,lag)));}
// Rig root may be placed at the chest; it is not necessarily the waist.
export function bodyAnchors(p){
 const joints=p.landmarks?.joints??{},neck=joints.neck?.position??[.5,.315];
 const valid=q=>Array.isArray(q)&&q.length===2&&q.every(Number.isFinite);
 const hips=[joints.hipLeft?.position,joints.hipRight?.position];
 const hip=hips.every(valid)?[(hips[0][0]+hips[1][0])/2,(hips[0][1]+hips[1][1])/2]:null;
 const root=joints.root?.position;
 const waist=valid(hip)&&hip[1]>neck[1]+.1?hip:valid(root)&&root[1]>neck[1]+.1?root:[neck[0],Math.max(.59,neck[1]+.2)];
 return {neck,waist};
}
export function bodyPoint(p,point,state){
 const {neck,waist}=bodyAnchors(p);
 const [x,y]=point,t=Math.max(0,Math.min(1,(waist[1]-y)/Math.max(.1,waist[1]-neck[1]))),w=t*t*(3-2*t);
 return [x+state.x*.0012*w-(y-waist[1])*state.z*.008*w,y+state.z*.008*(x-waist[0])*w+state.y*.00035*w];
}
export function faceOffset(p,u,v,values){
 const r=p.faceInput?.alignment?.rect??[.25,.01,.5,.35],x=(u-r[0])/r[2]-.5,y=(v-r[1])/r[3]-.55;
 const yaw=Math.max(-(p.parentMotion?.yawLimit??20)/20,Math.min((p.parentMotion?.yawLimit??20)/20,(values.headX||0)/20));
 // A shallow, smooth hair-free face template. Optional masked depth only changes its relief.
 const c=p.parentMotion?.surface?.coefficients??[.7,0,0,-1.2,0,-1.2];
 const relief=faceDepth(p,u,v);
 const strength=1;
 return [yaw*r[2]*(.045*relief*strength-.025*x*yaw),Math.max(-(p.parentMotion?.pitchLimit??15)/15,Math.min((p.parentMotion?.pitchLimit??15)/15,(values.headY||0)/15))*r[3]*.02*relief*strength];
}
export function groupAnchor(p,part,u){
 const r=p.faceInput?.alignment?.rect??[.25,.01,.5,.35],face=p.face;
 if(['irides','eyewhite','eyelash','eyebrow'].includes(part.tag))return u<(r[0]+r[2]/2)?(face?.leftEye??[r[0]+r[2]*.35,r[1]+r[3]*.5]):(face?.rightEye??[r[0]+r[2]*.65,r[1]+r[3]*.5]);
 if(part.tag==='mouth')return face?.mouth??[r[0]+r[2]*.5,r[1]+r[3]*.7];
 return [r[0]+r[2]/2,r[1]+r[3]*.55];
}
export function attachedPoint(p,part,point,values,state,restPoint=point){
 const s=settingsFor(p),[u,v]=point,head=part.scope==='face'||part.role==='head'||part.role.startsWith('hair');
 let x=u,y=v;
 if(head&&s.camera){
  const eye=['irides','eyewhite','eyelash','eyebrow'].includes(part.tag),mouth=part.tag==='mouth',anchor=groupAnchor(p,part,u);
  let q=cameraProject(p,part,point,values);
  if(eye||mouth){const a=cameraProject(p,{role:'head',tag:'face'},anchor,values),w=eye?s.eyes:s.mouth;q=[a[0]+u-anchor[0]+w*(q[0]-a[0]-(u-anchor[0])),a[1]+v-anchor[1]+w*(q[1]-a[1]-(v-anchor[1])),q[2]];}
  return [...bodyPoint(p,q,state),q[2]];
 }
 if(head){
  const hair=part.role.startsWith('hair'),eye=['irides','eyewhite','eyelash','eyebrow'].includes(part.tag),anchor=groupAnchor(p,part,u);
  const center=groupAnchor(p,{role:'head'},u),origin=faceOffset(p,...center,values);
  const faceAt=(x,y)=>faceOffset(p,x,y,values).map((n,i)=>origin[i]+s.face*(n-origin[i]));
  const a=faceAt(...anchor),d=faceAt(u,v),weight=hair?s.hair:eye?s.eyes:part.tag==='mouth'?s.mouth:1;
  // Hair roots inherit head motion only; eyes retain their face attachment even at zero shape weight.
  if(!hair){x+=a[0]+weight*(d[0]-a[0]);y+=a[1]+weight*(d[1]-a[1]);}
  else {const depth=hairDepth(p,part,u,v)-.35;x+=weight*d[0]+Math.sin((values.headX||0)*Math.PI/180)*depth*.08;y+=weight*d[1]+Math.sin((values.headY||0)*Math.PI/180)*depth*.05;}
  const pivot=p.landmarks?.joints?.neck?.position??[.5,.315],aspect=p.size[0]/p.size[1],angle=Math.max(-12,Math.min(12,values.headZ||0))*Math.PI/180;
  const dx=(x-pivot[0])*aspect,dy=y-pivot[1];x=pivot[0]+(Math.cos(angle)*dx-Math.sin(angle)*dy)/aspect+(values.headX||0)*.00045;y=pivot[1]+Math.sin(angle)*dx+Math.cos(angle)*dy;
 }
 if(!head){
  const depth=(hairDepth(p,part,restPoint[0],restPoint[1],true)-.35)*(part.role==='neck'?s.neckRelief:s.bodyRelief);
  const {neck,waist}=bodyAnchors(p);
  const t=Math.max(0,Math.min(1,(waist[1]-v)/Math.max(.1,waist[1]-neck[1]))),w=t*t*(3-2*t);
  // A shallow depth displacement layered after skinning. Neutral view stays exact.
  x+=Math.sin(state.x*Math.PI/180)*depth*.08*w;y+=Math.sin(state.y*Math.PI/180)*depth*.05*w;
  return [...bodyPoint(p,[x,y],state),(point[2]??0)+depth*.08*w];
 }
 return [...bodyPoint(p,[x,y],state),point[2]??0];
}
export function attachmentMesh(p,part,values,state){
 const previous=evaluation;evaluation={project:p,settings:{...defaults,...p.parentMotion}};
 try{return {vertices:attachmentUV.map(point=>attachedPoint(p,part,point,values,state)),uv:attachmentUV,triangles:attachmentTriangles};}
 finally{evaluation=previous;}
}

export function cursorGaze(p,point){
 const settings=settingsFor(p);
 if(settings.enabled&&(p.parentMotion?.cursorMode??(settings.camera===1?1:0))){const a=cursorAngles(p,point);return {gazeX:a.headX/settings.yawLimit,gazeY:a.headY/settings.pitchLimit};}
 const eyes=p.face,center=eyes?[(eyes.leftEye[0]+eyes.rightEye[0])/2,(eyes.leftEye[1]+eyes.rightEye[1])/2]:[.5,.2];
 const clamp=x=>Math.max(-1,Math.min(1,x));
 return {gazeX:clamp((point[0]-center[0])/.35),gazeY:clamp((point[1]-center[1])/.35)};
}
export function irisOffset(eye,settings,values){
 if(!settings.gazeEnabled)return [0,0];
 const [l,t,r,b]=eye.bbox,clamp=x=>Math.max(-1,Math.min(1,x||0));
 return [(r-l)*.5*settings.gazeRangeX*clamp(values.gazeX),(b-t)*.5*settings.gazeRangeY*clamp(values.gazeY)];
}

