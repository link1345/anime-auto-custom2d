import assert from 'node:assert/strict';
import {attachedPoint,follow,defaults,attachmentMesh} from '../web/parent-motion.js';
const p={size:[1024,1536],face:{leftEye:[.4,.19],rightEye:[.6,.19],mouth:[.5,.25]},parentMotion:{...defaults,eyes:0},faceInput:{alignment:{rect:[.25,.01,.5,.35]}}};
const state={x:0,y:0,z:0},v={headX:20},eye={role:'head',scope:'face',tag:'irides'};
const a=attachedPoint(p,eye,[.38,.19],v,state),b=attachedPoint(p,eye,[.42,.19],v,state);
assert.ok(Math.abs(b[0]-a[0]-.04)<1e-10,'zero shape weight preserves eye width');
assert.ok(a[0]>.38,'zero shape weight still follows attachment');
assert.deepEqual(attachedPoint(p,eye,p.face.leftEye,v,state),attachedPoint(p,{role:'head',tag:'face'},p.face.leftEye,v,state),'eye attachment stays on skin at every shape weight');
for(const tag of ['irides','eyewhite','eyelash'])assert.deepEqual(attachedPoint(p,{...eye,tag},[.4,.19],v,state),attachedPoint(p,eye,[.4,.19],v,state));
assert.deepEqual(attachedPoint(p,{role:'hairFront'},[.4,.19],v,state),attachedPoint({...p,parentMotion:{...p.parentMotion,face:1,eyes:1}},{role:'hairFront'},[.4,.19],v,state));
assert.deepEqual(attachedPoint(p,eye,[.4,.19],{},state),[.4,.19,0]);
assert.ok(follow(0,1,.016,.2)>0&&follow(0,1,.016,.2)<1);
assert.equal(follow(.4,1,0,.2),.4);
for(const headX of [-20,0,20])for(const headZ of [-12,12]){
 const m=attachmentMesh(p,{role:'head',tag:'face'},{headX,headZ},{x:5,y:2,z:3});
 for(const tri of m.triangles){const [a,b,c]=tri.map(i=>m.vertices[i]);assert.ok((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])>0);}
}
console.log('Attachment preservation, eye/mask consistency, hair independence, lag, and non-inversion passed');
const {cameraProject,cursorAngles}=await import('../web/parent-motion.js');
const cp={...p,parentMotion:{...p.parentMotion,camera:1,cursorMode:1}};
for(const part of [eye,{role:'hairFront'}])for(const q of [[.4,.19],[.6,.19],[.5,.3]]){
 const projected=cameraProject(cp,part,q,{});assert.ok(Math.abs(projected[0]-q[0])<1e-10&&Math.abs(projected[1]-q[1])<1e-10,'camera neutral matches artwork');
}
assert.ok(cursorAngles(cp,[.9,.2]).headX>0&&cursorAngles(cp,[.1,.2]).headX<0);
assert.notDeepEqual(cameraProject(cp,eye,[.6,.19],{headX:20}),cameraProject({...cp,parentMotion:{...cp.parentMotion,cameraDistance:0}},eye,[.6,.19],{headX:20}));
assert.deepEqual(cursorAngles(p,[.8,.3]),{headX:(.8-.5)*35,headY:0});
for(const x of [-20,20]){const m=attachmentMesh(cp,{role:'head',tag:'face'},{headX:x,headY:15},state);assert.ok(m.vertices.every(q=>q.every(Number.isFinite)));}
console.log('Perspective camera neutral alignment, distance response, cursor ray direction and legacy mapping passed');
const {lightingDelta}=await import('../web/parent-motion.js');
for(const dx of [-2,0,2])for(const dy of [-2,0,2])assert.equal(lightingDelta(dx,dy,{}),0);
assert.notEqual(lightingDelta(.3,.2,{headX:20}),0);
const {Renderer}=await import('../web/renderer.js');
Renderer.prototype.shadeSkin.call({}, {drawImage(){throw Error('Neutral color must not be modified');}},cp,{headX:0,headY:0,headZ:0},{light:1,shadow:1});
const flat={...cp,parentMotion:{...cp.parentMotion,relief:0}};
assert.notDeepEqual(cameraProject(cp,eye,[.5,.2],{headX:20}),cameraProject(flat,eye,[.5,.2],{headX:20}));
for(const q of [[.4,.19],[.6,.19]]){const a=cameraProject(flat,eye,q,{});assert.ok(Math.abs(a[0]-q[0])<1e-10&&Math.abs(a[1]-q[1])<1e-10);}
console.log('Neutral max-light/shadow bypass and independent face relief passed');
const wide={...cp,parentMotion:{...cp.parentMotion,yawLimit:45,pitchLimit:30}};
assert.equal(cursorAngles(wide,[100,100]).headX,45);assert.equal(cursorAngles(wide,[100,100]).headY,30);
assert.notDeepEqual(cameraProject(wide,eye,[.4,.19],{headX:40}),cameraProject(cp,eye,[.4,.19],{headX:40}));
assert.deepEqual(cameraProject(wide,eye,[.4,.19],{headX:100}),cameraProject(wide,eye,[.4,.19],{headX:45}));
const {hairDepth}=await import('../web/parent-motion.js');
const hp={role:'hairFront',id:'h',image:'hair.png',rect:[0,0,1,1]},grid={cols:49,rows:73,values:Array.from({length:49*73},(_,i)=>(i%49)/48),image:'hair.png',rect:[0,0,1,1],sourceArray:'points.npy'};
const hproject={...cp,depth:{arrays:{'points.npy':'points.npy'}},parentMotion:{...cp.parentMotion,hairRelief:1,hairSurfaces:{h:grid}}};
assert.equal(hairDepth(hproject,hp,0,.5),0);assert.equal(hairDepth(hproject,hp,1,.5),1);
assert.equal(hairDepth({...hproject,parentMotion:{...hproject.parentMotion,hairRelief:0}},hp,0,.5),.35);
assert.deepEqual(cameraProject(hproject,hp,[.4,.19],{headX:20}),cameraProject({...hproject,parentMotion:{...hproject.parentMotion,relief:0}},hp,[.4,.19],{headX:20}));
assert.equal(hairDepth(hproject,{...hp,image:'changed.png'},0,.5),.35);
console.log('Per-hair depth interpolation, independent strength and stale asset fallback passed');
const {balancedLighting}=await import('../web/parent-motion.js');
assert.deepEqual([...balancedLighting(new Float64Array([-.8,-.8]),[1,1])],[0,0]);
const balanced=balancedLighting(new Float64Array([-.9,-.3,.5]),[1,.5,0]);
assert.ok(Math.abs(balanced[0]+balanced[1]*.5)<1e-10);assert.ok([...balanced].every(x=>Math.abs(x)<=.12000001));
const amplified={...hproject,parentMotion:{...hproject.parentMotion,hairRelief:3,relief:3}};
assert.notDeepEqual(cameraProject(amplified,hp,[.4,.19],{headX:20}),cameraProject(hproject,hp,[.4,.19],{headX:20}));
console.log('Global shading removal, local contrast bound and 3x relief passed');
