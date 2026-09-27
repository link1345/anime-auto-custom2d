import assert from 'node:assert/strict';
import {Renderer} from '../web/renderer.js';
import {createLightingDelta} from '../web/parent-motion.js';
// Independent reference rotation: optimization must preserve the lighting signal.
function reference(dx,dy,v){
 const n=Math.hypot(dx,dy,1),a=v.headX*Math.PI/-180,b=v.headY*Math.PI/180,c=v.headZ*Math.PI/180;
 const x=dx/n*Math.cos(a)+1/n*Math.sin(a),z=-dx/n*Math.sin(a)+1/n*Math.cos(a);
 const y=dy/n*Math.cos(b)-z*Math.sin(b),zz=dy/n*Math.sin(b)+z*Math.cos(b);
 const clamp=v=>Math.max(0,Math.min(1,v));
 return clamp(-.4*(x*Math.cos(c)-y*Math.sin(c))-.35*(x*Math.sin(c)+y*Math.cos(c))+.85*zz)-clamp((-.4*dx-.35*dy+.85)/n);
}
for(const headX of [-45,0,45])for(const headY of [-30,0,30])for(const headZ of [-12,0,12]){
 const v={headX,headY,headZ},fn=createLightingDelta(v);
 for(const dx of [-3,0,3])for(const dy of [-3,0,3])assert.ok(Math.abs(fn(dx,dy)-reference(dx,dy,v))<1e-12);
}
let reads=0,shadowDraws=0;
globalThis.document={createElement(){const canvas={width:300,height:150};canvas.getContext=()=>({
 createImageData:()=>({data:new Uint8ClampedArray(128*128*4)}),
 drawImage(image){assert.notEqual(image,'live GPU surface');if(image==='hair')shadowDraws++;},
 getImageData(){reads++;return {data:new Uint8ClampedArray(128*128*4).fill(255)};},
 putImageData(){},scale(){},fillRect(){}
});return canvas;}};
const renderer={image:path=>path},part={image:'skin',rect:[0,0,1,1]},p={id:'p',parentMotion:{},parts:[{role:'hairFront',image:'hair'}]};
const ctx={canvas:'live GPU surface',save(){},restore(){},drawImage(){}};
const settings={light:1,shadow:1,relief:3};
for(let i=0;i<10;i++)Renderer.prototype.shadeSkin.call(renderer,ctx,p,{headX:i+1},settings,part);
assert.equal(reads,1,'alpha read only on first frame');assert.equal(shadowDraws,1,'shadow silhouette cached');
part.rect=[.1,0,.9,1];Renderer.prototype.shadeSkin.call(renderer,ctx,p,{headX:10},settings,part);assert.equal(reads,2,'rect change invalidates alpha');
p.parts[0].visible=false;Renderer.prototype.shadeSkin.call(renderer,ctx,p,{headX:10},settings,part);assert.equal(shadowDraws,1,'hidden hair not redrawn');
console.log('Lighting equivalence, no live GPU readback, alpha/shadow reuse and invalidation passed');
// Incoming simulation frames are coalesced; command responses are never dropped.
const {connect}=await import('../web/renderer.js');
let socket,raf,drawn=[],responses=[];
globalThis.location={host:'test'};globalThis.WebSocket=class {constructor(){socket=this;}close(){} };
globalThis.requestAnimationFrame=fn=>{raf=fn;return 1;};globalThis.cancelAnimationFrame=()=>{raf=null;};
const connection=connect(f=>drawn.push(f),()=>{},()=>{},p=>responses.push(p));
for(let i=0;i<10;i++)socket.onmessage({data:JSON.stringify({type:'frame',time:i})});
socket.onmessage({data:JSON.stringify({ok:true,id:'command'})});
assert.equal(drawn.length,0);raf();assert.equal(drawn.length,1);assert.equal(drawn[0].time,9);assert.equal(responses.length,1);
socket.onmessage({data:JSON.stringify({type:'frame',time:10})});connection.close();assert.equal(raf,null);
console.log('Frame coalescing, command delivery and close cancellation passed');
