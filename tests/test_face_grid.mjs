import assert from 'node:assert/strict';
import {defaults,faceDepth,cameraProject} from '../web/parent-motion.js';
const g={cols:128,rows:128,plane:[.5,.8,-.4],residual:Array(16384).fill(.1),confidence:Array(16384).fill(1),sourceAsset:'face',sourceArray:'depth',rect:[0,0,1,1]};
const p={size:[1024,1536],faceInput:{image:'face',alignment:{rect:g.rect}},faceDepths:{front:{arrays:{'points.npy':'depth'}}},parentMotion:{...defaults,camera:2,surface:{grid:g}}};
assert.equal(faceDepth(p,.5,.55),.6);
assert.equal(faceDepth({...p,parentMotion:{...p.parentMotion,relief:0}},.2,.2),.5);
const tilted={...p,parentMotion:{...p.parentMotion,faceTilt:1}};
assert.ok(faceDepth(tilted,.8,.55)>faceDepth(p,.8,.55));
const amplified={...p,parentMotion:{...p.parentMotion,relief:3}};
assert.equal(faceDepth(amplified,.5,.55),.8);
for(const camera of [1,2]){const q=cameraProject({...p,parentMotion:{...p.parentMotion,camera}},{role:'head'},[.6,.2],{});assert.ok(Math.abs(q[0]-.6)<1e-12&&Math.abs(q[1]-.2)<1e-12);}
console.log('Depth-grid sampling, independent tilt/relief and neutral camera alignment passed');
