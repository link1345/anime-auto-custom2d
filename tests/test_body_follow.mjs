import assert from 'node:assert/strict';
import {bodyAnchors,bodyPoint,attachedPoint,defaults} from '../web/parent-motion.js';
const p={size:[1024,1536],parentMotion:{...defaults},landmarks:{joints:{neck:{position:[.509,.326]},root:{position:[.5,.4]},hipLeft:{position:[.4,.669]},hipRight:{position:[.6,.667]}}}};
const before=JSON.stringify(p.landmarks),state={x:9,y:3,z:0};
assert.deepEqual(bodyAnchors(p).waist,[.5,.668]);
for(const y of [.38,.45,.55,.62])assert.ok(bodyPoint(p,[.5,y],state)[0]>.5,`torso at ${y} follows`);
assert.deepEqual(bodyPoint(p,[.5,.7],state),[.5,.7]);
assert.deepEqual(bodyPoint(p,[.5,.5],{x:0,y:0,z:0}),[.5,.5]);
for(const camera of [0,1,2])for(const cursorMode of [0,1]){
 const project={...p,parentMotion:{...defaults,camera,cursorMode}};
 assert.ok(attachedPoint(project,{role:'chest'},[.5,.5,0],{},state)[0]>.5);
}
assert.equal(JSON.stringify(p.landmarks),before);
assert.equal(bodyAnchors({landmarks:{joints:{neck:p.landmarks.joints.neck,root:p.landmarks.joints.root}}}).waist[1],.59);
console.log('Chest-level root regression, torso following, fixed lower body, all projection/cursor modes and unchanged landmarks passed');
