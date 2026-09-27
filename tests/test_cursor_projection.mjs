import assert from 'node:assert/strict';
import {defaults,cursorAngles,cameraProject,cursorGaze} from '../web/parent-motion.js';
const part={role:'head'},point=[.6,.2],v={headX:15,headY:5},p={size:[1024,1536],parentMotion:{...defaults}};
for(const camera of [1,2])for(const cursorMode of [0,1]){
 const a={...p,parentMotion:{...defaults,camera,cursorMode}};
 const q=cameraProject(a,part,point,{});assert.ok(Math.abs(q[0]-point[0])<1e-12&&Math.abs(q[1]-point[1])<1e-12);
 assert.deepEqual(cameraProject(a,part,point,v),cameraProject({...a,parentMotion:{...a.parentMotion,cursorMode:1-cursorMode}},part,point,v));
 assert.ok(cursorAngles(a,[.9,.3]).headX>0);assert.ok(cursorAngles(a,[.1,.3]).headX<0);
}
const ortho={...p,parentMotion:{...defaults,camera:2,cursorMode:1}};
assert.deepEqual(cameraProject(ortho,part,point,v).slice(0,2),cameraProject({...ortho,parentMotion:{...ortho.parentMotion,cameraDistance:0}},part,point,v).slice(0,2));
assert.notDeepEqual(cursorAngles(ortho,[.7,.3]),cursorAngles({...ortho,parentMotion:{...ortho.parentMotion,targetDepth:.9}},[.7,.3]));
assert.ok(cursorGaze(ortho,[.9,.3]).gazeX>0);
console.log('All camera/cursor combinations, neutral alignment, orthographic distance invariance and cursor depth passed');
