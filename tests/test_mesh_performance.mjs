import assert from 'node:assert/strict';
import {defaults,attachmentMesh,attachedPoint} from '../web/parent-motion.js';
const p={size:[1024,1536],parentMotion:{...defaults}},state={x:2,y:1,z:1},values={headX:12,headY:4,headZ:2};
for(const camera of [0,1,2])for(const part of [{role:'head',tag:'face'},{role:'head',tag:'irides'},{role:'hairFront'}]){
 p.parentMotion.camera=camera;
 const mesh=attachmentMesh(p,part,values,state);
 for(const i of [0,100,1500,3000,3576])assert.deepEqual(mesh.vertices[i],attachedPoint(p,part,mesh.uv[i],values,state));
 const second=attachmentMesh(p,part,{...values,headY:8,headX:16},state);
 assert.equal(mesh.triangles,second.triangles);assert.equal(mesh.uv,second.uv);
 assert.notDeepEqual(mesh.vertices,second.vertices);
}
p.parentMotion.relief=3;const part={role:'head',tag:'face'},m=attachmentMesh(p,part,values,state);
assert.deepEqual(m.vertices[1500],attachedPoint(p,part,m.uv[1500],values,state));
console.log('Optimized mesh matches independent point evaluation; topology reused; pose and setting updates preserved');

