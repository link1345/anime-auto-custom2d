import assert from 'node:assert/strict';
import {blinkMaskGeometry} from '../web/deformation.js';
const part={id:'lashes',tag:'eyelash',role:'head',mesh:{vertices:[[.2,.2,0],[.8,.2,0],[.5,.8,0]],triangles:[[0,1,2]],morphs:{closed:[[0,.2,0],[0,.2,0],[0,-.3,0]]}}};
const project={parts:[part],eyeMask:{rect:[0,0,1,1]},expressions:[{id:'closed',target:'eyes',driver:'blink',fit:{partId:'lashes'}}]};
const g=blinkMaskGeometry(project,.5);
assert.ok(Math.abs(g.vertices[0][1]-.3)<1e-12);assert.ok(Math.abs(g.vertices[2][1]-.65)<1e-12);
assert.deepEqual(g.uv,part.mesh.vertices.map(v=>v.slice(0,2)));
project.expressions[0].stale=true;assert.equal(blinkMaskGeometry(project,.5),null);
console.log('Aperture follows upper and lower eyelid morph, with stale fallback');
