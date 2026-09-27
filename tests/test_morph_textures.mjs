import assert from 'node:assert/strict';
import {drawMorphTextures} from '../web/renderer.js';
function render(t,target='closed'){
 const calls=[],stack=[],ctx={globalAlpha:.8,save(){stack.push(this.globalAlpha);},restore(){this.globalAlpha=stack.pop();}};
 drawMorphTextures(ctx,'open',target,t,(image,isTarget)=>calls.push([image,ctx.globalAlpha,isTarget]));
 assert.equal(ctx.globalAlpha,.8);return calls;
}
assert.deepEqual(render(1),[['closed',.8,true]],'Fully closed must never render original lashes');
assert.deepEqual(render(0),[['open',.8,false]]);
assert.deepEqual(render(.5),[['open',.4,false],['closed',.4,true]]);
assert.deepEqual(render(1,null),[['open',.8,false]],'Missing texture keeps original visible');
assert.deepEqual(render(2),render(1));
console.log('Closed-eye endpoint excludes source texture; open, intermediate and missing-target cases passed');
