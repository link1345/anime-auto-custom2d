import assert from 'node:assert/strict';
import {morphTextureWeight,deformedVertices} from '../web/deformation.js';
import {drawMorphTextures} from '../web/renderer.js';
for(const driver of ['blink','mouth']){
 const expression={id:'target',driver,fit:{partId:'p'}},project={expressions:[expression]};
 const part={id:'p',role:'head',mesh:{vertices:[[0,0,0],[1,0,0],[0,1,0]],morphs:{target:[[0,0,0],[0,0,0],[0,-.8,0]]}}};
 const calls=[],ctx={globalAlpha:1,save(){},restore(){}};
 const vertices=deformedVertices(part,project,{expressionId:'target',strength:.5},null);
 assert.equal(vertices[2][1],.6,'Geometry still interpolates at 50%');
 drawMorphTextures(ctx,'base','target',morphTextureWeight(expression,.5),(image,isTarget)=>calls.push(image));
 assert.deepEqual(calls,driver==='blink'?['base']:['target'],'Only one texture during middle of deformation');
 assert.equal(morphTextureWeight(expression,0),0);assert.equal(morphTextureWeight(expression,1),1);
 let prev=-1;for(let i=0;i<=100;i++){const w=morphTextureWeight(expression,i/100);assert.ok(w>=prev);prev=w;}
}
assert.equal(morphTextureWeight({driver:'blink'},.8),0);
assert.equal(morphTextureWeight({driver:'mouth'},.2),1);
assert.equal(morphTextureWeight({},.5),.5,'Normal expressions keep their crossfade');
console.log('Shape interpolation and closure-only texture transitions passed');
