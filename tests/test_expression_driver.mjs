import assert from 'node:assert/strict';
import {drivenExpression,deformedVertices} from '../web/deformation.js';
const part={id:'lash',role:'head',mesh:{vertices:[[0,0,0],[1,0,0],[0,1,0]],morphs:{blink:[[0,.1,0],[0,.1,0],[0,-.1,0]]}}};
const p={expressions:[{id:'blink',fit:{partId:'lash'},driver:'blink'}]},frame={values:{eyeOpen:.25,mouthOpen:0}};
const d=drivenExpression(p,part,frame);assert.equal(d.strength,.75);
const v=deformedVertices(part,p,{...frame,expressionId:d.expression.id,strength:d.strength},null);assert.ok(Math.abs(v[0][1]-.075)<1e-12);
p.expressions[0].driver='mouth';assert.equal(drivenExpression(p,part,{values:{mouthOpen:.6}}).strength,.6);
p.expressions[0].stale=true;assert.equal(drivenExpression(p,part,frame),null);
console.log('Blink and mouth drivers, vertex interpolation and stale suppression passed');
