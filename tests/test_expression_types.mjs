import assert from 'node:assert/strict';
import {expressionMatchesPart,drivenExpression,deformedVertices} from '../web/deformation.js';
const part={id:'lashes',tag:'eyelash',role:'head',mesh:{vertices:[[.5,.2,0]],morphs:{bad:[[0,.5,0]]}}};
const bad={id:'bad',target:'mouth',driver:'blink',fit:{partId:'lashes'}};
assert.equal(expressionMatchesPart(bad,part),false);
assert.equal(drivenExpression({expressions:[bad]},part,{values:{eyeOpen:0}}),null);
assert.deepEqual(deformedVertices(part,{expressions:[bad]},{expressionId:'bad',strength:1},null),part.mesh.vertices);
assert.equal(expressionMatchesPart({target:'eyes'},part),true);
console.log('Mouth-to-eye assignment and legacy corrupt morph are rejected');
