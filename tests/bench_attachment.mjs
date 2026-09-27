import {attachmentMesh} from '../web/parent-motion.js';
const p=await (await fetch('http://127.0.0.1:8765/api/project')).json();
const parts=p.parts.filter(p=>p.visible!==false&&['face','irides','mouth','front hair','back hair'].includes(p.tag));
for(let i=0;i<3;i++)for(const part of parts)attachmentMesh(p,part,{headY:1},{x:0,y:.1,z:0});
const start=performance.now();for(let i=0;i<20;i++)for(const part of parts)attachmentMesh(p,part,{headY:Math.sin(i)},{x:0,y:.1,z:0});
console.log(JSON.stringify({groups:parts.length,msPerFrame:(performance.now()-start)/20}));
