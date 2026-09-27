import {boneMatrices,deformedVertices,texturedMesh,partUV,headMatrix,isHeadPart,surfaceMesh,directionalMesh,multiviewMesh} from './deformation.js?v=perfgrid1';
import {defaults,follow,attachedPoint,attachmentMesh,createLightingDelta,balancedLighting,irisOffset,faceDepth} from './parent-motion.js?v=perfgrid1';
export class Renderer {
  constructor(canvas) {
    this.canvas=canvas; this.ctx=canvas.getContext('2d'); this.images=new Map();
    this.project=null; this.frame=null; this.view='model'; this.guides=false; this.mesh=false; this.selected='head';
    this.facePreview=false;this.facePreviewOpacity=.5;
  }
  url(path) {return `/assets/${this.project.id}/${path.split('/').pop()}`;}
  async load(project) {
    if(this.project?.id!==project.id){this.bodyState={x:0,y:0,z:0};this.bodyTime=undefined;}
    this.project=project;
    const paths=[project.source,...project.parts.map(p=>p.image),...project.expressions.map(e=>e.image),...(project.faceEffects||[]).filter(e=>e.image).map(e=>e.image)];
    if(project.faceInput)paths.push(project.faceInput.image);
    if(project.eyeMask)paths.push(project.eyeMask.image,project.eyeMask.irisImage);
    if(project.personMask)paths.push(project.personMask.image);
    await Promise.all(paths.map(async path=>{
      const url=this.url(path); if(this.images.has(url))return;
      const image=new Image(); image.src=url; await image.decode(); this.images.set(url,image);
    }));
  }
  image(path) {return this.images.get(this.url(path));}
  layout() {
    const [w,h]=this.project.size;
    const scale=Math.min(this.canvas.width*.76/w,this.canvas.height*.96/h);
    return {x:(this.canvas.width-w*scale)/2,y:(this.canvas.height-h*scale)/2,w:w*scale,h:h*scale};
  }
  modelPoint(event) {
    const rect=this.canvas.getBoundingClientRect();
    const contain=Math.min(rect.width/this.canvas.width,rect.height/this.canvas.height);
    const x=(event.clientX-rect.left-(rect.width-this.canvas.width*contain)/2)/contain;
    const y=(event.clientY-rect.top-(rect.height-this.canvas.height*contain)/2)/contain;
    const l=this.layout(); return [(x-l.x)/l.w,(y-l.y)/l.h];
  }
  headTransform(ctx,values) {
    ctx.transform(...headMatrix(this.project,values));
  }
  draw(frame=this.frame) {
    this.frame=frame; let ctx=this.ctx; const p=this.project;
    ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    if(!p||!frame)return;
    const l=this.layout(), v={...frame.values};
    if(this.eyePreview!==undefined)v.eyeOpen=this.eyePreview;
    if(this.directionPreview!==undefined)v.headX=this.directionPreview;
    ctx.save(); ctx.translate(l.x,l.y); ctx.scale(l.w,l.h);
    const source=this.image(p.source);
    if(this.view==='source') {if(source)ctx.drawImage(source,0,0,1,1);ctx.restore();return;}
    const matrices=boneMatrices(p,v);
    const parent=p.parentMotion?.enabled,settings={...defaults,...p.parentMotion};
    this.bodyState??={x:0,y:0,z:0};const dt=this.bodyTime===undefined?0:Math.max(0,frame.time-this.bodyTime);this.bodyTime=frame.time;
    if(parent)for(const [key,param] of [['x','headX'],['y','headY'],['z','headZ']])this.bodyState[key]=follow(this.bodyState[key],(v[param]||0)*settings.body,dt,settings.lag);
    let surface=parent?null:multiviewMesh(p,v)||directionalMesh(p,v)||(p.headSurface?.enabled?surfaceMesh(p.headSurface,v):null);
    const attachedMeshes=new Map();
    const attachedSurface=part=>{const key=part.role.startsWith('hair')?'hair:'+part.id:['irides','eyewhite','eyelash','eyebrow'].includes(part.tag)?'eyes':part.tag==='mouth'?'mouth':'face';if(!attachedMeshes.has(key))attachedMeshes.set(key,attachmentMesh(p,part,v,this.bodyState));return attachedMeshes.get(key);};
    const beginSurface=()=>{this.surfaceCanvas??=document.createElement('canvas');const layer=this.surfaceCanvas;if(layer.width!==1024){layer.width=1024;layer.height=1536;}const c=layer.getContext('2d');c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,layer.width,layer.height);c.scale(layer.width,layer.height);return c;};
    const finishSurface=output=>texturedMesh(output,this.surfaceCanvas,surface.vertices,surface.triangles,surface.uv);
    for(const part of [...p.parts].sort((a,b)=>a.depth-b.depth)) {
      if(part.visible===false||(p.display?.templatesVisible===false&&part.provenance==='template'))continue;
      let image=this.image(part.image); if(!image)continue;
      if(p.eyeMask&&part.scope==='face'&&['irides','eyewhite','eyelash'].includes(part.tag))image=this.maskEyeLayer(image,part.tag,v.eyeOpen,v);
      if(parent)surface=isHeadPart(part)?attachedSurface(part):null;
      const output=ctx,shared=surface&&isHeadPart(part);
      if(shared)ctx=beginSurface();
      ctx.save();
      if(isHeadPart(part)&&!shared)this.headTransform(ctx,v);
      ctx.translate(...part.pivot);
      if(!matrices&&part.role.startsWith('arm'))ctx.rotate((v[part.role]||0)*Math.PI/180);
      if(part.physics.enabled&&!isHeadPart(part))ctx.rotate(frame.hair[part.id]||frame.hair[part.role]||0);
      ctx.translate(-part.pivot[0],-part.pivot[1]);
      if(!p.eyeMask&&part.scope==='face'&&['irides','eyewhite','eyelash'].includes(part.tag)){
        ctx.beginPath();for(const [x,y] of [p.face.leftEye,p.face.rightEye]){ctx.moveTo(x+.05,y);ctx.ellipse(x,y,.05,.035*Math.max(.001,v.eyeOpen),0,0,Math.PI*2);}ctx.clip();
      }
      const fitted=p.expressions.find(e=>e.id===frame.expressionId&&!e.stale&&e.fit?.partId===part.id);
      const deform=isHeadPart(part)||!!matrices||!!fitted||part.mesh.source==='moge3';
      let vertices=deform?deformedVertices(part,p,frame,matrices):part.mesh.vertices;
      if(parent&&!isHeadPart(part))vertices=vertices.map((q,i)=>attachedPoint(p,part,q,v,this.bodyState,part.mesh.vertices[i]));
      if(deform||parent){
        texturedMesh(ctx,image,vertices,part.mesh.triangles,partUV(part));
        if(fitted&&frame.strength>0){ctx.save();ctx.globalAlpha=frame.strength;texturedMesh(ctx,this.image(fitted.image),vertices,part.mesh.triangles,fitted.fit.uv);ctx.restore();}
      }else ctx.drawImage(image,...(part.rect||[0,0,1,1]));
      if(this.mesh&&!shared) {
        ctx.strokeStyle='#9be4da88';ctx.lineWidth=.0008;
        const mesh=part.mesh;
        for(const triangle of mesh.triangles) {
          ctx.beginPath();triangle.forEach((index,i)=>{const q=vertices[index]; i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.closePath();ctx.stroke();
        }
      }
      ctx.restore();
      if(parent&&shared&&['skin','face'].includes(part.tag))this.shadeSkin(ctx,p,v,settings,part);
      if(shared){ctx=output;finishSurface(ctx);}
    }
    if(parent)surface=attachedSurface({role:'head',scope:'face'});
    const overlayOutput=ctx;if(surface)ctx=beginSurface();
    ctx.save();if(!surface)this.headTransform(ctx,v);
    // Temporary facial overlays are explicitly marked as templates in the editor.
    // The pupil is covered by an eyelid, never vertically scaled with the eye.
    const face=p.face;
    if(face&&!p.segmentation?.face) {
      for(const center of [face.leftEye,face.rightEye]) {
        const closure=1-v.eyeOpen;
        if(closure>.05) {
          ctx.fillStyle=face.skin;ctx.beginPath();ctx.ellipse(center[0],center[1]-.014+closure*.014,.027,.025*closure,0,0,Math.PI*2);ctx.fill();
          ctx.strokeStyle='#65463e';ctx.lineWidth=.0028;ctx.beginPath();ctx.moveTo(center[0]-.025,center[1]+.01*closure);ctx.quadraticCurveTo(center[0],center[1]+.024*closure,center[0]+.025,center[1]+.01*closure);ctx.stroke();
        }
      }
      if(v.mouthOpen>.04||v.smile>.05) {
        const [x,y]=face.mouth;
        ctx.fillStyle=face.skin;ctx.beginPath();ctx.ellipse(x,y,.032,.01+v.mouthOpen*.009,0,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#713e43';ctx.beginPath();ctx.ellipse(x,y+.002,.021+v.smile*.006,.001+v.mouthOpen*.018,0,0,Math.PI*2);ctx.fill();
        if(v.mouthOpen>.3){ctx.fillStyle='#e99a9b';ctx.beginPath();ctx.ellipse(x,y+v.mouthOpen*.012,.012,v.mouthOpen*.004,0,0,Math.PI*2);ctx.fill();}
      }
    }
    this.drawEffects(ctx,p,{...frame,values:v});
    if(this.facePreview&&p.faceInput){const image=this.image(p.faceInput.image);if(image){ctx.save();ctx.globalAlpha=this.facePreviewOpacity;ctx.drawImage(image,...p.faceInput.alignment.rect);ctx.restore();}}
    const expression=p.expressions.find(e=>e.id===frame.expressionId&&!e.stale);
    if(expression&&!expression.fit&&frame.strength>0) {
      const image=this.image(expression.image);if(image){
        ctx.save();ctx.beginPath();
        if(expression.target==='mouth')ctx.ellipse(...face.mouth,.046,.036,0,0,Math.PI*2);
        else if(expression.target==='eyes'){
          for(const [x,y] of [face.leftEye,face.rightEye]){ctx.moveTo(x+.045,y);ctx.ellipse(x,y,.045,.037,0,0,Math.PI*2);}
        }else if(expression.target==='brows'){
          for(const [x,y] of [face.leftEye,face.rightEye])ctx.rect(x-.047,y-.058,.094,.029);
        }else ctx.ellipse(.5,.195,.143,.101,0,0,Math.PI*2);
        ctx.clip();ctx.globalAlpha=frame.strength;ctx.drawImage(image,...expression.rect);ctx.restore();
      }
    }
    ctx.restore();
    if(surface){ctx=overlayOutput;finishSurface(ctx);}
    if(surface&&this.mesh){ctx.strokeStyle='#9be4da88';ctx.lineWidth=.0006;for(const tri of surface.triangles){if(tri.some(i=>surface.uv[i][1]>.48))continue;ctx.beginPath();tri.forEach((i,k)=>k?ctx.lineTo(...surface.vertices[i].slice(0,2)):ctx.moveTo(...surface.vertices[i].slice(0,2)));ctx.closePath();ctx.stroke();}}
    if(this.view==='compare'&&source) {
      ctx.save();ctx.beginPath();ctx.rect(0,0,.5,1);ctx.clip();ctx.drawImage(source,0,0,1,1);ctx.restore();
      ctx.strokeStyle='#c5ead7';ctx.lineWidth=.002;ctx.beginPath();ctx.moveTo(.5,0);ctx.lineTo(.5,1);ctx.stroke();
    }
    if(this.jointGuides&&p.landmarks){
      ctx.lineWidth=.002;ctx.font='.013px sans-serif';
      if(p.rig){
        ctx.strokeStyle='#89e5d5';
        for(const bone of p.rig.bones){if(bone.parent<0)continue;ctx.beginPath();ctx.moveTo(...p.rig.bones[bone.parent].position.slice(0,2));ctx.lineTo(...bone.position.slice(0,2));ctx.stroke();}
        ctx.strokeStyle='#edb27888';
        for(const collider of p.rig.colliders){ctx.beginPath();ctx.arc(...collider.center,collider.radius,0,Math.PI*2);ctx.stroke();}
      }
      for(const [name,joint] of Object.entries(p.landmarks.joints||{})){const [x,y]=joint.position;ctx.fillStyle=joint.source==='manual'?'#ffcd78':'#78eaca';ctx.beginPath();ctx.arc(x,y,.008,0,Math.PI*2);ctx.fill();ctx.fillText(name,x+.01,y);}
    }
    if(this.guides) {
      const part=p.parts.find(x=>x.id===this.selected);
      if(part) {
        ctx.strokeStyle='#baf5ce';ctx.fillStyle='#baf5ce';ctx.lineWidth=.002;
        ctx.beginPath();part.polygon.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();
        for(const [x,y]of part.polygon){ctx.beginPath();ctx.ellipse(x,y,.006,.004,0,0,Math.PI*2);ctx.fill();}
        ctx.strokeStyle='#ffc17d';ctx.beginPath();ctx.ellipse(...part.pivot,.016,.010,0,0,Math.PI*2);ctx.stroke();
      }
    }
    if(this.splitGuide){const {axis,position}=this.splitGuide;ctx.strokeStyle='#ffca79';ctx.lineWidth=.002;ctx.setLineDash([.01,.005]);ctx.beginPath();ctx.moveTo(axis==='x'?position:0,axis==='x'?0:position);ctx.lineTo(axis==='x'?position:1,axis==='x'?1:position);ctx.stroke();ctx.setLineDash([]);}
    ctx.restore();
  }

  shadeSkin(ctx,p,v,s,part){
    if((!s.light&&!s.shadow)||Math.hypot(v.headX||0,v.headY||0,v.headZ||0)<1e-6)return;
    const r=p.faceInput?.alignment?.rect??[.25,.01,.5,.35],coef=p.parentMotion.surface?.coefficients??[.7,0,0,-1.2,0,-1.2];
    this.lightCanvas??=document.createElement('canvas');const canvas=this.lightCanvas;
    if(canvas.width!==128||canvas.height!==128){canvas.width=128;canvas.height=128;}
    const c=canvas.getContext('2d');this.lightPixels??=c.createImageData(128,128);const pixels=this.lightPixels;
    // Read the static source alpha once, never the GPU-rendered frame.
    const key=JSON.stringify([p.id,part.image,part.rect,r]);
    this.skinWeights??=new Map();let weights=this.skinWeights.get(key);
    if(!weights){
      const mask=document.createElement('canvas');mask.width=mask.height=128;
      const mc=mask.getContext('2d',{willReadFrequently:true}),pr=part.rect??[0,0,1,1];
      mc.drawImage(this.image(part.image),(pr[0]-r[0])/r[2]*128,(pr[1]-r[1])/r[3]*128,pr[2]/r[2]*128,pr[3]/r[3]*128);
      const skin=mc.getImageData(0,0,128,128).data;weights=new Float64Array(128*128);
      for(let n=0;n<weights.length;n++)weights[n]=skin[n*4+3]/255;
      if(this.skinWeights.size>16)this.skinWeights.clear();this.skinWeights.set(key,weights);
    }
    this.lightDeltas??=new Float64Array(128*128);const deltas=this.lightDeltas,deltaAt=createLightingDelta(v);
    const slopeKey=JSON.stringify([p.id,r,s.relief,s.faceTilt,coef]);
    if(this.lightSlopeKey!==slopeKey||this.lightSurface!==p.parentMotion.surface){
      this.lightSlopes=new Float64Array(128*128*2);const step=1/127;
      for(let j=0;j<128;j++)for(let i=0;i<128;i++){
        const u=r[0]+i*step*r[2],v=r[1]+j*step*r[3],k=(j*128+i)*2;
        this.lightSlopes[k]=(faceDepth(p,u+step*r[2],v)-faceDepth(p,u-step*r[2],v))/(2*step)*.35;
        this.lightSlopes[k+1]=(faceDepth(p,u,v+step*r[3])-faceDepth(p,u,v-step*r[3]))/(2*step)*.35;
      }
      this.lightSlopeKey=slopeKey;this.lightSurface=p.parentMotion.surface;
    }
    for(let n=0;n<deltas.length;n++)deltas[n]=deltaAt(this.lightSlopes[n*2],this.lightSlopes[n*2+1]);
    const balanced=balancedLighting(deltas,weights);
    for(let n=0;n<balanced.length;n++){const delta=balanced[n],k=n*4;pixels.data[k]=pixels.data[k+1]=pixels.data[k+2]=delta>0?255:0;pixels.data[k+3]=Math.round(Math.abs(delta)*s.light*255);}
    c.putImageData(pixels,0,0);ctx.save();ctx.globalCompositeOperation='source-atop';ctx.drawImage(canvas,...r);
    if(s.shadow){
      const hair=p.parts.filter(part=>part.visible!==false&&part.role==='hairFront'&&part.tag!=='back hair');
      const shadowKey=JSON.stringify([p.id,hair.map(part=>[part.image,part.rect])]);
      this.shadowCanvas??=document.createElement('canvas');const shadow=this.shadowCanvas;
      if(this.shadowKey!==shadowKey){
        shadow.width=512;shadow.height=768;const sc=shadow.getContext('2d');sc.scale(512,768);
        for(const part of hair){const im=this.image(part.image);if(im)sc.drawImage(im,...(part.rect||[0,0,1,1]));}
        sc.globalCompositeOperation='source-in';sc.fillStyle='#312635';sc.fillRect(0,0,1,1);this.shadowKey=shadowKey;
      }
      ctx.globalAlpha=s.shadow*Math.min(1,Math.hypot(v.headX||0,v.headY||0,v.headZ||0)/20);ctx.drawImage(shadow,.004+(v.headX||0)*.00015,.005,1,1);
    }
    ctx.restore();
  }

  maskEyeLayer(image,tag,open,values=this.frame?.values??{}){
    const m=this.project.eyeMask,mask=this.image(m.image);if(!mask||!image)return image;
    open=Math.max(0,Math.min(1,Math.round(open*30)/30));const scale=m.scale??1;
    this.eyeCache??=new Map();const key=this.url(m.image)+':'+tag+':'+image.src;
    const settings={...defaults,...this.project.parentMotion};
    const gazeKey=tag==='irides'?JSON.stringify(m.eyes.map(eye=>irisOffset(eye,settings,values).map(x=>Math.round(x*mask.width)))):tag==='eyewhite'?String(settings.gazeEnabled):'';
    const cached=this.eyeCache.get(key);if(cached?.open===open&&cached.scale===scale&&cached.source===image&&cached.gazeKey===gazeKey)return cached.canvas;
    const raw=document.createElement('canvas');raw.width=mask.width;raw.height=mask.height;
    const c=raw.getContext('2d'),w=raw.width,h=raw.height;
    if(tag==='eyelash'){
      for(let i=0;i<2;i++){
        const [l,t,r,b]=m.eyes[i].bbox,mid=(t+b)/2*h,factor=.18+.82*open;
        c.save();c.beginPath();c.rect(i*w/2,0,w/2,h);c.clip();c.translate(0,mid*(1-factor));c.scale(1,factor);c.drawImage(image,0,0,w,h);c.restore();
      }
    }else{
      if(tag==='irides'){
        for(let i=0;i<2;i++){const [dx,dy]=irisOffset(m.eyes[i],settings,values);c.save();c.beginPath();c.rect(i*w/2,0,w/2,h);c.clip();c.drawImage(image,i*w/2,0,w/2,h,i*w/2+Math.round(dx*w),Math.round(dy*h),w/2,h);c.restore();}
      }else{
        // Fill the original iris hole before moving the independent iris.
        if(settings.gazeEnabled){c.drawImage(mask,0,0,w,h);c.globalCompositeOperation='source-in';c.fillStyle='#fff8f3';c.fillRect(0,0,w,h);c.globalCompositeOperation='source-over';}
        c.drawImage(image,0,0,w,h);
      }
      c.globalCompositeOperation='destination-in';c.drawImage(mask,0,0);
      // Apply a second mask as a rectangle path. At zero openness this clears
      // both eyes completely, while the underlying skin remains visible.
      c.beginPath();
      for(const eye of m.eyes){const [l,t,r,b]=eye.bbox,mid=(t+b)/2;c.rect(l*w,(mid-(b-t)*open/2)*h,(r-l)*w,(b-t)*open*h);}
      c.fill();
    }
    const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const out=canvas.getContext('2d');
    for(let i=0;i<2;i++){
      const [l,t,r,b]=m.eyes[i].bbox,cx=(l+r)/2*w,cy=(t+b)/2*h;
      out.save();out.beginPath();out.rect(i*w/2,0,w/2,h);out.clip();out.translate(cx,cy);out.scale(scale,scale);out.translate(-cx,-cy);
      out.drawImage(raw,i*w/2,0,w/2,h,i*w/2,0,w/2,h);out.restore();
    }
    if(this.eyeCache.size>64)this.eyeCache.clear();
    this.eyeCache.set(key,{open,scale,source:image,canvas,gazeKey});return canvas;
  }

  drawEffects(ctx,project,frame){
    const f=project.face;
    for(const effect of project.faceEffects||[]){
      if(!effect.visible)continue;
      ctx.save();ctx.globalAlpha=effect.opacity;
      if(effect.kind==='image'){
        const image=this.image(effect.image);const [x,y,w,h]=effect.rect;
        if(image)ctx.drawImage(image,x+w*(1-effect.scale)/2,y+h*(1-effect.scale)/2,w*effect.scale,h*effect.scale);
      }else if(effect.kind==='blush'){
        for(const [x,y]of [f.leftEye,f.rightEye]){
          ctx.save();ctx.translate(x,y+.035);ctx.scale(.04*effect.scale,.014*effect.scale);
          const gradient=ctx.createRadialGradient(0,0,0,0,0,1);gradient.addColorStop(0,'#f27998');gradient.addColorStop(1,'#f2799800');ctx.fillStyle=gradient;ctx.beginPath();ctx.arc(0,0,1,0,Math.PI*2);ctx.fill();ctx.restore();
        }
      }else if(frame.values.eyeOpen>.1){
        const output=ctx,mask=project.eyeMask;let layer;
        if(mask){layer=document.createElement('canvas');layer.width=1024;layer.height=1536;ctx=layer.getContext('2d');ctx.scale(1024,1536);}
        ctx.fillStyle=effect.kind==='star'?'#fff4a4':'#fc81b3';
        for(const [x,y]of [f.leftEye,f.rightEye]){
          ctx.save();if(!mask){ctx.beginPath();ctx.ellipse(x,y,.02,.022*frame.values.eyeOpen,0,0,Math.PI*2);ctx.clip();}
          ctx.translate(x,y);ctx.scale(.012*effect.scale,.010*effect.scale);ctx.beginPath();
          if(effect.kind==='star')for(let i=0;i<10;i++){const a=i*Math.PI/5-Math.PI/2,r=i%2?.42:1;i?ctx.lineTo(Math.cos(a)*r,Math.sin(a)*r):ctx.moveTo(Math.cos(a)*r,Math.sin(a)*r);}
          else{ctx.moveTo(0,1);ctx.bezierCurveTo(-2,-.3,-.7,-1.7,0,-.6);ctx.bezierCurveTo(.7,-1.7,2,-.3,0,1);}
          ctx.closePath();ctx.fill();ctx.restore();
        }
        if(mask){ctx.globalCompositeOperation='destination-in';ctx.drawImage(this.maskEyeLayer(this.image(mask.irisImage),'irides',frame.values.eyeOpen),...mask.rect);ctx=output;ctx.drawImage(layer,0,0,1,1);}
      }
      ctx.restore();
    }
  }
}

export function connect(onFrame,onProject,onStatus,onResponse) {
  let socket, timer, closed=false, pendingFrame=null, animation=null;
  const render=()=>{animation=null;const frame=pendingFrame;pendingFrame=null;if(!closed&&frame)onFrame(frame);};
  function open(){
    socket=new WebSocket(`ws://${location.host}/ws`);
    socket.onopen=()=>onStatus?.(true);
    socket.onmessage=event=>{const packet=JSON.parse(event.data);if(packet.type==='frame'){pendingFrame=packet;if(animation===null)animation=requestAnimationFrame(render);}else if(packet.type==='projectChanged')onProject();else onResponse?.(packet);};
    socket.onclose=()=>{onStatus?.(false);if(!closed)timer=setTimeout(open,1200);};
    socket.onerror=()=>socket.close();
  }
  open();
  return {send(message){if(socket.readyState===WebSocket.OPEN)socket.send(JSON.stringify({id:crypto.randomUUID(),...message}));},close(){closed=true;clearTimeout(timer);if(animation!==null)cancelAnimationFrame(animation);pendingFrame=null;socket.close();}};
}
