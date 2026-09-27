// One parent transform for every facial layer, mask, decoration and hair root.
// Z remains asset geometry; it is deliberately not a facial motion parameter.
export const isHeadPart=part=>part.role==='head'||part.scope==='face'||part.role.startsWith('hair');
export function headMatrix(project,values){
  const pivot=project.landmarks?.joints?.neck?.position??[.5,.315];
  const aspect=(project.size?.[0]??1024)/(project.size?.[1]??1536);
  const yaw=Math.max(-20,Math.min(20,values.headX||0))*Math.PI/180;
  const pitch=Math.max(-15,Math.min(15,values.headY||0))*Math.PI/180;
  const roll=Math.max(-12,Math.min(12,values.headZ||0))*Math.PI/180;
  const c=Math.cos(roll),s=Math.sin(roll),sx=Math.cos(yaw),sy=Math.cos(pitch);
  const a=c*sx,b=s*sx*aspect,d=c*sy,cc=-s*sy/aspect;
  const dx=.035*Math.sin(yaw),dy=.025*Math.sin(pitch);
  return [a,b,cc,d,pivot[0]+dx-a*pivot[0]-cc*pivot[1],pivot[1]+dy-b*pivot[0]-d*pivot[1]];
}

export function boneMatrices(project,values){
  if(!project.rig||project.rig.stale)return null;
  const bones=project.rig.bones,world=[];
  const angles={head:values.headZ||0,shoulderLeft:values.armLeft||0,shoulderRight:values.armRight||0,
    elbowLeft:values.elbowLeft||0,elbowRight:values.elbowRight||0,wristLeft:values.wristLeft||0,wristRight:values.wristRight||0};
  for(const bone of bones){
    const parent=bone.parent>=0?world[bone.parent]:[1,0,0,1,0,0];
    const rest=bone.parent>=0?bones[bone.parent].position:[0,0,0];
    const x=bone.position[0]-rest[0],y=bone.position[1]-rest[1],angle=(angles[bone.name]||0)*Math.PI/180;
    const c=Math.cos(angle),s=Math.sin(angle);
    world.push([parent[0]*c+parent[2]*s,parent[1]*c+parent[3]*s,parent[2]*c-parent[0]*s,parent[3]*c-parent[1]*s,
      parent[0]*x+parent[2]*y+parent[4],parent[1]*x+parent[3]*y+parent[5]]);
  }
  return world.map((m,i)=>{const [x,y]=bones[i].position;return [...m.slice(0,4),m[4]-m[0]*x-m[2]*y,m[5]-m[1]*x-m[3]*y];});
}

export function expressionMatchesPart(e,part){
 if(e.tag)return e.tag===part.tag;
 if(e.target==='mouth')return part.tag==='mouth';
 if(e.target==='eyes')return ['eyelash','eyewhite','irides'].includes(part.tag);
 if(e.target==='brows')return part.tag==='eyebrow';
 return true;
}
// Shape interpolation is independent of the short texture handoff near closure.
export function morphTextureWeight(expression,strength){
 const t=Math.max(0,Math.min(1,strength||0));
 const smooth=(x,a,b)=>{const v=Math.max(0,Math.min(1,(x-a)/(b-a)));return v*v*(3-2*v);};
 if(expression?.driver==='blink')return smooth(t,.8,1);
 if(expression?.driver==='mouth')return smooth(t,0,.2);
 return t;
}
export function drivenExpression(project,part,frame){
 const e=project.expressions.find(e=>!e.stale&&e.fit?.partId===part.id&&expressionMatchesPart(e,part)&&['blink','mouth'].includes(e.driver));
 if(!e)return null;
 const strength=Math.max(0,Math.min(1,e.driver==='blink'?1-(frame.values.eyeOpen??1):(frame.values.mouthOpen??0)));
 return {expression:e,strength};
}
export function deformedVertices(part,project,frame,matrices){
  const mesh=part.mesh,expression=project.expressions?.find(e=>e.id===frame.expressionId);
  const morph=expression&&(!expression.stale)&&expressionMatchesPart(expression,part)&&(!expression.fit||expression.fit.partId===part.id)?mesh.morphs?.[frame.expressionId]:undefined,strength=frame.strength||0;
  const head=isHeadPart(part),hair=part.role.startsWith('hair')&&part.physics?.enabled;
  const ys=hair?mesh.vertices.map(v=>v[1]):[0,1],top=Math.min(...ys),bottom=Math.max(...ys);
  const eyeY=project.face?((project.face.leftEye[1]+project.face.rightEye[1])/2):top+(bottom-top)*.4;
  const rootEnd=Math.max(top,Math.min(eyeY,top+(bottom-top)*.5));
  const angle=hair?Math.max(-.06,Math.min(.06,frame.hair?.[part.id]??frame.hair?.[part.role]??0)):0;
  const aspect=(project.size?.[0]??1024)/(project.size?.[1]??1536);
  return mesh.vertices.map((v,i)=>{
    let [x,y,z]=v;if(morph){x+=morph[i][0]*strength;y+=morph[i][1]*strength;z+=morph[i][2]*strength;}
    if(!head&&matrices&&mesh.weights){let xx=0,yy=0;for(let k=0;k<4;k++){const w=mesh.weights[i][k],m=matrices[mesh.joints[i][k]];xx+=w*(m[0]*x+m[2]*y+m[4]);yy+=w*(m[1]*x+m[3]*y+m[5]);}x=xx;y=yy;}
    if(hair&&angle&&v[1]>rootEnd){
      const t=Math.max(0,Math.min(1,(v[1]-rootEnd)/Math.max(bottom-rootEnd,1e-6)));
      const bend=angle*t*t*(3-2*t),c=Math.cos(bend),s=Math.sin(bend);
      const px=part.pivot[0],dx=(x-px)*aspect,dy=y-rootEnd;
      x=px+(c*dx-s*dy)/aspect;y=rootEnd+s*dx+c*dy;
    }
    return [x,y,z];
  });
}

export function texturedMesh(ctx,image,vertices,triangles,uv){
  if(typeof document!=='undefined'){
    meshGPU??=new MeshGPU();
    if(meshGPU.gl){ctx.drawImage(meshGPU.render(image,vertices,triangles,uv),0,0,1,1);return;}
  }
  // Expand only the clip edge by a subpixel to avoid Canvas antialias cracks between adjacent triangles.
  const transform=ctx.getTransform(),epsilon=.65/Math.max(Math.hypot(transform.a,transform.b),Math.hypot(transform.c,transform.d));
  for(const tri of triangles){
    const [i,j,k]=tri,[a,b,c]=[vertices[i],vertices[j],vertices[k]],s=[uv[i],uv[j],uv[k]].map(([u,v])=>[u*image.width,v*image.height]);
    const ux=s[1][0]-s[0][0],uy=s[1][1]-s[0][1],vx=s[2][0]-s[0][0],vy=s[2][1]-s[0][1],det=ux*vy-uy*vx;
    if(Math.abs(det)<1e-10)continue;
    const ax=b[0]-a[0],ay=b[1]-a[1],bx=c[0]-a[0],by=c[1]-a[1];
    const A=(ax*vy-bx*uy)/det,B=(ay*vy-by*uy)/det,C=(bx*ux-ax*vx)/det,D=(by*ux-ay*vx)/det;
    const center=[(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3];
    const expanded=[a,b,c].map(p=>{const dx=p[0]-center[0],dy=p[1]-center[1],scale=epsilon/Math.max(Math.hypot(dx,dy),1e-9);return [p[0]+dx*scale,p[1]+dy*scale];});
    ctx.save();ctx.beginPath();expanded.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.clip();
    ctx.transform(A,B,C,D,a[0]-A*s[0][0]-C*s[0][1],a[1]-B*s[0][0]-D*s[0][1]);ctx.drawImage(image,0,0);ctx.restore();
  }
}

export function partUV(part){const r=part.rect||[0,0,1,1];return part.mesh.vertices.map(v=>[(v[0]-r[0])/r[2],(v[1]-r[1])/r[3]]);}

let meshGPU;
class MeshGPU{
  constructor(){
    this.canvas=document.createElement('canvas');this.canvas.width=768;this.canvas.height=1152;
    const gl=this.canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:true,antialias:true,preserveDrawingBuffer:true});this.gl=gl;
    if(!gl)return;
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    this.program=gl.createProgram();
    gl.attachShader(this.program,shader(gl.VERTEX_SHADER,'#version 300 es\nin vec2 position;in vec2 texcoord;out vec2 uv;void main(){uv=texcoord;gl_Position=vec4(position.x*2.0-1.0,1.0-position.y*2.0,0.0,1.0);}'));
    gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision mediump float;in vec2 uv;uniform sampler2D image;out vec4 color;void main(){vec4 c=texture(image,uv);if(any(lessThan(uv,vec2(0.0)))||any(greaterThan(uv,vec2(1.0))))c=vec4(0.0);color=vec4(c.rgb*c.a,c.a);}'));
    gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
    this.position=gl.getAttribLocation(this.program,'position');this.texcoord=gl.getAttribLocation(this.program,'texcoord');
    this.vbo=gl.createBuffer();this.ibo=gl.createBuffer();this.textures=new Map();this.indexArrays=new WeakMap();
  }
  render(image,vertices,triangles,uv){
    const gl=this.gl;gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(this.program);
    let texture=this.textures.get(image);
    if(!texture){texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);this.textures.set(image,texture);
      if(this.textures.size>48){const old=this.textures.keys().next().value;gl.deleteTexture(this.textures.get(old));this.textures.delete(old);}
    }else {gl.bindTexture(gl.TEXTURE_2D,texture);if(image instanceof HTMLCanvasElement)gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,gl.RGBA,gl.UNSIGNED_BYTE,image);}
    if(this.vertexData?.length!==vertices.length*4)this.vertexData=new Float32Array(vertices.length*4);
    const data=this.vertexData;for(let i=0;i<vertices.length;i++){const k=i*4;data[k]=vertices[i][0];data[k+1]=vertices[i][1];data[k+2]=uv[i][0];data[k+3]=uv[i][1];}
    gl.bindBuffer(gl.ARRAY_BUFFER,this.vbo);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(this.position);gl.enableVertexAttribArray(this.texcoord);gl.vertexAttribPointer(this.position,2,gl.FLOAT,false,16,0);gl.vertexAttribPointer(this.texcoord,2,gl.FLOAT,false,16,8);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.ibo);
    if(this.lastTriangles!==triangles){
      let indices=this.indexArrays.get(triangles);if(!indices){indices=new Uint16Array(triangles.length*3);for(let i=0;i<triangles.length;i++)indices.set(triangles[i],i*3);this.indexArrays.set(triangles,indices);}
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW);this.lastTriangles=triangles;
    }
    gl.drawElements(gl.TRIANGLES,triangles.length*3,gl.UNSIGNED_SHORT,0);return this.canvas;
  }
}


export function surfaceMesh(surface,values){
  const {cols,rows,intrinsics:K,pivot}=surface,[pu,pv,pz]=pivot;
  const fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2];
  const px=(pu-cx)*pz/fx,py=(pv-cy)*pz/fy;
  const rad=Math.PI/180,limit=surface.maxAngle??15;
  // Camera Z grows away from the viewer. Positive screen-right headX must
  // move a nearer nose toward screen-right, matching the reference views.
  const yaw=-Math.max(-limit,Math.min(limit,values.headX||0))*rad;
  const pitch=Math.max(-limit,Math.min(limit,values.headY||0))*rad;
  const roll=(values.headZ||0)*rad;
  const vertices=[],uv=[],triangles=[];
  for(let j=0;j<rows;j++)for(let i=0;i<cols;i++){
    const u=i/(cols-1),v=j/(rows-1),z=pz+(surface.depth[j*cols+i]-pz)*(surface.strength??1);
    let x=(u-cx)*z/fx-px,y=(v-cy)*z/fy-py,zz=z-pz;
    const xx=Math.cos(yaw)*x+Math.sin(yaw)*zz,zx=-Math.sin(yaw)*x+Math.cos(yaw)*zz;
    const yy=Math.cos(pitch)*y-Math.sin(pitch)*zx,zy=Math.sin(pitch)*y+Math.cos(pitch)*zx;
    x=Math.cos(roll)*xx-Math.sin(roll)*yy;y=Math.sin(roll)*xx+Math.cos(roll)*yy;
    const cameraZ=Math.max(pz*.1,zy+pz);
    vertices.push([fx*(x+px)/cameraZ+cx,fy*(y+py)/cameraZ+cy,cameraZ]);uv.push([u,v]);
    if(i<cols-1&&j<rows-1){const a=j*cols+i;triangles.push([a,a+1,a+cols],[a+1,a+cols+1,a+cols]);}
  }
  return {vertices,uv,triangles};
}


export function directionalMesh(project,values){
  const group=project.directionalFace;if(!group?.enabled)return null;
  const yaw=values.headX||0,entry=group.views?.[yaw<0?'left':'right'];
  const valid=entry?.fitted&&(!project.faceInput||(entry.frontImage===project.faceInput.image&&JSON.stringify(entry.rect)===JSON.stringify(project.faceInput.alignment.rect)));
  const cols=49,rows=73,t=valid?Math.min(1,Math.abs(yaw)/entry.angle):0;
  const smooth=t*t*(3-2*t),matrix=headMatrix(project,{...values,headX:0});
  const vertices=[],uv=[],triangles=[];
  for(let j=0;j<rows;j++)for(let i=0;i<cols;i++){
    const k=j*cols+i,u=i/(cols-1),v=j/(rows-1),delta=valid?entry.offsets[k]:[0,0];
    const x=u+delta[0]*smooth,y=v+delta[1]*smooth;
    vertices.push([matrix[0]*x+matrix[2]*y+matrix[4],matrix[1]*x+matrix[3]*y+matrix[5],0]);uv.push([u,v]);
    if(i<cols-1&&j<rows-1)triangles.push([k,k+1,k+cols],[k+1,k+cols+1,k+cols]);
  }
  return {vertices,uv,triangles};
}


export function multiviewMesh(project,values){
  const m=project.multiViewFace;if(!m?.enabled)return null;
  if(project.faceInput&&(m.frontImage!==project.faceInput.image||JSON.stringify(m.rect)!==JSON.stringify(project.faceInput.alignment.rect)))return null;
  const side=(values.headX||0)<0?'left':'right',v=m.views[side];
  if(!v)return surfaceMesh({...m,strength:1},values);
  if(project.directionalFace?.views?.[side]&&v.image!==project.directionalFace.views[side].image)return null;
  const t=Math.min(1,Math.abs(values.headX||0)/v.angle),R=v.rotation;
  const angle=Math.acos(Math.max(-1,Math.min(1,(R[0][0]+R[1][1]+R[2][2]-1)/2)));
  let axis=[R[2][1]-R[1][2],R[0][2]-R[2][0],R[1][0]-R[0][1]],norm=Math.hypot(...axis);
  if(norm<1e-7){axis=[1,0,0];if(angle>1){axis=R.map((row,i)=>Math.sqrt(Math.max(0,(row[i]+1)/2)));norm=Math.hypot(...axis)||1;axis=axis.map(x=>x/norm);}}else axis=axis.map(x=>x/norm);
  const [ax,ay,az]=axis,c=Math.cos(angle*t),s=Math.sin(angle*t),d=1-c;
  const rot=[[c+ax*ax*d,ax*ay*d-az*s,ax*az*d+ay*s],[ay*ax*d+az*s,c+ay*ay*d,ay*az*d-ax*s],[az*ax*d-ay*s,az*ay*d+ax*s,c+az*az*d]];
  const K=m.intrinsics,k=v.intrinsics.map((row,i)=>row.map((x,j)=>K[i][j]+(x-K[i][j])*t));
  const scale=Math.exp(Math.log(v.scale)*t),mat=headMatrix(project,{...values,headX:0}),smooth=t*t*(3-2*t);
  const vertices=[],uv=[],triangles=[],cols=m.cols,rows=m.rows;
  for(let j=0;j<rows;j++)for(let i=0;i<cols;i++){
    const n=j*cols+i,u=i/(cols-1),vv=j/(rows-1),z=m.depth[n],p=[(u-K[0][2])*z/K[0][0],(vv-K[1][2])*z/K[1][1],z];
    const q=rot.map((row,a)=>row.reduce((sum,x,b)=>sum+x*p[b],0)*scale+v.translation[a]*t);
    const zz=Math.max(.00001,q[2]),offset=v.residualOffsets[n];
    const x=k[0][0]*q[0]/zz+k[0][2]+offset[0]*v.correctionGain*smooth,y=k[1][1]*q[1]/zz+k[1][2]+offset[1]*v.correctionGain*smooth;
    vertices.push([mat[0]*x+mat[2]*y+mat[4],mat[1]*x+mat[3]*y+mat[5],zz]);uv.push([u,vv]);
    if(i<cols-1&&j<rows-1)triangles.push([n,n+1,n+cols],[n+1,n+cols+1,n+cols]);
  }
  return {vertices,uv,triangles};
}

// Warp the original eye aperture with the exact same pinned morph as the lashes.
export function blinkMaskGeometry(project,open){
 const part=project.parts.find(p=>p.visible!==false&&p.tag==='eyelash'&&drivenExpression(project,p,{values:{eyeOpen:open}}));
 if(!part||!project.eyeMask)return null;
 const driven=drivenExpression(project,part,{values:{eyeOpen:open}}),r=project.eyeMask.rect;
 const local=q=>[(q[0]-r[0])/r[2],(q[1]-r[1])/r[3]];
 return {expression:driven.expression,part,vertices:deformedVertices(part,project,{values:{},expressionId:driven.expression.id,strength:driven.strength},null).map(local),
 triangles:part.mesh.triangles,uv:part.mesh.vertices.map(local)};
}
