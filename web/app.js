import {Renderer,connect} from './renderer.js?v=perfgrid1';
import {createSegmentationUI} from './segmentation.js';
import {createCharacterUI} from './character.js?v=perfgrid1';
import {cursorAngles,cursorGaze} from './parent-motion.js?v=perfgrid1';

const $=id=>document.getElementById(id);
const renderer=new Renderer($('avatar'));
let project, selected='head', candidate=null, expressionId=null, frame=null, loadingPromise=null, loadAgain=false;
const segmentationUI=createSegmentationUI({api,json:(method,data)=>json(method,data),action,toast,load,renderer,getProject:()=>project,getSelected:()=>selected,showCandidate});
const characterUI=createCharacterUI({api,json:(method,data)=>json(method,data),action,load,renderer,getProject:()=>project,getSelected:()=>selected,toast});
let renderTimes=[], lastRender=performance.now(), frameCount=0, reportAt=performance.now();
function toast(message,error=false){$('toast').textContent=message;$('toast').style.background=error?'#633e37':'#36473e';$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,6000);}
async function api(path,options={}) {
  const response=await fetch(`/api/${path}`,options);
  const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data;
}
const json=(method,data)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
async function action(fn){try{await fn();}catch(error){toast(error.message,true);}}
const socket=connect(packet=>{
  frame=packet; renderer.draw(frame);
  $('pause').textContent=packet.playing?'Ⅱ 停止':'▶ 再生';
  const now=performance.now();renderTimes.push(now-lastRender);lastRender=now;frameCount++;
  if(now-reportAt>2000){const samples=renderTimes.sort((a,b)=>a-b);$('performance').textContent=`${(frameCount*1000/(now-reportAt)).toFixed(1)} FPS · p95 ${samples[Math.floor(samples.length*.95)].toFixed(0)} ms`;renderTimes=[];frameCount=0;reportAt=now;}
},()=>action(load),connected=>{$('connection').textContent=connected?'ローカル接続':'再接続中';},packet=>{if(!packet.ok)toast(packet.error,true);});

function load(){
  loadAgain=true;
  if(loadingPromise)return loadingPromise;
  loadingPromise=(async()=>{do{loadAgain=false;project=await api('project');await renderer.load(project);candidate=null;$('candidate').hidden=true;renderUI();}while(loadAgain);})().finally(()=>{loadingPromise=null;});
  return loadingPromise;
}
function renderUI(){
  $('characterName').textContent=project.name;$('partCount').textContent=project.parts.length;
  $('backend').textContent=[project.depth?'MoGe3 深度':'仮の深度',project.rig?`${project.rig.bones.length}ボーン${project.rig.stale?' / 再構築が必要':''}`:'テンプレート分割'].join(' / ');
  if(project.faceInput)$('faceStatus').textContent=`登録済み ${project.faceInput.validation.width} × ${project.faceInput.validation.height} / ${project.faceInput.validation.upscaledByApp?'FFmpeg拡大済み':'入力原寸'} / 原画は別保存`;
  else $('faceStatus').textContent='未登録。顔を大きく収めた正方形を登録してください。FFmpegで2160px以上に拡大できます。';
  $('warningCount').textContent=project.warnings.length;
  $('warnings').replaceChildren(...project.warnings.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  if(project.stale.length){const li=document.createElement('li');li.textContent='再処理が必要: '+project.stale.join(', ');$('warnings').append(li);}
  $('parts').replaceChildren(...project.parts.map(part=>{
    const button=document.createElement('button');button.className='part'+(part.id===selected?' active':'');
    const text=document.createElement('b');text.textContent=part.name;const status=document.createElement('span');status.textContent=part.locked?'LOCK':part.provenance==='see-through'?'推定':part.provenance==='manual-split'?'分割':'仮';button.append(text,status);button.onclick=()=>{selected=part.id;renderer.selected=selected;renderUI();};return button;
  }));
  const part=project.parts.find(p=>p.id===selected)||project.parts[0];selected=part.id;renderer.selected=selected;
  $('selectedName').textContent=part.name;$('partName').value=part.name;$('locked').checked=part.locked;
  $('pivotX').value=part.pivot[0];$('pivotY').value=part.pivot[1];$('depth').value=part.depth;
  $('physicsEnabled').checked=part.physics.enabled;
  for(const key of ['strength','stiffness','damping'])$(key).value=part.physics[key];
  $('expressions').replaceChildren(...project.expressions.map(expression=>{
    const button=document.createElement('button');button.className='expression-card';const image=document.createElement('img');image.src=renderer.url(expression.image);image.alt=expression.name;button.append(image,document.createTextNode(expression.name));button.onclick=()=>{expressionId=expression.id;sendExpression();};return button;
  }));
  action(async()=>{const projects=await api('projects');$('projects').replaceChildren(...projects.map(p=>new Option(p.name,p.id,false,p.id===project.id)));});
  renderer.draw(frame);
  segmentationUI.render();
  characterUI.render();
  for(const [label,key,fallback] of [['頭・左右','yawLimit',20],['頭・上下','pitchLimit',15]]){const input=document.querySelector(`input[aria-label="${label}"]`);if(input){const limit=project.parentMotion?.enabled?(project.parentMotion[key]??fallback):fallback;input.min=-limit;input.max=limit;input.step=limit/50;}}
}
async function showCandidate(value){candidate=value;await renderer.load(value.project);$('candidate').hidden=false;const summary=$('candidate').querySelector('span');summary.textContent=`${value.parts.length}レイヤーをプレビュー中。${value.warnings?.join(' / ')||'位置と重なりを確認して採用してください。'}`;toast('See-through結果を候補として読み込みました');}
async function edit(values){await api(`parts/${selected}`,json('PATCH',values));await load();}
$('partName').onchange=()=>action(()=>edit({name:$('partName').value}));
$('locked').onchange=()=>action(()=>edit({locked:$('locked').checked}));
for(const id of ['pivotX','pivotY'])$(id).onchange=()=>action(()=>edit({pivot:[+$('pivotX').value,+$('pivotY').value]}));
$('depth').onchange=()=>action(()=>edit({depth:+$('depth').value}));
for(const id of ['physicsEnabled','strength','stiffness','damping'])$(id).onchange=()=>action(()=>edit({physics:{enabled:$('physicsEnabled').checked,strength:+$('strength').value,stiffness:+$('stiffness').value,damping:+$('damping').value}}));
$('projects').onchange=()=>action(async()=>{await api(`projects/${$('projects').value}/select`,{method:'POST'});await load();});
$('new').onclick=()=>$('sourceFile').click();
$('faceInput').onclick=()=>{if(!$('faceConfirmed').checked){toast('顔を大きく収めた画像であることを確認してください',true);return;}$('faceFile').click();};
$('faceFile').onchange=()=>action(async()=>{const file=$('faceFile').files[0];if(!file)return;const data=new FormData();data.append('image',file);data.append('faceFillsFrame',String($('faceConfirmed').checked));data.append('upscale',String($('ffmpegUpscale').checked));data.append('size',$('faceResolution').value);toast('顔素材を保存・FFmpeg処理中…');await api('face',{method:'POST',body:data});await load();toast('顔素材と原画像を保存しました');$('faceFile').value='';});
$('sourceFile').onchange=()=>action(async()=>{const file=$('sourceFile').files[0];if(!file)return;const data=new FormData();data.append('image',file);data.append('name',file.name.replace(/\.[^.]+$/,''));toast('原画を取り込み、仮のパーツを生成中…');await api('projects',{method:'POST',body:data});await load();toast('原画を保存しました。分割と顔の基準位置を確認してください。');});
for(const [button,input,path,field]of [['import','archiveFile','import','file'],['result','resultFile','result','file'],['replace','partFile',()=>`parts/${selected}/image`,'image']]){
  $(button).onclick=()=>$(input).click();$(input).onchange=()=>action(async()=>{const file=$(input).files[0];if(!file)return;const data=new FormData();data.append(field,file);await api(typeof path==='function'?path():path,{method:'POST',body:data});await load();toast('取り込みました');$(input).value='';});
}
for(const id of ['undo','redo','save'])$(id).onclick=()=>action(async()=>{await api(`history/${id==='save'?'checkpoint':id}`,{method:'POST'});await load();toast(id==='save'?'チェックポイントを保存しました':'編集を復元しました');});
$('regenerate').onclick=()=>action(async()=>{toast('生成中…');candidate=await api('regenerate',{method:'POST'});const preview=structuredClone(project);preview.parts=preview.parts.map(p=>p.locked?p:(candidate.parts.find(x=>x.id===p.id)||p));await renderer.load(preview);$('candidate').hidden=false;toast('候補をプレビュー中です');});
$('adopt').onclick=()=>action(async()=>{await api('adopt',json('POST',{id:candidate.id}));await load();toast('候補を採用しました');});
$('discard').onclick=()=>action(async()=>{await renderer.load(project);candidate=null;$('candidate').hidden=true;});
for(const [id,view]of [['modelView','model'],['sourceView','source'],['compareView','compare']])$(id).onclick=()=>{renderer.view=view;for(const key of ['modelView','sourceView','compareView'])$(key).classList.toggle('active',key===id);renderer.draw(frame);};
$('guides').onchange=()=>renderer.guides=$('guides').checked;$('mesh').onchange=()=>renderer.mesh=$('mesh').checked;
$('pause').onclick=()=>socket.send({type:'pause'});$('motion').onchange=()=>socket.send({type:'motion',name:$('motion').value});
$('resetPose').onclick=()=>{socket.send({type:'stop'});socket.send({type:'parameters',source:'human',values:{headX:0,headY:0,headZ:0,eyeOpen:1,mouthOpen:0,armLeft:0,armRight:0,elbowLeft:0,elbowRight:0,wristLeft:0,wristRight:0}});};
for(const [id,live]of [['editMode',false],['liveMode',true]])$(id).onclick=()=>{$('editorPanel').hidden=live;$('livePanel').hidden=!live;$('editMode').classList.toggle('active',!live);$('liveMode').classList.toggle('active',live);};
$('obsUrl').value=`http://${location.host}/output`;$('wsUrl').textContent=`ws://${location.host}/ws`;
const params=[['headX','頭・左右',-20,20],['headY','頭・上下',-15,15],['headZ','首の傾き',-12,12],['eyeOpen','目の開き',0,1],['mouthOpen','口の開き',0,1],['smile','笑顔',0,1],['armLeft','右腕',-30,30],['armRight','左腕',-30,30],['elbowLeft','右肘',-90,90],['elbowRight','左肘',-90,90],['wristLeft','右手首',-60,60],['wristRight','左手首',-60,60]];
const owners={};
for(const [key,label,min,max]of params){
  const container=document.createElement('label');container.textContent=label;const input=document.createElement('input');input.type='range';input.min=min;input.max=max;input.step=(max-min)/100;input.value=key==='eyeOpen'?1:0;input.setAttribute('aria-label',label);
  input.oninput=()=>{const source=frame?.mode==='ai'?'ai':frame?.mode==='human'?'human':frame?.owners[key]||'human';socket.send({type:'parameters',source,values:{[key]:+input.value}});};container.append(input);$('controls').append(container);
  const field=document.createElement('label');field.textContent=label;const select=document.createElement('select');select.append(new Option('人','human'),new Option('AI','ai'));select.value=key.startsWith('head')?'human':'ai';owners[key]=select.value;select.onchange=()=>{owners[key]=select.value;setMode();};field.append(select);$('ownership').append(field);
}
function setMode(){socket.send({type:'mode',mode:$('inputMode').value,owners});}
$('inputMode').onchange=setMode;
function sendExpression(){if(expressionId)socket.send({type:'expression',expressionId,strength:+$('expressionStrength').value,duration:.25});}
$('expressionStrength').oninput=sendExpression;
for(const value of [0,.25,.5,.75,1]){const button=document.createElement('button');button.textContent=`${value*100}%`;button.onclick=()=>{$('expressionStrength').value=value;sendExpression();};$('stops').append(button);}
$('addExpression').onclick=()=>$('expressionDialog').showModal();$('cancelExpression').onclick=()=>$('expressionDialog').close();
$('expressionForm').onsubmit=event=>{event.preventDefault();action(async()=>{const data=new FormData(event.target);data.set('rect',JSON.stringify(String(data.get('rect')).split(',').map(Number)));await api('expressions',{method:'POST',body:data});$('expressionDialog').close();await load();toast('表情を保存しました');});};
let dragging=-1, moved=false;
$('avatar').onpointerdown=event=>{
  if(!renderer.guides||candidate||!project)return;
  const part=project.parts.find(p=>p.id===selected);if(part.locked){toast('保護を解除してから編集してください');return;}
  const point=renderer.modelPoint(event);dragging=part.polygon.findIndex(p=>Math.hypot(p[0]-point[0],p[1]-point[1])<.025);moved=false;$('avatar').setPointerCapture(event.pointerId);
};
$('avatar').onpointermove=event=>{
  if(!project)return;const point=renderer.modelPoint(event);
  if(dragging>=0){project.parts.find(p=>p.id===selected).polygon[dragging]=point.map(x=>Math.max(0,Math.min(1,x)));moved=true;renderer.draw(frame);}
  else if($('pointerTracking').checked){socket.send({type:'parameters',source:'human',values:{...cursorAngles(project,point),...cursorGaze(project,point)}});}
};
$('avatar').onpointerleave=()=>socket.send({type:'parameters',source:'human',values:{gazeX:0,gazeY:0}});
$('avatar').onpointerup=event=>{
  if(!renderer.guides||candidate||!project)return;const part=project.parts.find(p=>p.id===selected);if(part.locked)return;
  if(dragging>=0){dragging=-1;if(moved)action(()=>edit({polygon:part.polygon}));}
  else {const point=renderer.modelPoint(event);if(point.every(x=>x>=0&&x<=1))action(()=>edit({pivot:point}));}
};
await action(load);
