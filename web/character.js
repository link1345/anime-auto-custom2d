import {createDirectionUI} from './direction.js';
import {createParentUI} from './parent-ui.js';
export function createCharacterUI({api,json,action,load,renderer,getProject,getSelected,toast}){
  const $=id=>document.getElementById(id),panel=document.createElement('section');
  panel.innerHTML=`<div class="section-title">自動推定・MoGe3</div>
    <div class="row"><button id="localPose">人物・顔・関節を推定</button><button id="localDepth">MoGe3 深度を推定</button></div>
    <details><summary>ローカル推論の設定</summary><label>推論環境のPython<input id="computePython"></label><label>MoGe3重み / ファイル<input id="computeCheckpoint" value="Ruicheng/moge-3-vitl"></label><label>実行デバイス<select id="computeDevice"><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option></select></label><label>解像度レベル (0〜9)<input id="depthLevel" type="number" min="0" max="9" value="5"></label><label>精細化回数<input id="depthRefine" type="number" min="0" max="10" value="3"></label><button id="saveCompute">設定を保存</button></details>
    <p class="hint">初回は専用Python環境に推論依存を導入してください。GPU推論中も制作画面を操作できます。</p>
    <div class="row"><button id="computeStatus">進捗を確認</button><button id="computeCancel">処理を中止</button><button id="computeApply">完了結果を取り込む</button></div><pre id="computeLog" class="compute-log">未実行</pre>
    <a class="button wide" href="/api/compute/notebook/moge3">MoGe3 Colab ↓</a><a class="button wide" href="/api/job?backend=moge3">MoGe3 ジョブ ZIP ↓</a>
    <a class="button wide" href="/api/compute/notebook/mediapipe">人物・顔・関節 Colab ↓</a><a class="button wide" href="/api/job?backend=mediapipe">人物・顔・関節 ジョブ ZIP ↓</a>
    <button id="computeImport">深度 / 人物推定の結果 ZIP</button><input id="computeFile" type="file" accept=".zip" hidden>
    <div class="section-title">関節・スキニング</div><label class="check"><input id="jointGuides" type="checkbox">関節を表示・クリックで配置</label><label>関節<select id="jointName"></select></label><label>位置 x,y<input id="jointPosition" value="0.5,0.4"></label><button id="saveJoint">関節位置を保存</button><button id="buildRig">ボーン・スキンを構築</button>
    <p id="rigStatus" class="hint"></p><a class="button wide" href="/api/character.glb">スキン・モーフ付き GLB ↓</a>`;
  document.querySelector('.left').append(panel);
  const joints=['root','neck','head','shoulderLeft','elbowLeft','wristLeft','handLeft','shoulderRight','elbowRight','wristRight','handRight','hipLeft','kneeLeft','ankleLeft','hipRight','kneeRight','ankleRight'];
  $('jointName').replaceChildren(...joints.map(k=>new Option(k,k)));
  const tools=document.createElement('section');tools.innerHTML=`<div class="section-title">腕・マスク・隠れ部分</div><label>分割する腕<select id="armSide"><option value="Left">画面左</option><option value="Right">画面右</option></select></label><button id="splitArm">選択パーツを上腕・前腕・手へ分割</button><div class="section-title">共通立体顔面 / MoGe3</div><button id="buildHeadSurface">MoGe3から共通立体顔面を構築</button><label class="check"><input id="headSurfaceEnabled" type="checkbox">共通立体顔面を使う</label><label>顔面の凹凸の強さ<input id="headSurfaceStrength" type="range" min="0" max="1.5" step="0.05" value="1"></label><button id="saveHeadSurface">立体顔面設定を保存</button><p id="headSurfaceStatus" class="hint"></p><p class="hint">共有面をカメラ座標で回転。左右・上下は最大15度。元画像の見えていない横顔は生成しません。前髪の表示はパーツ設定で変更できます。</p><div class="section-title">目の共通サイズ・まばたき</div><label>目の共通サイズ（まつげ・白目・瞳）<input id="eyeScale" type="range" min="0.5" max="1.5" step="0.01" value="1"><output id="eyeScaleValue">100%</output></label><button id="syncEyes">目の配置・変形を統一</button><button id="eyeMasks">共通サイズ・白目マスクを保存</button><label>まばたき確認<select id="eyePreview"><option value="auto">自動</option><option value="1">開く</option><option value="0.5">半分閉じる</option><option value="0">閉じる</option></select></label><p class="hint">左右分割にも対応。統一は表示中の目パーツを白目と同じ配置・変形に揃えます。瞳は白目の輪郭内に制限し、閉じると下の肌が見え、まつげが閉じ線になります。肌は目の描かれていない補完済み素材を使ってください。Undoで戻せます。</p><label>隠れ境界の補完幅 (px)<input id="hiddenPixels" type="number" min="1" max="32" value="8"></label><button id="extendHidden">選択パーツの隠れ境界を延長</button><p class="hint">前のパーツに隠れる範囲だけ色を延長します。生成補完はSee-throughの推論結果を使います。</p>
    <div class="section-title">表情の輪郭対応・モーフ</div><label>表情<select id="fitExpression"></select></label><details><summary>対応点を手動指定（任意）</summary><label>基準パーツの点 [[x,y],…]<textarea id="fitBase" placeholder="空なら透過輪郭から取得"></textarea></label><label>表情画像の点（同じ順序）<textarea id="fitTarget" placeholder="空なら透過輪郭から取得"></textarea></label></details><button id="fitMorph">選択パーツを基準にフィット・モーフ生成</button><p class="hint">対応点は各画像内の0〜1座標。輪郭の自動対応は透過シルエットを使います。反転メッシュは拒否します。生成後は表情強度で変形を確認できます。</p>`;
  $('editorPanel').append(tools);
  const edit=async data=>{await api('character',json('POST',data));await load();toast('モデル構成を保存しました');};
  for(const [id,backend] of [['localPose','mediapipe'],['localDepth','moge3']])$(id).onclick=()=>action(async()=>{await api('compute/local',json('POST',{backend}));await poll();});
  async function poll(){clearTimeout(timer);const state=await api('compute/local');$('computeLog').textContent=[state.status||'未実行',state.error||'',state.tail||''].join('\n');if(['queued','running'].includes(state.status))timer=setTimeout(()=>action(poll),3000);}
  let timer; $('computeStatus').onclick=()=>action(poll);
  $('computeCancel').onclick=()=>action(async()=>{await api('compute/local',json('POST',{cancel:true}));await poll();});
  $('computeApply').onclick=()=>action(async()=>{await api('compute/local',json('POST',{apply:true}));await load();await poll();});
  $('saveCompute').onclick=()=>action(async()=>{await api('compute/settings',json('PATCH',{python:$('computePython').value,checkpoint:$('computeCheckpoint').value,device:$('computeDevice').value,resolutionLevel:+$('depthLevel').value,refineSteps:+$('depthRefine').value}));toast('推論設定を保存しました');});
  $('computeImport').onclick=()=>$('computeFile').click();$('computeFile').onchange=()=>action(async()=>{const file=$('computeFile').files[0];if(!file)return;const data=new FormData();data.append('file',file);await api('result',{method:'POST',body:data});await load();$('computeFile').value='';});
  $('jointGuides').onchange=()=>{renderer.jointGuides=$('jointGuides').checked;renderer.draw();};
  $('jointName').onchange=()=>{const joint=getProject()?.landmarks?.joints[$('jointName').value];if(joint)$('jointPosition').value=joint.position.join(',');};
  $('saveJoint').onclick=()=>action(()=>edit({operation:'joint',name:$('jointName').value,position:$('jointPosition').value.split(',').map(Number)}));
  $('avatar').addEventListener('pointerdown',event=>{if(!$('jointGuides').checked)return;event.stopImmediatePropagation();const position=renderer.modelPoint(event);$('jointPosition').value=position.map(x=>x.toFixed(4)).join(',');action(()=>edit({operation:'joint',name:$('jointName').value,position}));},true);
  $('buildRig').onclick=()=>action(()=>edit({operation:'rig'}));
  $('splitArm').onclick=()=>action(()=>edit({operation:'split-arm',partId:getSelected(),side:$('armSide').value}));
  $('buildHeadSurface').onclick=()=>action(()=>edit({operation:'head-surface'}));
  $('saveHeadSurface').onclick=()=>action(()=>edit({operation:'head-surface-settings',enabled:$('headSurfaceEnabled').checked,strength:+$('headSurfaceStrength').value}));
  $('headSurfaceStrength').oninput=()=>{if(getProject().headSurface){getProject().headSurface.strength=+$('headSurfaceStrength').value;renderer.draw();}};
  $('headSurfaceEnabled').onchange=()=>{if(getProject().headSurface){getProject().headSurface.enabled=$('headSurfaceEnabled').checked;renderer.draw();}};
  $('eyeMasks').onclick=()=>action(()=>edit({operation:'eye-masks',scale:+$('eyeScale').value}));
  $('syncEyes').onclick=()=>action(()=>edit({operation:'eye-masks',scale:+$('eyeScale').value,synchronize:true}));
  $('eyeScale').oninput=()=>{$('eyeScaleValue').textContent=Math.round(+$('eyeScale').value*100)+'%';if(getProject().eyeMask){getProject().eyeMask.scale=+$('eyeScale').value;renderer.draw();}};
  $('eyePreview').onchange=()=>{renderer.eyePreview=$('eyePreview').value==='auto'?undefined:+$('eyePreview').value;renderer.draw();};
  $('extendHidden').onclick=()=>action(()=>edit({operation:'hidden',partId:getSelected(),pixels:+$('hiddenPixels').value}));
  $('fitMorph').onclick=()=>action(()=>edit({operation:'fit-expression',partId:getSelected(),expressionId:$('fitExpression').value,basePoints:$('fitBase').value?JSON.parse($('fitBase').value):null,targetPoints:$('fitTarget').value?JSON.parse($('fitTarget').value):null}));
  action(async()=>{const settings=await api('compute/settings');$('computePython').value=settings.python;$('computeCheckpoint').value=settings.checkpoint;$('computeDevice').value=settings.device;$('depthLevel').value=settings.resolutionLevel;$('depthRefine').value=settings.refineSteps;await poll();});
  const directionUI=createDirectionUI({api,json,action,load,renderer,getProject,toast});
  const parentUI=createParentUI({api,json,action,load,getProject,toast});
  return {render(){const p=getProject();directionUI.render();parentUI.render();$('headSurfaceEnabled').checked=!!p.headSurface?.enabled;$('headSurfaceStrength').value=p.headSurface?.strength??1;$('headSurfaceStatus').textContent=p.headSurface?`${p.headSurface.cols} × ${p.headSurface.rows} 共有頂点 / MoGe3 3D点から構築`:'MoGe3の結果を取り込んだ後に構築してください。';$('eyeScale').value=p.eyeMask?.scale??1;$('eyeScaleValue').textContent=Math.round((p.eyeMask?.scale??1)*100)+'%';$('fitExpression').replaceChildren(...p.expressions.map(e=>new Option(e.name,e.id)));$('rigStatus').textContent=p.rig?(p.rig.stale?'関節変更あり。リグを再構築してください。':`${p.rig.bones.length}ボーン / スキン有効 / 頭・肩の髪衝突有効`):'推定または手動配置後、リグを構築してください。';}};
}
