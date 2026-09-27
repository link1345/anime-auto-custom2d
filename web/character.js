import {expressionMatchesPart} from './deformation.js?v=morphpin1';
import {createPointEditor} from './point-editor.js?v=morphpin1';
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
    <div class="section-title">表情の輪郭対応・モーフ</div>
    <p class="hint">元パーツ → 目標の部品画像 → 形を合わせる → 動作を保存、の順で設定します。顔全体の画像は先にSee-throughで部品抽出してください。</p>
    <label>① 動かしたい元パーツ<select id="fitPart"></select></label>
    <label>② 変形先の表情・部品画像<select id="fitExpression"></select></label>
    <div class="row"><figure>元パーツ<img id="fitBaseImage" alt="元パーツ" style="width:100%;max-height:130px;object-fit:contain"></figure><figure>変形先<img id="fitTargetImage" alt="変形先の部品" style="width:100%;max-height:130px;object-fit:contain"></figure></div>
    <p id="morphStatus" class="hint" aria-live="polite"></p>
    <details><summary>うまく合わない場合：対応点を指定</summary><button id="editMorphPoints">画像をクリックして対応点を置く</button><label class="check"><input id="fitFaceRect" type="checkbox">見本の配置に現在の「顔の位置合わせ」を使う</label><p class="hint">同じ顔キャンバスから抽出した画像用。切り抜き画像には使わないでください。配置の矩形と、形を合わせる対応点は別です。</p><p class="hint">まず空欄で試してください。手動指定は両方を入力します。同じ順序で目頭・目尻など対応する点を指定。座標は各画像の左上が[0,0]、右下が[1,1]です。</p><label>元パーツの点 [[x,y],…]<textarea id="fitBase"></textarea></label><label>変形先の点（同じ順序・個数）<textarea id="fitTarget"></textarea></label></details>
    <button id="fitMorph">③ 形を合わせる（モーフ生成）</button>
    <label>④ 連動する動作<select id="morphDriver"><option value="blink">まばたき（まつげ）</option><option value="mouth">口パク（口）</option><option value="">通常の表情</option></select></label>
    <button id="bindMorph">④ 動作への割り当てを保存</button>
    <label>⑤ 開閉を試す（このプレビューのみ）<input id="morphPreview" type="range" min="0" max="1" step="0.01" value="0"><output id="morphPreviewValue">0%</output></label><button id="morphPreviewReset">自動再生へ戻す</button>
    <p class="hint">まばたき：0%で開眼、100%で閉眼。開いたまつげを変形し、最後の20%で閉じ目画像へ切り替えます。口パク：0%で閉口、100%で開口。開き口画像をメッシュで縮め、閉じる直前の20%だけ閉じ口画像へ切り替えます。通常の表情は「登録表情の強度」で確認します。保存はUndoで戻せます。</p>`;
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
  const pointEditor=createPointEditor({renderer,getItems:()=>{const p=getProject();return [p.parts.find(x=>x.id===$('fitPart').value),p.expressions.find(x=>x.id===$('fitExpression').value)];},onApply:points=>{for(const [i,id] of ['fitBase','fitTarget'].entries())$(id).value=JSON.stringify(points[i].map(p=>p.map(v=>+v.toFixed(6))));toast('対応点を入力しました。③でモーフを生成してください');}});
  $('editMorphPoints').onclick=()=>action(()=>pointEditor.open(['fitBase','fitTarget'].map(id=>$(id).value?JSON.parse($(id).value):[])));
  $('bindMorph').onclick=()=>action(()=>edit({operation:'bind-expression',expressionId:$('fitExpression').value,channel:$('morphDriver').value||null}));
  $('fitMorph').onclick=()=>action(()=>edit({operation:'fit-expression',partId:$('fitPart').value,expressionId:$('fitExpression').value,basePoints:$('fitBase').value?JSON.parse($('fitBase').value):null,targetPoints:$('fitTarget').value?JSON.parse($('fitTarget').value):null,useFaceAlignment:$('fitFaceRect').checked}));
  function updateMorph(){
    const p=getProject(),part=p.parts.find(x=>x.id===$('fitPart').value),expr=p.expressions.find(x=>x.id===$('fitExpression').value);
    for(const [id,item] of [['fitBaseImage',part],['fitTargetImage',expr]]){const image=$(id);image.hidden=!item;if(item)image.src=renderer.url(item.image);else image.removeAttribute('src');}
    const ready=!!part&&!!expr&&expressionMatchesPart(expr,part)&&!!expr?.fit&&expr.fit.partId===part?.id&&!expr.stale;
    const matches=!!part&&!!expr&&expressionMatchesPart(expr,part);
    $('fitMorph').disabled=!matches||expr?.staleReason==='source-replaced';
    const channel=$('morphDriver').value,compatible=!channel||(channel==='blink'?part?.tag==='eyelash':part?.tag==='mouth');
    $('bindMorph').disabled=!ready||!compatible;
    $('morphPreview').disabled=!ready||!expr.driver;
    if(!ready){renderer.eyePreview=undefined;renderer.mouthPreview=undefined;}
    $('morphStatus').textContent=!expr?'表情の部品が未登録です。顔を登録 → Colabで抽出 → 結果ZIPを読み込み・採用してください。':!matches?'見本とパーツの種類が違います。目には閉じ目、口には開き口を選んでください。':expr.stale?(expr.staleReason==='source-replaced'?'素材が更新されています。再抽出した結果を採用してください。':'メッシュが更新されています。③で再フィットしてください。'):!expr.tag?'この画像は通常の差分登録です。顔全体なら先に部品抽出が必要です。':!compatible?'この動作には、まばたき＝まつげ、口パク＝口パーツを選んでください。':ready?(expr.driver?'設定済み。⑤で開閉を確認できます。':'形合わせ済み。④で動作を保存してください。'):'元パーツと変形先を確認し、③を押してください。';
  }
  function renderMorph(){
    const p=getProject(),oldPart=$('fitPart').value,oldExpression=$('fitExpression').value;
    const parts=p.parts.filter(x=>x.visible!==false&&x.scope==='face');
    $('fitPart').replaceChildren(...parts.map(x=>new Option(x.name+' / '+(x.tag||x.role),x.id)));
    if(parts.some(x=>x.id===oldPart))$('fitPart').value=oldPart;
    else {const part=parts[0];$('morphDriver').value=part?.tag==='eyelash'?'blink':part?.tag==='mouth'?'mouth':'';}
    $('fitExpression').replaceChildren(...p.expressions.map(e=>new Option(e.name,e.id)));
    if(p.expressions.some(x=>x.id===oldExpression))$('fitExpression').value=oldExpression;
    updateMorph();
  }
  $('fitPart').onchange=()=>{renderer.eyePreview=undefined;renderer.mouthPreview=undefined;const part=getProject().parts.find(x=>x.id===$('fitPart').value);$('morphDriver').value=part?.tag==='eyelash'?'blink':part?.tag==='mouth'?'mouth':'';$('fitBase').value='';$('fitTarget').value='';updateMorph();};
  $('fitExpression').onchange=()=>{renderer.eyePreview=undefined;renderer.mouthPreview=undefined;$('fitBase').value='';$('fitTarget').value='';updateMorph();};
  $('morphDriver').onchange=updateMorph;
  $('morphPreview').oninput=()=>{const e=getProject().expressions.find(x=>x.id===$('fitExpression').value),value=+$('morphPreview').value;renderer.eyePreview=e?.driver==='blink'?1-value:undefined;renderer.mouthPreview=e?.driver==='mouth'?value:undefined;$('morphPreviewValue').textContent=Math.round(value*100)+'%';renderer.draw();};
  $('morphPreviewReset').onclick=()=>{renderer.eyePreview=undefined;renderer.mouthPreview=undefined;$('eyePreview').value='auto';renderer.draw();};
  action(async()=>{const settings=await api('compute/settings');$('computePython').value=settings.python;$('computeCheckpoint').value=settings.checkpoint;$('computeDevice').value=settings.device;$('depthLevel').value=settings.resolutionLevel;$('depthRefine').value=settings.refineSteps;await poll();});
  const directionUI=createDirectionUI({api,json,action,load,renderer,getProject,toast});
  const parentUI=createParentUI({api,json,action,load,getProject,toast});
  return {render(){const p=getProject();directionUI.render();parentUI.render();$('headSurfaceEnabled').checked=!!p.headSurface?.enabled;$('headSurfaceStrength').value=p.headSurface?.strength??1;$('headSurfaceStatus').textContent=p.headSurface?`${p.headSurface.cols} × ${p.headSurface.rows} 共有頂点 / MoGe3 3D点から構築`:'MoGe3の結果を取り込んだ後に構築してください。';$('eyeScale').value=p.eyeMask?.scale??1;$('eyeScaleValue').textContent=Math.round((p.eyeMask?.scale??1)*100)+'%';renderMorph();$('rigStatus').textContent=p.rig?(p.rig.stale?'関節変更あり。リグを再構築してください。':`${p.rig.bones.length}ボーン / スキン有効 / 頭・肩の髪衝突有効`):'推定または手動配置後、リグを構築してください。';}};
}
