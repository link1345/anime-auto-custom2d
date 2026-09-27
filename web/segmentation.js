/** Authoring controls for remote segmentation and non-destructive face decoration. */
export function createSegmentationUI({api,json,action,toast,load,renderer,getProject,getSelected,showCandidate}){
  const $=id=>document.getElementById(id);
  const facePanel=document.createElement('section');facePanel.className='segmentation-panel';
  facePanel.innerHTML=`<div class="section-title">顔入力・FFmpeg</div>
    <label class="check"><input id="ffmpegUpscale" type="checkbox" checked>FFmpegで必要解像度へ拡大</label>
    <label>入力サイズ<select id="faceResolution"><option value="2160">2160 × 2160</option><option value="3072">3072 × 3072</option></select></label>
    <p class="hint">原画像は別に保持。Lanczosで拡大し、拡大履歴を保存します。生成による細部復元は行いません。</p>
    <div class="section-title">顔の位置合わせ</div><label>配置 x, y, 幅, 高さ<input id="faceRect" value="0.245,0.01,0.52,0.3467"></label>
    <label class="check"><input id="faceOverlay" type="checkbox">顔素材を重ねて確認</label>
    <label>比較の透明度<input id="faceOverlayOpacity" type="range" min="0" max="1" step="0.05" value="0.5"></label>
    <button class="wide" id="alignFace">位置合わせを保存</button>
    <div class="section-title">See-through / Colab</div>
    <a class="button wide" href="/api/colab">① Colabノートブック ↓</a>
    <a class="button wide" href="/api/face-job?scope=face">② 顔8種類のジョブ ZIP ↓</a>
    <a class="button wide" href="/api/face-job?scope=body">② 髪・体のジョブ ZIP ↓</a>
    <button class="wide" id="segmentationResult">③ See-through結果 ZIP</button>
    <input hidden id="segmentationFile" type="file" accept=".zip">
    <p class="hint">Colabでノートブックを開き、GPUを設定してジョブを実行。結果を読み込むと候補を確認してから採用できます。</p>
    <div id="segmentationStatus" class="hint"></div>
    <label class="check"><input id="templatesVisible" type="checkbox" checked>元の仮パーツを表示</label>
    <button id="restoreCandidate" class="wide">未採用の結果候補を再表示</button>`;
  document.querySelector('.left').append(facePanel);
  const expressions=document.createElement('section');expressions.innerHTML=`<div class="section-title">表情別の顔・部品抽出</div><label>表情名<input id="expressionSourceName" value="閉じ目"></label><label>顔の正方形画像<input id="expressionSourceFile" type="file" accept="image/png,image/jpeg,image/webp"></label><label class="check"><input id="expressionUpscale" type="checkbox" checked>FFmpegで2160pxへ拡大</label><p class="hint">顔が大きく入った正方形を使用。チェックを外す場合は2160px以上が必要です。同じ表情名は上書きします。元の顔・パーツは置換しません。</p><label>表情顔の配置 x,y,幅,高さ<input id="expressionSourceRect" placeholder="空欄なら正面顔と同じ配置"></label><div id="expressionTags"></div><button id="saveExpressionSource">表情の顔を登録</button><label>登録した顔<select id="expressionSourceList"></select></label><a id="expressionJob" class="button wide">選択部品のColabジョブ ZIP ↓</a><p class="hint">上のColabノートブックで実行し、See-through結果ZIPを読み込んで採用。抽出結果は表情ライブラリへ追加されます。抽出対象を絞ってもモデル本体は全レイヤーを推論します。</p>`;
  document.querySelector('.left').append(expressions);
  for(const [tag,name] of Object.entries({eyebrow:'眉毛',eyelash:'まつげ',mouth:'口',nose:'鼻',face:'肌',ears:'耳',irides:'瞳',eyewhite:'白目'})){
    const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=tag;input.checked=['eyelash','mouth'].includes(tag);label.append(input,name);$('expressionTags').append(label);
  }
  $('saveExpressionSource').onclick=()=>action(async()=>{
    const file=$('expressionSourceFile').files[0];if(!file)throw Error('顔画像を選んでください');
    const data=new FormData();data.set('image',file);data.set('name',$('expressionSourceName').value);data.set('upscale',String($('expressionUpscale').checked));data.set('rect',JSON.stringify($('expressionSourceRect').value.trim()?$('expressionSourceRect').value.split(',').map(Number):(getProject().faceInput?.alignment?.rect??[0,0,1,1])));data.set('tags',JSON.stringify([...$('expressionTags').querySelectorAll('input:checked')].map(x=>x.value)));
    await api('expression-source',{method:'POST',body:data});await load();toast('表情顔を登録しました');
  });
  $('expressionSourceList').onchange=()=>{$('expressionJob').href='/api/face-job?scope=face&expression='+encodeURIComponent($('expressionSourceList').value);};
  const splitPanel=document.createElement('section');splitPanel.innerHTML=`<div class="section-title">レイヤー・細分化</div>
    <label class="check"><input type="checkbox" id="partVisible" checked>このパーツを表示</label>
    <label>動く部位<select id="partRole"><option value="head">頭</option><option value="neck">首</option><option value="chest">胸</option><option value="armLeft">右腕（画面左）</option><option value="armRight">左腕（画面右）</option><option value="forearmLeft">前腕・画面左</option><option value="forearmRight">前腕・画面右</option><option value="handLeft">手・画面左</option><option value="handRight">手・画面右</option><option value="skirt">スカート</option><option value="legs">足</option><option value="feet">靴</option><option value="hairFront">前髪</option><option value="hairBack">後ろ髪</option><option value="hairLeft">横髪・右</option><option value="hairRight">横髪・左</option></select></label>
    <label>切断方向<select id="splitAxis"><option value="x">縦線で左右に分割</option><option value="y">横線で上下に分割</option></select></label>
    <label>切断位置（原画座標）<input id="splitPosition" type="range" min="0.01" max="0.99" step="0.005" value="0.5"></label>
    <label class="check"><input id="showSplit" type="checkbox">切断線を表示</label>
    <div class="row"><button id="splitLayer">2つに分割</button><button id="splitIslands">独立領域を分割</button></div>
    <p class="hint">髪の毛束や服・袖を切断線で細分化できます。「独立領域」は透明な縦の隙間で左右に分かれた領域が対象です。隠れた形状は作りません。元パーツを保管し、Undoできます。</p>`;
  $('editorPanel').append(splitPanel);
  const effectsPanel=document.createElement('section');effectsPanel.className='effects-panel';effectsPanel.innerHTML=`<div class="eyebrow">FACE DECORATION</div><h2>顔の要素を追加</h2><div class="row"><button id="addBlush">＋ 頬の赤み</button><button id="addStar">＋ 星の瞳</button><button id="addHeart">＋ ハートの瞳</button><button id="addFaceImage">＋ 透過PNG</button></div><p class="hint">独立した装飾として保存。非表示・強度変更・削除をしても肌や瞳の画像を変更しません。</p><div id="faceEffects"></div>`;
  document.querySelector('main').append(effectsPanel);
  const dialog=document.createElement('dialog');dialog.id='effectDialog';dialog.innerHTML=`<form id="effectForm"><h2>顔の装飾PNGを追加</h2><label>透過PNG<input required name="image" type="file" accept="image/png"></label><label>名前<input required name="name" value="顔の装飾"></label><label>配置 x, y, 幅, 高さ<input name="rect" value="0.38,0.21,0.24,0.05"></label><div class="row"><button type="button" id="closeEffect">キャンセル</button><button type="submit">追加して保存</button></div></form>`;document.body.append(dialog);
  $('alignFace').onclick=()=>action(async()=>{await api('face/alignment',json('PATCH',{rect:$('faceRect').value.split(',').map(Number)}));await load();toast('顔の位置を保存しました');});
  function overlay(){renderer.facePreview=$('faceOverlay').checked;renderer.facePreviewOpacity=+$('faceOverlayOpacity').value;renderer.draw();}
  $('faceOverlay').onchange=overlay;$('faceOverlayOpacity').oninput=overlay;
  $('faceRect').oninput=()=>{const p=renderer.project,rect=$('faceRect').value.split(',').map(Number);if(p?.faceInput&&rect.length===4&&rect.every(Number.isFinite)&&rect[2]>0&&rect[3]>0){renderer.project={...p,faceInput:{...p.faceInput,alignment:{...p.faceInput.alignment,rect}}};renderer.draw();}};
  $('segmentationResult').onclick=()=>$('segmentationFile').click();
  $('segmentationFile').onchange=()=>action(async()=>{const file=$('segmentationFile').files[0];if(!file)return;const form=new FormData();form.append('file',file);toast('レイヤーを検査して読み込み中…');await showCandidate(await api('segmentation/result',{method:'POST',body:form}));$('segmentationFile').value='';});
  $('templatesVisible').onchange=()=>action(async()=>{await api('display',json('PATCH',{templatesVisible:$('templatesVisible').checked}));await load();});
  $('restoreCandidate').onclick=()=>action(async()=>{const candidates=await api('segmentation/candidates');if(!candidates.length)throw Error('現在の編集状態に対応する候補はありません');await showCandidate(candidates[candidates.length-1]);});
  $('partVisible').onchange=()=>action(async()=>{await api(`parts/${getSelected()}`,json('PATCH',{visible:$('partVisible').checked}));await load();});
  $('partRole').onchange=()=>action(async()=>{await api(`parts/${getSelected()}`,json('PATCH',{role:$('partRole').value}));await load();});
  function guide(){renderer.splitGuide=$('showSplit').checked?{axis:$('splitAxis').value,position:+$('splitPosition').value}:null;renderer.draw();}
  $('showSplit').onchange=guide;$('splitPosition').oninput=guide;$('splitAxis').onchange=guide;
  for(const [id,auto]of [['splitLayer',false],['splitIslands',true]])$(id).onclick=()=>action(async()=>{await api(`parts/${getSelected()}/split`,json('POST',{auto,axis:$('splitAxis').value,position:+$('splitPosition').value}));await load();toast('分割しました。子パーツは保護済みです。編集時は保護を解除してください。');});
  for(const [id,kind]of [['addBlush','blush'],['addStar','star'],['addHeart','heart']])$(id).onclick=()=>action(async()=>{await api('face/effects',json('POST',{kind}));await load();});
  $('addFaceImage').onclick=()=>$('effectDialog').showModal();$('closeEffect').onclick=()=>$('effectDialog').close();
  $('effectForm').onsubmit=event=>{event.preventDefault();action(async()=>{const form=new FormData(event.target);form.set('rect',JSON.stringify(String(form.get('rect')).split(',').map(Number)));await api('face/effect-image',{method:'POST',body:form});$('effectDialog').close();await load();});};
  return {render(){const sources=getProject().expressionSources||[];$('expressionSourceList').replaceChildren(...sources.map(e=>new Option(e.name,e.id)));$('expressionJob').hidden=!sources.length;$('expressionSourceList').onchange();
    const project=getProject(),part=project.parts.find(p=>p.id===getSelected());
    if(!part)return;
    $('partVisible').checked=part.visible!==false;$('partVisible').disabled=part.locked;
    $('partRole').value=part.role;$('partRole').disabled=part.locked;
    $('templatesVisible').checked=project.display?.templatesVisible!==false;
    if(project.faceInput)$('faceRect').value=project.faceInput.alignment.rect.join(',');
    const statuses=Object.entries(project.segmentation||{}).map(([scope,s])=>`${scope==='face'?'顔':'髪・体'}: 採用済み / 未検出 ${s.missing.length}`);
    $('segmentationStatus').textContent=statuses.join(' ・ ');
    $('faceEffects').replaceChildren(...(project.faceEffects||[]).map(effect=>{
      const row=document.createElement('div');row.className='effect-row';const check=document.createElement('input');check.type='checkbox';check.checked=effect.visible;check.setAttribute('aria-label',`${effect.name}を表示`);
      const name=document.createElement('span');name.textContent=effect.name;
      const opacity=document.createElement('input');opacity.type='range';opacity.min=0;opacity.max=1;opacity.step=.05;opacity.value=effect.opacity;opacity.setAttribute('aria-label',`${effect.name}の強度`);
      const scale=document.createElement('input');scale.type='range';scale.min=0;scale.max=3;scale.step=.1;scale.value=effect.scale;scale.setAttribute('aria-label',`${effect.name}の大きさ`);
      const remove=document.createElement('button');remove.textContent='削除';remove.setAttribute('aria-label',`${effect.name}を削除`);
      async function update(values){await api('face/effects',json('POST',{id:effect.id,...values}));await load();}
      check.onchange=()=>action(()=>update({visible:check.checked}));opacity.onchange=()=>action(()=>update({opacity:+opacity.value}));scale.onchange=()=>action(()=>update({scale:+scale.value}));remove.onclick=()=>action(async()=>{await api('face/effects',json('POST',{delete:effect.id}));await load();});
      row.append(check,name,opacity,scale,remove);return row;
    }));
  }};
}
