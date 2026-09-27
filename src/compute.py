"""Persisted compute jobs, strict result integration and asynchronous local workers."""
import asyncio
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
from .project import uid,safe_id
from .character import landmarks_from_detection

ROOT=Path(__file__).resolve().parent.parent


def make_job(store,p,backend,scope=None):
    if backend not in ('moge3','mediapipe'):raise ValueError('Unknown compute backend')
    asset_path=p['source'];size=p['size']
    if scope:
        if backend!='moge3' or scope not in ('front','left','right'):raise ValueError('Invalid face depth scope')
        asset_path=p.get('faceInput',{}).get('image') if scope=='front' else p.get('directionalFace',{}).get('views',{}).get(scope,{}).get('image')
        if not asset_path:raise ValueError('対象の顔画像を先に登録してください')
        with Image.open(store.folder(p['id'])/asset_path) as im:size=list(im.size)
        if size[0]!=size[1] or size[0]<2160:raise ValueError('顔画像は2160px以上の正方形が必要です')
    raw=(store.folder(p['id'])/asset_path).read_bytes()
    job={'schemaVersion':2,'id':uid(),'projectId':p['id'],'revision':p['revision'],'backend':backend,
         'inputSha256':hashlib.sha256(raw).hexdigest(),'sourceAsset':asset_path,'sourceSize':size}
    if scope:job['faceScope']=scope
    folder=store.folder(p['id'])/'jobs';folder.mkdir(exist_ok=True)
    (folder/(job['id']+'.json')).write_text(json.dumps(job),'utf-8')
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('job.json',json.dumps(job));z.writestr('source.png',raw)
    return job,output.getvalue()


def integrate(store,project,payload):
    p=copy.deepcopy(project);folder=store.folder(p['id'])
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=z.namelist()
        allowed={'result.json','depth.npy','mask.npy','points.npy','intrinsics.npy','person-mask.npy'}
        if len(names)!=len(set(names)) or not set(names)<=allowed or sum(x.file_size for x in z.infolist())>400_000_000:raise ValueError('Invalid compute archive')
        meta=json.loads(z.read('result.json'));jobpath=folder/'jobs'/(safe_id(meta['jobId'])+'.json')
        if not jobpath.is_file():raise ValueError('このプロジェクトのジョブではありません')
        job=json.loads(jobpath.read_text('utf-8'))
        if meta.get('schemaVersion')!=2 or meta.get('status')!='success' or meta.get('projectId')!=p['id'] or meta.get('backend')!=job['backend'] or meta.get('inputSha256')!=job['inputSha256']:
            raise ValueError('Compute result/job mismatch')
        scope=job.get('faceScope');current=p['source']
        if scope:current=p.get('faceInput',{}).get('image') if scope=='front' else p.get('directionalFace',{}).get('views',{}).get(scope,{}).get('image')
        if not current or current!=job['sourceAsset'] or hashlib.sha256((folder/current).read_bytes()).hexdigest()!=job['inputSha256']:raise ValueError('推論対象の原画・顔画像が差し替わっています')
        arrays={}
        for name,digest in meta.get('files',{}).items():
            if name not in allowed or name=='result.json':raise ValueError('Invalid array name')
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Array hash mismatch')
            a=np.load(io.BytesIO(raw),allow_pickle=False)
            if not np.issubdtype(a.dtype,np.number) and a.dtype!=bool:raise ValueError('Invalid array type')
            if not np.isfinite(a).all():raise ValueError('Nonfinite array')
            arrays[name]=a
    w,h=job['sourceSize']
    if job['backend']=='mediapipe':
        for key,maximum in [('pose',33),('face',478)]:
            values=meta.get(key,[])
            if values:
                a=np.asarray(values,dtype=float)
                if a.ndim!=2 or a.shape[1]!=5 or len(a)>maximum or not np.isfinite(a).all():raise ValueError('Invalid landmarks')
        hands=meta.get('hands',[])
        if len(hands)>2:raise ValueError('Too many hands')
        for hand in hands:
            a=np.asarray(hand,dtype=float)
            if a.shape!=(21,5) or not np.isfinite(a).all():raise ValueError('Invalid hand landmarks')
        p=landmarks_from_detection(p,meta)
        if 'person-mask.npy' in arrays:
            mask=arrays['person-mask.npy']
            if mask.shape!=(h,w) or mask.min()<0 or mask.max()>1:raise ValueError('Invalid person mask')
            name='assets/person-'+uid()+'.png';Image.fromarray((mask*255).astype('uint8')).save(folder/name)
            p['personMask']={'image':name,'source':'mediapipe'}
        p['landmarks']['provenance']=meta
    else:
        depth=arrays.get('depth.npy');mask=arrays.get('mask.npy');points=arrays.get('points.npy');intrinsics=arrays.get('intrinsics.npy')
        if depth is None or mask is None or depth.shape!=(h,w) or mask.shape!=(h,w) or points is None or points.shape!=(h,w,3) or intrinsics is None or intrinsics.shape!=(3,3):raise ValueError('Geometry dimensions do not match source')
        mask=mask.astype(bool)&(depth>0)
        if not mask.any():raise ValueError('No valid depth')
        lo,hi=np.percentile(depth[mask],[5,95]);paths={}
        for key,array in arrays.items():
            path='assets/'+uid()+'-'+key;np.save(folder/path,array);paths[key]=path
        if scope:
            p.setdefault('faceDepths',{})[scope]={'arrays':paths,'provenance':meta,'sourceAsset':job['sourceAsset'],'inputSha256':job['inputSha256']}
            if p.get('multiViewFace'):p['multiViewFace']['enabled']=False
            return p
        for part in p['parts']:
            if part['locked']:continue
            for vertex in part['mesh']['vertices']:
                x=min(w-1,max(0,int(vertex[0]*w)));y=min(h-1,max(0,int(vertex[1]*h)))
                if mask[y,x]:vertex[2]=part['depth']+float(np.clip((depth[y,x]-lo)/max(hi-lo,1e-6),0,1)*.08)
            part['mesh']['source']='moge3'
        if p.get('headSurface'):p['headSurface']['enabled']=False
        p['depth']={'image':paths['depth.npy'],'arrays':paths,'provenance':meta,'range':[float(lo),float(hi)]}
    return p


def configuration():
    path=ROOT/'data/compute-settings.json'
    settings=json.loads(path.read_text('utf-8')) if path.is_file() else {}
    return {'python':settings.get('python',os.environ.get('AAC2D_COMPUTE_PYTHON',sys.executable)),
            'checkpoint':settings.get('checkpoint','Ruicheng/moge-3-vitl'),'device':settings.get('device','cuda'),
            'resolutionLevel':settings.get('resolutionLevel',5),'refineSteps':settings.get('refineSteps',3)}


def restore_state(store,p):
    folder=store.folder(p['id'])/'jobs';paths=sorted(folder.glob('*-status.json'),key=lambda x:x.stat().st_mtime,reverse=True)
    if not paths:return {}
    state=json.loads(paths[0].read_text('utf-8'));key=safe_id(state['id'])
    state.update(projectId=p['id'],log=str(folder/(key+'.log')),result=str(folder/(key+'-result.zip')))
    if state.get('status') in ('running','queued'):state.update(status='interrupted',error='前回のアプリ終了で中断しました。ジョブを再実行してください。')
    state.pop('process',None)
    return state


def failure_hint(log_text):
    """Explain a native-library failure without mistaking it for model failure."""
    if 'DLL load failed' in log_text and any(value in log_text.lower() for value in ('アクセスが拒否', 'access is denied', 'access denied')):
        return ('推論用DLLへのアクセスが拒否されました。制限された実行環境で起動している場合は、'
                '制作サーバーを終了し、通常のPowerShellから .\\start.ps1 を起動してください。'
                '通常起動でも再発する場合はDLLのアクセス権とセキュリティ製品の記録を確認してください。'
                'このエラーはモデルの推論開始前に発生しています。詳細は下のログを参照してください。')
    return '推論プロセスが失敗しました。下のログを確認してください。'


async def execute_local(store,p,backend,state,settings):
    job,payload=make_job(store,p,backend,settings.get('faceScope'));folder=store.folder(p['id'])/'jobs';path=folder/(job['id']+'.zip');path.write_bytes(payload)
    result=folder/(job['id']+'-result.zip');log=folder/(job['id']+'.log')
    command=[settings['python'],str(ROOT/'scripts'/('run_depth_job.py' if backend=='moge3' else 'run_landmark_job.py')),str(path.resolve()),str(result.resolve())]
    if backend=='moge3':command+=['--checkpoint',settings['checkpoint'],'--device',settings['device'],'--resolution-level',str(settings['resolutionLevel']),'--refine-steps',str(settings['refineSteps'])]
    else:command+=['--models',str((ROOT/'data/models/mediapipe').resolve())]
    state.update(id=job['id'],projectId=p['id'],status='running',backend=backend,faceScope=settings.get('faceScope'),log=str(log),result=str(result))
    def persist():
        (folder/(job['id']+'-status.json')).write_text(json.dumps({k:v for k,v in state.items() if k!='process'}),'utf-8')
    persist()
    try:
        with log.open('wb') as stream:
            proc=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=stream,stderr=asyncio.subprocess.STDOUT)
            state['process']=proc;code=await proc.wait()
        state['status']='ready' if code==0 and result.is_file() else 'failed';state['exitCode']=code
        if state['status']=='failed':state['error']=failure_hint(log.read_text('utf-8',errors='replace'))
    except asyncio.CancelledError:
        if state.get('process') and state['process'].returncode is None:state['process'].terminate();await state['process'].wait()
        state['status']='cancelled'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:state.pop('process',None);persist()


def compute_notebook(backend):
    from scripts.run_depth_job import MOGE_COMMIT,MODEL_ID
    if backend not in ('moge3','mediapipe'):raise ValueError('Unknown backend')
    def cell(kind,text):
        c={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
        if kind=='code':c.update(outputs=[],execution_count=None)
        return c
    workers={name:(ROOT/'scripts'/name).read_text('utf-8') for name in ('run_depth_job.py','run_landmark_job.py')}
    setup=f"subprocess.run([sys.executable,'-m','pip','install','git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT}'],check=True)" if backend=='moge3' else "subprocess.run([sys.executable,'-m','pip','install','mediapipe==0.10.32'],check=True)"
    cells=[cell('markdown',f'# Nagi Studio — {backend}\nジョブZIPをアップロードして実行し、結果ZIPを制作画面に取り込む。MoGe3はGPU推奨。既定の公式重みは初回約1.48GBダウンロード。手持ちの重みパスも指定可能。人物推定はアニメで未検出になる場合があり、検出を捏造しない。'),
      cell('code',f"USE_DRIVE=True\nCHECKPOINT={MODEL_ID!r}\nDEVICE='cuda'\nRESOLUTION_LEVEL=5\nREFINE_STEPS=3\n"),
      cell('code',"import sys,subprocess,json,zipfile,io\nfrom pathlib import Path\nfrom google.colab import files,drive\nif USE_DRIVE:\n    drive.mount('/content/drive')\nROOT=Path('/content/drive/MyDrive/NagiStudio-Compute' if USE_DRIVE else '/content/nagi-compute')\nROOT.mkdir(parents=True,exist_ok=True)\n"+setup),
      cell('code',f"workers={workers!r}\nfor name,source in workers.items():\n    Path('/content',name).write_text(source,encoding='utf-8')\nuploaded=files.upload()\nif len(uploaded)!=1:raise ValueError('ジョブZIPを1つ選択してください')\nJOB=ROOT/'job.zip'\nJOB.write_bytes(next(iter(uploaded.values())))\nsys.path.insert(0,'/content')\nfrom run_depth_job import read_job\njob,_=read_job(JOB,{backend!r})\nRESULT=ROOT/(job['id']+'-result.zip')\n"),
      cell('code',f"command=[sys.executable,'/content/{'run_depth_job.py' if backend=='moge3' else 'run_landmark_job.py'}',str(JOB),str(RESULT)]\n"+("command+=['--checkpoint',CHECKPOINT,'--device',DEVICE,'--resolution-level',str(RESOLUTION_LEVEL),'--refine-steps',str(REFINE_STEPS)]\n" if backend=='moge3' else "command+=['--models',str(ROOT/'models')]\n")+"with (ROOT/(job['id']+'.log')).open('w') as log:\n    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)\n    for line in process.stdout:\n        print(line,end='');log.write(line);log.flush()\n    if process.wait()!=0:raise RuntimeError('推論失敗。保存したログを確認してください')\nfiles.download(str(RESULT))\n")]
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':cells}
