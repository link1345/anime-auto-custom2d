"""MoGe-3 local/Colab worker with explicit checkpoint and validity masks."""
import argparse
import io
import json
import time
from pathlib import Path
import zipfile
import hashlib
import numpy as np
from PIL import Image

MOGE_COMMIT = '74fbce054ebed49800de42d0ad0e83495065719a'
MODEL_ID = 'Ruicheng/moge-3-vitl'
MODEL_REVISION = '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be'


def read_job(path, backend='moge3'):
    with zipfile.ZipFile(path) as z:
        if len(z.infolist())!=2 or set(z.namelist())!={'job.json','source.png'} or sum(i.file_size for i in z.infolist())>100_000_000:
            raise ValueError('Invalid job archive')
        job=json.loads(z.read('job.json'));raw=z.read('source.png')
    if job.get('schemaVersion')!=2 or job.get('backend')!=backend or hashlib.sha256(raw).hexdigest()!=job.get('inputSha256'):
        raise ValueError('Job schema/backend/input hash mismatch')
    im=Image.open(io.BytesIO(raw))
    if im.width*im.height>20_000_000:raise ValueError('Image too large')
    return job,im.convert('RGBA')


def package(job, output_path, metadata, arrays):
    meta={'schemaVersion':2,'status':'success','backend':job['backend'],'jobId':job['id'],
          'projectId':job['projectId'],'inputSha256':job['inputSha256'],**metadata,'files':{}}
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,array in arrays.items():
            b=io.BytesIO();np.save(b,array,allow_pickle=False);raw=b.getvalue()
            z.writestr(name,raw);meta['files'][name]=hashlib.sha256(raw).hexdigest()
        z.writestr('result.json',json.dumps(meta,ensure_ascii=False))
    temp.replace(out)


def run(input_path, output_path, checkpoint=None, device='cuda', resolution_level=5, refine_steps=3):
    job,image=read_job(input_path)
    official=not checkpoint or checkpoint == MODEL_ID
    if official:
        print('Fetching pinned official MoGe-3 checkpoint...',flush=True)
        from huggingface_hub import hf_hub_download
        checkpoint=hf_hub_download(MODEL_ID,'model.pt',revision=MODEL_REVISION)
    if not Path(checkpoint).is_file():raise ValueError('Checkpoint file not found')
    import torch
    from moge.model.v3 import MoGeModel
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Select --device cpu or Colab GPU.')
    start=time.monotonic()
    print('Loading MoGe-3 on '+device,flush=True)
    background=Image.new('RGBA',image.size,'white');background.alpha_composite(image)
    model=MoGeModel.from_pretrained(checkpoint).to(device).eval()
    tensor=torch.tensor(np.asarray(background.convert('RGB')).copy(),dtype=torch.float32,device=device).permute(2,0,1)/255
    with torch.inference_mode():
        print(f'Inferring {image.width}x{image.height}, level={resolution_level}, refinement={refine_steps}',flush=True)
        result=model.infer(tensor,resolution_level=resolution_level,refine_steps=refine_steps,use_fp16=False)
    arrays={k:result[k].detach().cpu().numpy() for k in ('depth','mask','points','intrinsics')}
    valid=arrays['mask'].astype(bool)&np.isfinite(arrays['depth'])&(arrays['depth']>0)&(np.asarray(image)[:,:,3]>0)
    if valid.shape!=(image.height,image.width) or not valid.any():raise ValueError('No valid foreground depth')
    arrays['mask']=valid
    arrays['depth']=np.where(valid,arrays['depth'],0).astype('float32')
    arrays['points']=np.where(valid[:,:,None],arrays['points'],0).astype('float32')
    for key,value in arrays.items():
        if not np.isfinite(value).all():raise ValueError('Non-finite '+key)
    package(job,output_path,{'inferred':True,'sourceSize':list(image.size),'checkpointName':Path(checkpoint).name,
        'torch':torch.__version__,'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
        'elapsedSeconds':time.monotonic()-start,'resolutionLevel':resolution_level,'refineSteps':refine_steps,'modelRevision':MODEL_REVISION if official else None},
        {k+'.npy':v for k,v in arrays.items()})
    print('Result saved: '+str(output_path),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('result');p.add_argument('--checkpoint',default=MODEL_ID)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--resolution-level',type=int,choices=range(10),default=5)
    p.add_argument('--refine-steps',type=int,choices=range(11),default=3)
    a=p.parse_args();run(a.job,a.result,a.checkpoint,a.device,a.resolution_level,a.refine_steps)
