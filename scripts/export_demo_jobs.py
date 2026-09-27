"""Prepare downloadable Colab jobs for the active project without running GPU AI."""
import argparse
import asyncio
from pathlib import Path
from aiohttp import ClientSession, FormData


async def export(base,face,output):
    output.mkdir(parents=True,exist_ok=True)
    async with ClientSession() as client:
        async with client.get(base+'/api/project') as response:
            response.raise_for_status();project=await response.json()
        if not project.get('faceInput'):
            form=FormData();form.add_field('image',face.read_bytes(),filename=face.name,content_type='image/png')
            form.add_field('faceFillsFrame','true');form.add_field('upscale','true');form.add_field('size','2160')
            async with client.post(base+'/api/face',data=form) as response:
                if response.status!=200:raise RuntimeError(await response.text())
        for route,name in [('face-job?scope=face','see-through-face-job.zip'),('face-job?scope=body','see-through-body-job.zip'),('colab','Nagi_SeeThrough.ipynb')]:
            async with client.get(base+'/api/'+route) as response:
                response.raise_for_status();(output/name).write_bytes(await response.read())
                print(output/name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--face',type=Path,default=Path('assets/nagi-face-source.png'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/colab'))
    args=parser.parse_args();asyncio.run(export(args.url,args.face,args.output))
