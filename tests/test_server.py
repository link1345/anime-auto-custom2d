import asyncio
import io
import tempfile
import unittest
import zipfile
import json
from pathlib import Path

from aiohttp import web, ClientSession, FormData
from PIL import Image

from src.server import create_app
from scripts.run_face_job import read_job, prepare_input, package_result


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.app=create_app(self.tmp.name,8876)
        self.runner=web.AppRunner(self.app)
        await self.runner.setup()
        await web.TCPSite(self.runner,'127.0.0.1',8876).start()
        self.client=ClientSession()
        self.base='http://127.0.0.1:8876'

    async def asyncTearDown(self):
        await self.client.close();await self.runner.cleanup();self.tmp.cleanup()

    async def test_origin_and_host_rejection(self):
        for headers in [{'Origin':'https://evil.example'},{'Host':'evil.example:8876'}]:
            async with self.client.get(self.base+'/api/project',headers=headers) as r:self.assertEqual(r.status,403)

    async def test_edit_undo_checkpoint_and_export(self):
        async with self.client.get(self.base+'/api/project') as r:p=await r.json()
        key=p['parts'][0]['id'];pivot=p['parts'][0]['pivot']
        async with self.client.patch(self.base+'/api/parts/'+key,json={'pivot':[.2,.4]}) as r:self.assertEqual(r.status,200)
        async with self.client.post(self.base+'/api/history/undo') as r:
            self.assertEqual((await r.json())['parts'][0]['pivot'],pivot)
        async with self.client.post(self.base+'/api/history/checkpoint') as r:self.assertEqual(r.status,200)
        async with self.client.get(self.base+'/api/export/model') as r:self.assertTrue((await r.read()).startswith(b'PK'))

    async def test_face_upload_rejects_undersized_asset(self):
        image=io.BytesIO();Image.new('RGB',(1024,1024)).save(image,'PNG')
        data=FormData();data.add_field('image',image.getvalue(),filename='face.png',content_type='image/png');data.add_field('faceFillsFrame','true')
        async with self.client.post(self.base+'/api/face',data=data) as r:
            self.assertEqual(r.status,400);self.assertIn('2160', (await r.json())['error'])

    async def test_face_job_retains_2160_pixels(self):
        image=io.BytesIO();Image.new('RGB',(2160,2160),(23,94,167)).save(image,'PNG')
        data=FormData();data.add_field('image',image.getvalue(),filename='face.png',content_type='image/png');data.add_field('faceFillsFrame','true')
        async with self.client.post(self.base+'/api/face',data=data) as r:
            self.assertEqual(r.status,200)
        async with self.client.get(self.base+'/api/face-job') as r:
            self.assertEqual(r.status,200)
            with zipfile.ZipFile(io.BytesIO(await r.read())) as z:
                self.assertEqual(Image.open(io.BytesIO(z.read('input.png'))).size,(2160,2160))
                self.assertTrue(json.loads(z.read('job.json'))['validation']['faceFillsFrame'])

    async def test_websocket_discovery_errors_and_shared_frames(self):
        a=await self.client.ws_connect(self.base+'/ws');b=await self.client.ws_connect(self.base+'/ws')
        await a.receive_json();await b.receive_json()
        await a.send_json({'id':'discover-test','type':'discover'})
        response=None
        for _ in range(20):
            packet=await a.receive_json(timeout=2)
            if packet.get('id')=='discover-test':response=packet;break
        self.assertTrue(response['ok']);self.assertIn('headX',response['result']['parameters'])
        await a.send_str('not json')
        for _ in range(20):
            packet=await a.receive_json(timeout=2)
            if packet.get('type')=='response':break
        self.assertFalse(packet['ok'])
        frames=[]
        for socket in (a,b):
            for _ in range(20):
                packet=await socket.receive_json(timeout=2)
                if packet['type']=='frame':frames.append(packet);break
        self.assertEqual(frames[0]['projectId'],frames[1]['projectId'])
        await a.close();await b.close()

    async def test_segmentation_persisted_candidate_adopt_and_stale_revision(self):
        async with self.client.get(self.base+'/api/face-job?scope=body') as response:
            job_path=Path(self.tmp.name)/'body-job.zip';job_path.write_bytes(await response.read())
        job,image=read_job(job_path);padded,offset=prepare_input(image,'body')
        layers=Path(self.tmp.name)/'fixture';layers.mkdir()
        # Synthetic nonempty layer tests transport/adoption, not inference quality.
        Image.new('RGBA',padded.size,(40,100,120,255)).save(layers/'front hair.png')
        result=Path(self.tmp.name)/'result.zip'
        package_result(job,image,padded.width,offset,layers,result,{'fixture':True})
        form=FormData();form.add_field('file',result.read_bytes(),filename='result.zip')
        async with self.client.post(self.base+'/api/segmentation/result',data=form) as response:
            self.assertEqual(response.status,200);candidate=await response.json()
        self.app['candidates'].clear()  # adoption must also work from disk after restart
        async with self.client.get(self.base+'/api/segmentation/candidates') as response:
            self.assertEqual((await response.json())[0]['id'],candidate['id'])
        async with self.client.get(self.base+'/api/project') as response:
            self.assertEqual(len((await response.json())['parts']),6)
        async with self.client.post(self.base+'/api/adopt',json={'id':candidate['id']}) as response:
            self.assertEqual(response.status,200);adopted=await response.json()
            self.assertEqual(len(adopted['parts']),7)
            self.assertEqual(adopted['parts'][-1]['tag'],'front hair')
        stored=self.app['store'].load(adopted['id'])
        self.assertEqual(stored['segmentation']['body']['jobId'],job['id'])
        async with self.client.post(self.base+'/api/adopt',json={'id':candidate['id']}) as response:
            self.assertEqual(response.status,400)
        async with self.client.post(self.base+'/api/history/undo') as response:
            self.assertEqual(len((await response.json())['parts']),6)

    async def test_face_effects_and_custom_png_survive_model_export(self):
        for kind in ('blush','star','heart'):
            async with self.client.post(self.base+'/api/face/effects',json={'kind':kind}) as response:
                self.assertEqual(response.status,200)
        image=io.BytesIO();Image.new('RGBA',(16,16),(255,20,60,150)).save(image,'PNG')
        data=FormData();data.add_field('image',image.getvalue(),filename='effect.png',content_type='image/png')
        data.add_field('rect','[0.3,0.2,0.2,0.1]');data.add_field('name','custom')
        async with self.client.post(self.base+'/api/face/effect-image',data=data) as response:
            self.assertEqual(response.status,200);project=await response.json()
        self.assertEqual(len(project['faceEffects']),4)
        eid=project['faceEffects'][0]['id']
        async with self.client.post(self.base+'/api/face/effects',json={'id':eid,'opacity':.2,'visible':False}) as response:
            self.assertEqual(response.status,200)
        async with self.client.get(self.base+'/api/export/model') as response:
            self.assertEqual(response.status,200);archive=await response.read()
        restored=self.app['store'].import_archive(archive)
        self.assertFalse(restored['faceEffects'][0]['visible'])
        self.assertEqual(restored['faceEffects'][0]['opacity'],.2)
        self.assertTrue((self.app['store'].folder(restored['id'])/restored['faceEffects'][-1]['image']).is_file())

    async def test_character_rig_glb_and_colab_routes(self):
        from tests.test_character import POSITIONS
        for name,position in POSITIONS.items():
            async with self.client.post(self.base+'/api/character',json={'operation':'joint','name':name,'position':position}) as r:self.assertEqual(r.status,200)
        async with self.client.post(self.base+'/api/character',json={'operation':'rig'}) as r:
            self.assertEqual(r.status,200);p=await r.json();self.assertEqual(len(p['rig']['bones']),11)
        async with self.client.get(self.base+'/api/character.glb') as r:
            self.assertEqual(r.status,200);self.assertTrue((await r.read()).startswith(b'glTF'))
        async with self.client.get(self.base+'/api/compute/notebook/moge3') as r:
            self.assertEqual(r.status,200);book=await r.json()
            for cell in book['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'colab','exec')
        async with self.client.post(self.base+'/api/character',json={'operation':'joint','name':'elbowLeft','position':[.2,.5]}) as r:self.assertEqual(r.status,200)
        async with self.client.get(self.base+'/api/character.glb') as r:self.assertEqual(r.status,400)




    async def test_named_expression_source_overwrite_and_undo(self):
        async def upload(name,color):
            raw=io.BytesIO();Image.new('RGBA',(2160,2160),color).save(raw,'PNG')
            form=FormData();form.add_field('image',raw.getvalue(),filename='face.png',content_type='image/png');form.add_field('name',name);form.add_field('tags','["mouth"]')
            async with self.client.post(self.base+'/api/expression-source',data=form) as r:
                self.assertEqual(r.status,200);return await r.json()
        first=await upload('test','red');old=first['expressionSources'][-1]
        second=await upload(' test ','blue');new=second['expressionSources'][-1]
        self.assertEqual(len(second['expressionSources']),1);self.assertEqual(old['id'],new['id']);self.assertNotEqual(old['image'],new['image'])
        async with self.client.post(self.base+'/api/history/undo') as r:
            self.assertEqual((await r.json())['expressionSources'][-1]['image'],old['image'])


if __name__=='__main__':unittest.main()
