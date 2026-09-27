import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from scripts.run_face_job import FACE_TAGS, BODY_TAGS, package_result, prepare_input, read_job
from src.colab import notebook
from src.media import ffmpeg_path, upscale_face
from src.pipeline import create_project
from src.project import Store
from src.segmentation import make_job, import_result, split_part


def png(image):
    out=io.BytesIO();image.save(out,'PNG');return out.getvalue()


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=Store(self.root/'projects')
        self.project=create_project(self.store,png(Image.new('RGBA',(64,96),(50,80,100,220))))
        face=Image.new('RGBA',(2160,2160),(160,120,100,255))
        self.face_bytes=png(face)
        folder=self.store.folder(self.project['id']);(folder/'assets/face.png').write_bytes(self.face_bytes)
        self.project['faceInput']={'image':'assets/face.png','validation':{'width':2160,'height':2160,'faceFillsFrame':True},'alignment':{'rect':[.25,.05,.5,.34],'confirmed':True}}
        self.store.save(self.project)

    def result(self,scope='face',extra=False):
        job,payload=make_job(self.store,self.project,scope)
        jobfile=self.root/'job.zip';jobfile.write_bytes(payload)
        loaded,image=read_job(jobfile)
        padded,offset=prepare_input(image,scope)
        layers=self.root/'layers';layers.mkdir(exist_ok=True)
        for index,tag in enumerate(FACE_TAGS if scope=='face' else BODY_TAGS):
            im=Image.new('RGBA',padded.size)
            draw=ImageDraw.Draw(im);x,y=offset
            draw.rectangle((x+image.width//4,y+image.height//4,x+image.width*3//4,y+image.height*3//4),fill=(index*20,100,160,255))
            im.save(layers/f'{tag}.png')
        if extra:Image.new('RGBA',padded.size,'red').save(layers/'front hair.png')
        resultfile=self.root/'result.zip'
        meta=package_result(loaded,image,padded.width,offset,layers,resultfile,{'fixture':True})
        return job,resultfile.read_bytes(),meta

    def test_face_exports_only_eight_requested_tags(self):
        job,payload,meta=self.result(extra=True)
        self.assertEqual(set(x['tag'] for x in meta['layers']),set(FACE_TAGS))
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            self.assertFalse(any('hair' in n for n in z.namelist()))
            self.assertEqual(Image.open(io.BytesIO(z.read('layers/face.png'))).size,(2160,2160))

    def test_face_import_is_candidate_and_preserves_locked_parts_and_effects(self):
        _,payload,_=self.result()
        candidate=import_result(self.store,self.project,payload)
        self.assertEqual(len(candidate['project']['parts']),len(self.project['parts'])+8)
        self.assertEqual(len(self.store.load(self.project['id'])['parts']),6)
        adopted=candidate['project'];part=next(p for p in adopted['parts'] if p.get('tag')=='mouth')
        part['locked']=True;part['name']='手修正・口'
        adopted['faceEffects']=[{'id':'effect','kind':'heart','opacity':.5,'scale':1,'visible':True}]
        next_candidate=import_result(self.store,adopted,payload)
        self.assertEqual(next(p for p in next_candidate['project']['parts'] if p.get('tag')=='mouth'),part)
        self.assertEqual(next_candidate['project']['faceEffects'],adopted['faceEffects'])

    def test_input_replacement_rejects_stale_result(self):
        _,payload,_=self.result()
        (self.store.folder(self.project['id'])/'assets/face.png').write_bytes(png(Image.new('RGBA',(2160,2160),'blue')))
        with self.assertRaisesRegex(ValueError,'入力素材'):import_result(self.store,self.project,payload)

    def test_import_rejects_nonface_tag_even_with_valid_hash(self):
        _,payload,_=self.result()
        output=io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload)) as source,zipfile.ZipFile(output,'w') as target:
            for name in source.namelist():
                data=source.read(name)
                if name=='result.json':
                    meta=json.loads(data);meta['layers'][0]['tag']='front hair';data=json.dumps(meta).encode()
                target.writestr(name,data)
        with self.assertRaisesRegex(ValueError,'対象外'):import_result(self.store,self.project,output.getvalue())

    def test_body_job_unpads_to_original_and_has_expected_roles(self):
        _,payload,_=self.result('body')
        candidate=import_result(self.store,self.project,payload)
        roles={p['role'] for p in candidate['parts']}
        self.assertTrue({'neck','chest','arms','skirt','legs','hairFront','hairBack'}<=roles)
        self.assertEqual(candidate['parts'][0]['rect'],[0,0,1,1])

    def test_split_is_exact_and_non_destructive(self):
        part=self.project['parts'][0];original=Image.open(self.store.folder(self.project['id'])/part['image']).convert('RGBA')
        result=split_part(self.store,self.project,part['id'],'x',.5)
        self.assertTrue(result['parts'][0]['locked']);self.assertFalse(result['parts'][0]['visible'])
        a,b=result['parts'][-2:]
        combined=Image.alpha_composite(Image.open(self.store.folder(self.project['id'])/a['image']),Image.open(self.store.folder(self.project['id'])/b['image']))
        # RGB under alpha==0 is irrelevant; all visible pixels and alpha are retained.
        before=np.array(original);after=np.array(combined)
        np.testing.assert_array_equal(before[:,:,3],after[:,:,3])
        np.testing.assert_array_equal(before[before[:,:,3]>0],after[after[:,:,3]>0])
        self.assertEqual(self.project['parts'][0],part)

    def test_notebook_cells_compile_and_embed_worker(self):
        book=notebook()
        for index,cell in enumerate(book['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell-{index}','exec')
        content=json.dumps(book)
        self.assertIn('GROUP_OFFLOAD',content);self.assertIn('inputSha256',content)
        self.assertNotIn('extractall',content)

    def test_real_ffmpeg_upscales_preserving_original_and_alpha(self):
        try:ffmpeg_path()
        except ValueError:self.skipTest('FFmpeg not visible in this environment')
        source=self.root/'small.png';target=self.root/'large.png'
        im=Image.new('RGBA',(128,128));ImageDraw.Draw(im).ellipse((20,20,100,100),fill=(255,20,30,255));im.save(source)
        before=source.read_bytes();metadata=upscale_face(source,target,2160)
        self.assertTrue(metadata['upscaledByApp']);self.assertEqual(source.read_bytes(),before)
        with Image.open(target) as out:
            self.assertEqual(out.size,(2160,2160));self.assertEqual(out.getpixel((0,0))[3],0)


if __name__=='__main__':unittest.main()
