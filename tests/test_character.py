import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from src.project import Store,validate
from src.pipeline import create_project
from src.character import build_rig,manual_joint,fit_expression,split_arm,build_eye_masks,extend_hidden,landmarks_from_detection
from src.compute import make_job,integrate,compute_notebook
from src.collision import constrain_angle
from src.glb import export_glb
from scripts.run_depth_job import package


POSITIONS={'root':[.5,.7],'neck':[.5,.3],'head':[.5,.2],
 'shoulderLeft':[.3,.35],'elbowLeft':[.22,.55],'wristLeft':[.17,.73],'handLeft':[.16,.8],
 'shoulderRight':[.7,.35],'elbowRight':[.78,.55],'wristRight':[.83,.73],'handRight':[.84,.8]}


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.store=Store(self.tmp.name)
        image=Image.new('RGBA',(96,144));ImageDraw.Draw(image).ellipse((4,3,92,140),fill=(80,130,100,255))
        raw=io.BytesIO();image.save(raw,'PNG');self.p=create_project(self.store,raw.getvalue())
        for name,position in POSITIONS.items():self.p=manual_joint(self.p,name,position)

    def test_face_alignment_reused_for_manual_correspondence(self):
        p=self.p;part=p['parts'][3];part['rect']=[.2,.1,.5,.4]
        p['faceInput']={'alignment':{'rect':part['rect']}}
        p['expressions']=[{'id':'target','image':part['image'],'rect':[0,0,1,1]}]
        a=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]
        fitted=fit_expression(self.store,p,'target',part['id'],a,a,True)
        np.testing.assert_allclose(fitted['parts'][3]['mesh']['morphs']['target'],0,atol=1e-12)
        self.assertEqual(fitted['expressions'][0]['rect'],part['rect'])
        self.assertEqual(p['expressions'][0]['rect'],[0,0,1,1])
        with self.assertRaises(ValueError):fit_expression(self.store,p,'target',part['id'],a,None)

    def test_pins_on_edges_and_inside_preserve_depth_and_skin(self):
        from src.character import insert_control_vertices,displacement
        part={'id':'test','mesh':{'vertices':[[0,0,0],[1,0,1],[1,1,2],[0,1,1]],'triangles':[[0,1,2],[0,2,3]],'weights':[[1,0,0,0]]*4,'joints':[[0,0,0,0]]*4,'morphs':{'old':[[0,0,0]]*4}}}
        p={'expressions':[{'fit':{'partId':'test','uv':[[0,0],[1,0],[1,1],[0,1]]}}]}
        a=np.array([[.5,.5],[.7,.3],[.3,.7],[.2,.2]])
        insert_control_vertices(p,part,a);m=part['mesh'];v=np.array(m['vertices'])
        self.assertEqual(len(v),8);np.testing.assert_allclose(v[:,2],v[:,0]+v[:,1])
        self.assertEqual(len(m['weights']),8);self.assertEqual(len(m['morphs']['old']),8);self.assertEqual(len(p['expressions'][0]['fit']['uv']),8)
        b=a+np.array([[0,.01],[0,.02],[0,-.02],[0,-.01]])
        delta=displacement(v,a,b)
        for source,target in zip(a,b):
            i=np.argmin(np.linalg.norm(v[:,:2]-source,axis=1));np.testing.assert_allclose(v[i,:2]+delta[i],target,atol=1e-10)
        for tri in m['triangles']:
            x,y,z=v[tri,:2];self.assertGreater(np.linalg.det(np.stack([y-x,z-x])),0)

    def test_rig_hierarchy_and_normalized_weights(self):
        p=build_rig(self.p);validate(p)
        self.assertEqual(len(p['rig']['bones']),11)
        for part in p['parts']:
            np.testing.assert_allclose(np.sum(part['mesh']['weights'],axis=1),1)
        missing=copy.deepcopy(self.p);del missing['landmarks']['joints']['elbowLeft']
        with self.assertRaisesRegex(ValueError,'elbowLeft'):build_rig(missing)
        p=manual_joint(p,'elbowLeft',[.2,.6]);self.assertTrue(p['rig']['stale'])

    def test_arm_split_retains_visible_pixels_and_three_segments(self):
        p=split_arm(self.store,self.p,'armLeft','Left')
        self.assertEqual([x['role'] for x in p['parts'][-3:]],['armLeft','forearmLeft','handLeft'])
        for child in p['parts'][-3:]:self.assertEqual(child['mesh']['vertices'],self.p['parts'][1]['mesh']['vertices'])
        original=Image.open(self.store.folder(p['id'])/self.p['parts'][1]['image']).convert('RGBA')
        result=Image.new('RGBA',original.size)
        for part in p['parts'][-3:]:result.alpha_composite(Image.open(self.store.folder(p['id'])/part['image']))
        np.testing.assert_array_equal(np.array(original)[:,:,3],np.array(result)[:,:,3])

    def test_morph_correspondence_translation_and_glb(self):
        p=build_rig(self.p);part=p['parts'][3]
        p['expressions']=[{'id':'smile','name':'Smile','image':part['image'],'rect':[0,0,1,1],'meshVersion':1,'target':'face'}]
        a=[[.2,.1],[.8,.1],[.8,.3],[.2,.3]];b=[[x+.01,y+.02] for x,y in a]
        p=fit_expression(self.store,p,'smile',part['id'],a,b)
        delta=np.asarray(p['parts'][3]['mesh']['morphs']['smile'])
        np.testing.assert_allclose(delta[:,0],.01);np.testing.assert_allclose(delta[:,1],.02)
        validate(p);raw=export_glb(self.store,p);magic,version,size=struct.unpack('<4sII',raw[:12]);self.assertEqual((magic,version,size),(b'glTF',2,len(raw)))
        length=struct.unpack('<I',raw[12:16])[0];g=json.loads(raw[20:20+length])
        self.assertEqual(len(g['skins'][0]['joints']),11);self.assertTrue(g['meshes'][3]['primitives'][0]['targets'])
        binary=raw[28+length:];accessor=g['accessors'][g['skins'][0]['inverseBindMatrices']];view=g['bufferViews'][accessor['bufferView']]
        matrices=np.frombuffer(binary,dtype='<f4',count=11*16,offset=view.get('byteOffset',0)).reshape(11,4,4).transpose(0,2,1)
        for i,bone in enumerate(p['rig']['bones']):
            x,y,z=bone['position'];rest=np.array([(x-.5)*96/144,1-y,z,1]);np.testing.assert_allclose(matrices[i]@rest,[0,0,0,1],atol=1e-7)

    def test_geometry_import_masks_invalid_pixels_and_rejects_source_change(self):
        job,_=make_job(self.store,self.p,'moge3');depth=np.ones((144,96),dtype='float32');mask=np.ones_like(depth,dtype=bool);mask[0]=False
        out=Path(self.tmp.name)/'result.zip';package(job,out,{'inferred':False},{'depth.npy':depth,'mask.npy':mask,'points.npy':np.zeros((144,96,3),'float32'),'intrinsics.npy':np.eye(3,dtype='float32')})
        p=integrate(self.store,self.p,out.read_bytes());self.assertEqual(p['parts'][0]['mesh']['source'],'moge3');self.assertIn('mask.npy',p['depth']['arrays'])
        (self.store.folder(p['id'])/p['source']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'原画'):integrate(self.store,self.p,out.read_bytes())

    def test_detection_missing_is_not_fabricated_and_manual_locks_survive(self):
        p=landmarks_from_detection(self.p,{'pose':[],'face':[],'hands':[]})
        self.assertEqual(p['landmarks']['joints'],self.p['landmarks']['joints'])
        empty=copy.deepcopy(self.p);empty.pop('landmarks');p=landmarks_from_detection(empty,{})
        self.assertEqual(p['landmarks']['joints'],{})

    def test_eye_mask_excludes_protruding_iris_and_synchronizes(self):
        p=copy.deepcopy(self.p);folder=self.store.folder(p['id']);parts=[]
        alphas=[]
        for i,tag in enumerate(('eyewhite','irides','eyelash')):
            im=Image.new('RGBA',(96,144));d=ImageDraw.Draw(im)
            for x in (20,60):d.rectangle((x+i,20,x+10+i,30),fill=(255,255,255,150+i*40))
            path='assets/'+tag+'.png';im.save(folder/path);part=copy.deepcopy(p['parts'][3]);part.update(id=tag,scope='face',tag=tag,image=path,rect=[0,0,1,1]);parts.append(part);alphas.append(np.asarray(im)[:,:,3])
        p['parts']+=parts
        # Split each tag into two visible halves, as in the user's project.
        split=[]
        for part in parts:
            for side in (0,1):
                child=copy.deepcopy(part);child['id']+='-'+str(side)
                im=Image.open(folder/part['image']);a=np.array(im);a[:,48:] = 0 if side==0 else a[:,48:];a[:,:48]=0 if side==1 else a[:,:48]
                child['image']='assets/'+child['id']+'.png';Image.fromarray(a).save(folder/child['image']);split.append(child)
        p['parts']=p['parts'][:-3]+split
        result=build_eye_masks(self.store,p,1.2,True);validate(result)
        alpha=np.asarray(Image.open(folder/result['eyeMask']['image']))[:,:,3]
        self.assertEqual(alpha[25,32],0)  # iris extends beyond the white
        self.assertEqual(alpha[25,25],255)
        self.assertEqual(len(result['eyeMask']['layerIds']),6)
        self.assertEqual(result['eyeMask']['scale'],1.2)
        self.assertEqual(result['parts'][-1]['mesh'],result['parts'][-6]['mesh'])
        self.assertNotIn('eyeMask',p)
        with self.assertRaises(ValueError):build_eye_masks(self.store,p,float('nan'))

    def test_shared_head_surface_camera_and_validation(self):
        from src.character import build_head_surface
        p=copy.deepcopy(self.p);folder=self.store.folder(p['id']);arrays={}
        points=np.zeros((144,96,3),dtype='float32');points[:,:,2]=2
        points[20:35,35:60,2]=1.85
        for name,array in [('points.npy',points),('mask.npy',np.ones((144,96),bool)),('intrinsics.npy',np.array([[1,0,.5],[0,1,.5],[0,0,1]]))]:
            path='assets/'+name;np.save(folder/path,array);arrays[name]=path
        p['depth']={'image':arrays['points.npy'],'arrays':arrays}
        result=build_head_surface(self.store,p);validate(result)
        surface=result['headSurface'];self.assertEqual(len(surface['depth']),49*73)
        self.assertLess(min(surface['depth']),max(surface['depth']))
        self.assertNotIn('headSurface',p)
        invalid=copy.deepcopy(result);invalid['headSurface']['intrinsics'][0][0]=0
        with self.assertRaises(ValueError):validate(invalid)

    def test_directional_fit_is_saved_without_changing_parts(self):
        from src.character import fit_direction
        p=copy.deepcopy(self.p);image=p['parts'][3]['image']
        p['faceInput']={'image':image,'alignment':{'rect':[.2,.02,.6,.4]}}
        p['directionalFace']={'enabled':False,'views':{'right':{'image':image,'original':image,'fitted':False}}}
        a=[[.3,.3],[.7,.3],[.5,.5],[.4,.65],[.6,.65],[.5,.8]];b=[[x+.025,y] for x,y in a]
        result=fit_direction(p,'right',a,b,18);validate(result)
        self.assertTrue(result['directionalFace']['views']['right']['fitted'])
        self.assertEqual(result['parts'],p['parts'])
        self.assertFalse(p['directionalFace']['enabled'])
        self.assertEqual(len(result['directionalFace']['views']['right']['offsets']),49*73)
        with self.assertRaises(ValueError):fit_direction(p,'right',[a[0]]*6,b)
        with self.assertRaises(ValueError):fit_direction(p,'left',a,b)
        with self.assertRaises(ValueError):fit_direction(p,'right',a,[[1-x,y] for x,y in a])

    def test_collision_projects_out_of_obstacle(self):
        obstacles=[{'center':[.47,.7],'radius':.025}]
        value,hit=constrain_angle([.5,.2],[.5,.7],.06,obstacles)
        self.assertTrue(hit);self.assertNotAlmostEqual(value,.06)
        x=.5-math.sin(value)*.5;y=.2+math.cos(value)*.5
        self.assertGreaterEqual(math.hypot(x-.47,y-.7),.025-1e-6)

    def test_notebooks_compile(self):
        for backend in ('moge3','mediapipe'):
            for c in compute_notebook(backend)['cells']:
                if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')

    def test_hidden_extension_only_fills_behind_occluder(self):
        p=copy.deepcopy(self.p);folder=self.store.folder(p['id'])
        back=Image.new('RGBA',(96,144));ImageDraw.Draw(back).rectangle((20,30,40,70),fill=(20,90,140,255))
        front=Image.new('RGBA',(96,144));ImageDraw.Draw(front).rectangle((40,25,55,75),fill=(180,20,50,255))
        back.save(folder/'assets/back.png');front.save(folder/'assets/front.png')
        p['parts']=p['parts'][:2];p['parts'][0].update(image='assets/back.png',depth=0,locked=False)
        p['parts'][1].update(image='assets/front.png',depth=.1,locked=False)
        result=extend_hidden(self.store,p,p['parts'][0]['id'],4)
        array=np.asarray(Image.open(folder/result['parts'][0]['image']));original=np.asarray(back)
        np.testing.assert_array_equal(array[original[:,:,3]>0],original[original[:,:,3]>0])
        self.assertEqual(array[40,44,3],255);self.assertEqual(array[40,45,3],0);self.assertEqual(array[29,39,3],0)
        self.assertEqual(result['parts'][0]['hiddenOriginal'],'assets/back.png')


import math
if __name__=='__main__':unittest.main()
