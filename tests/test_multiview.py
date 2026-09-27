import copy,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from src.multiview import similarity,fit_multiview
from src.compute import make_job
from src.project import Store,validate
from src.pipeline import create_project

class MultiViewTests(unittest.TestCase):
 def test_similarity_recovers_camera_transform(self):
  rng=np.random.default_rng(4);a=rng.normal(size=(20,3));t=np.array([.4,-.2,1]);angle=.3
  r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
  rr,s,tt,error=similarity(a,a@r.T*1.3+t)
  np.testing.assert_allclose(rr,r,atol=1e-10);self.assertAlmostEqual(s,1.3);np.testing.assert_allclose(tt,t,atol=1e-10)
  self.assertLess(error.max(),1e-10)
 def test_face_jobs_and_fit_do_not_replace_body_geometry(self):
  with tempfile.TemporaryDirectory() as directory:
   store=Store(directory);buf=io.BytesIO();Image.new('RGBA',(96,144),'white').save(buf,'PNG');p=create_project(store,buf.getvalue());folder=store.folder(p['id'])
   Image.new('RGBA',(2160,2160),'white').save(folder/'assets/front.png')
   p['faceInput']={'image':'assets/front.png','alignment':{'rect':[.2,.02,.6,.4]}}
   pts=[[.3,.3],[.7,.3],[.5,.5],[.35,.65],[.65,.65],[.5,.8]]
   p['directionalFace']={'enabled':True,'views':{'right':{'image':'assets/front.png','original':'assets/front.png','frontImage':'assets/front.png','rect':[.2,.02,.6,.4],'angle':18,'basePoints':pts,'targetPoints':pts,'fitted':True,'cols':49,'rows':73,'offsets':[[0,0]]*(49*73)}}}
   job,payload=make_job(store,p,'moge3','front');self.assertEqual(job['sourceSize'],[2160,2160]);self.assertEqual(job['sourceAsset'],'assets/front.png')
   with zipfile.ZipFile(io.BytesIO(payload)) as z:self.assertEqual(hashlib.sha256(z.read('source.png')).hexdigest(),job['inputSha256'])
   size=80;y,x=np.mgrid[0:size,0:size]/(size-1);depth=2-.15*np.exp(-((x-.5)**2+(y-.5)**2)/.1)
   points=np.stack([(x-.5)*depth,(y-.5)*depth,depth],axis=2);arrays={}
   for name,array in [('points.npy',points),('mask.npy',np.ones((size,size),bool)),('intrinsics.npy',np.array([[1,0,.5],[0,1,.5],[0,0,1]]))]:
    path='assets/'+name;np.save(folder/path,array);arrays[name]=path
   record={'sourceAsset':'assets/front.png','inputSha256':job['inputSha256'],'arrays':arrays}
   p['faceDepths']={'front':copy.deepcopy(record),'right':copy.deepcopy(record)}
   result=fit_multiview(store,p);validate(result);self.assertEqual(result['parts'],p['parts'])
   self.assertTrue(result['multiViewFace']['enabled']);self.assertFalse(result['directionalFace']['enabled'])
   self.assertLess(result['multiViewFace']['views']['right']['rmseAfterPixels'],1)
   self.assertNotIn('multiViewFace',p)
   (folder/'assets/front.png').write_bytes(b'replaced')
   with self.assertRaisesRegex(ValueError,'古く'):fit_multiview(store,p)
