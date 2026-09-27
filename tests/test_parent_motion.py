import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from src.parent_motion import build_surface
from src.parent_motion import DEFAULTS,validate_settings

class ParentMotionTests(unittest.TestCase):
    def test_hair_depth_ignores_background(self):
        from src.parent_motion import build_hair_surfaces
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);n=128;yy,xx=np.mgrid[:n,:n]
            a=np.zeros((n,n,4),np.uint8);a[:,:64,3]=255;Image.fromarray(a).save(folder/'hair.png')
            points=np.zeros((n,n,3));points[:,:,2]=2+xx*.01;np.save(folder/'points.npy',points);np.save(folder/'mask.npy',np.ones((n,n),bool))
            p={'id':'x','depth':{'arrays':{'points.npy':'points.npy','mask.npy':'mask.npy'}},'parts':[{'id':'h','role':'hairFront','depth':1,'image':'hair.png'}]};store=SimpleNamespace(folder=lambda _:folder)
            a=build_hair_surfaces(store,p);points[:,64:,2]=10000;np.save(folder/'points.npy',points)
            self.assertEqual(a,build_hair_surfaces(store,p));self.assertGreater(a['h']['samples'],8)
    def test_runtime_angle_limits(self):
        from src.runtime import Runtime
        r=Runtime();r.project={'parentMotion':{**DEFAULTS,'yawLimit':45,'pitchLimit':30}}
        r.command({'type':'parameters','source':'human','values':{'headX':80,'headY':-70}})
        self.assertEqual(r.inputs['human']['headX'][0],45);self.assertEqual(r.inputs['human']['headY'][0],-30)
        self.assertEqual(r.command({'type':'discover'})['parameters']['headX'],(-45,45))
        with self.assertRaises(ValueError):validate_settings({**DEFAULTS,'yawLimit':46})
    def test_import_rejects_invalid_settings(self):
        for key in ('eyes','body','light','lag','relief'):
            for value in (-1,4,float('nan'),True):
                with self.assertRaises(ValueError):validate_settings({**DEFAULTS,key:value})
        validate_settings(DEFAULTS)
    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):validate_settings({**DEFAULTS,'surface':{'coefficients':[1,2]}})
    def test_amplified_relief(self):
        validate_settings({**DEFAULTS,'relief':3,'hairRelief':3})
        for key in ('relief','hairRelief'):
            with self.assertRaises(ValueError):validate_settings({**DEFAULTS,key:3.01})
    def test_front_hair_depth_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);n=128
            skin=np.full((n,n,4),255,np.uint8);hair=np.zeros_like(skin);hair[:70,:,3]=255
            Image.fromarray(skin).save(folder/'skin.png');Image.fromarray(hair).save(folder/'hair.png')
            yy,xx=np.mgrid[:n,:n];points=np.zeros((n,n,3));points[:,:,2]=2+yy*.001+xx*.0002
            np.save(folder/'points.npy',points);np.save(folder/'mask.npy',np.ones((n,n),bool))
            p={'id':'x','faceInput':{'image':'front.png','alignment':{'rect':[0,0,1,1]}},'faceDepths':{'front':{'sourceAsset':'front.png','arrays':{'points.npy':'points.npy','mask.npy':'mask.npy'}}},'parts':[{'role':'head','tag':'face','depth':0,'image':'skin.png'},{'role':'hairFront','depth':1,'image':'hair.png'}]}
            store=SimpleNamespace(folder=lambda _:folder);a=build_surface(store,p)
            points[:70,:,2]=1000;np.save(folder/'points.npy',points);b=build_surface(store,p)
            self.assertEqual(a,b);self.assertEqual(a['source'],'skin-only-moge3-depth-grid');self.assertEqual(len(a['grid']['residual']),128*128)

    def test_grid_retains_local_bump_removes_plane(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);n=128;yy,xx=np.mgrid[:n,:n]
            Image.fromarray(np.full((n,n,4),255,np.uint8)).save(folder/'skin.png')
            points=np.zeros((n,n,3));points[:,:,2]=2+xx*.001+yy*.002
            np.save(folder/'points.npy',points);np.save(folder/'mask.npy',np.ones((n,n),bool))
            p={'id':'x','faceInput':{'image':'front.png','alignment':{'rect':[0,0,1,1]}},'faceDepths':{'front':{'sourceAsset':'front.png','arrays':{'points.npy':'points.npy','mask.npy':'mask.npy'}}},'parts':[{'role':'head','tag':'face','depth':0,'image':'skin.png'}]}
            store=SimpleNamespace(folder=lambda _:folder)
            plane=build_surface(store,p)
            self.assertLess(np.max(np.abs(plane['grid']['residual'])),1e-10)
            points[:,:,2]-=.05*np.exp(-((xx-64)**2+(yy-75)**2)/40)
            np.save(folder/'points.npy',points);surface=build_surface(store,p)
            self.assertGreater(surface['grid']['residual'][75*128+64],.05)
            self.assertEqual(surface['grid']['confidence'][0],0)
            validate_settings({**DEFAULTS,'surface':surface})
