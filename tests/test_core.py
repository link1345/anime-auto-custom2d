import copy
import io
import json
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from src.pipeline import create_project, image_from_bytes, validate_face_image
from src.project import Store, merge_generated
from src.runtime import Runtime, DEFAULTS
from scripts.run_face_job import prepare_face


def png(size=(128,192)):
    output=io.BytesIO()
    Image.new("RGBA",size,(24,128,90,220)).save(output,"PNG")
    return output.getvalue()


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name)
        self.p=create_project(self.store,png(),"テスト")

    def test_rest_pose_preserves_original_pixels(self):
        folder=self.store.folder(self.p["id"])
        composed=Image.new("RGBA",tuple(self.p["size"]))
        for part in self.p["parts"]:
            composed=Image.alpha_composite(composed,Image.open(folder/part["image"]))
        np.testing.assert_array_equal(np.array(composed),np.array(Image.open(folder/self.p["source"])))

    def test_archive_round_trip_preserves_edits_and_assets(self):
        self.p["parts"][0]["locked"]=True
        self.p["parts"][0]["pivot"]=[.23,.56]
        self.store.save(self.p)
        imported=self.store.import_archive(self.store.export(self.p["id"],True))
        self.assertNotEqual(imported["id"],self.p["id"])
        self.assertEqual(imported["parts"],self.p["parts"])
        self.assertEqual(imported["attachments"],[])

    def test_regeneration_preserves_locks_and_manual_edits(self):
        self.p["parts"][0]["locked"]=True
        self.p["parts"][1]["edits"]={"pivot":[.2,.2]}
        candidate={"parts":copy.deepcopy(self.p["parts"])}
        candidate["parts"][0]["image"]="assets/replaced.png"
        candidate["parts"][1]["edits"]={}
        result=merge_generated(self.p,candidate)
        self.assertEqual(result["parts"][0],self.p["parts"][0])
        self.assertEqual(result["parts"][1]["edits"],{"pivot":[.2,.2]})

    def test_archive_traversal_is_rejected(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,"w") as z:
            z.writestr("project.json",json.dumps(self.p));z.writestr("../evil.txt","bad")
        with self.assertRaises(ValueError):self.store.import_archive(data.getvalue())

    def test_stale_expression_not_silently_used(self):
        self.p["expressions"]=[{"id":"smile","meshVersion":0,"image":"assets/source.png","rect":[0,0,1,1]}]
        self.store.save(self.p)
        self.assertTrue(self.store.load(self.p["id"])["expressions"][0]["stale"])

    def test_stale_model_export_is_blocked(self):
        self.p["stale"]=["mesh"]
        self.store.save(self.p)
        with self.assertRaises(ValueError):self.store.export(self.p["id"],True)

    def test_2160_face_source_is_not_downsampled(self):
        image=image_from_bytes(png((2160,2160)))
        self.assertEqual(image.size,(2160,2160))
        self.assertEqual(validate_face_image(image,True)["width"],2160)

    def test_face_requirement_rejects_small_nonsquare_and_unconfirmed(self):
        for size,confirmed in [((1254,1254),True),((2160,2200),True),((2160,2160),False)]:
            with self.subTest(size=size,confirmed=confirmed):
                with self.assertRaises(ValueError):validate_face_image(Image.new("RGB",size),confirmed)

    def test_face_padding_keeps_source_pixels_at_2176(self):
        original=Image.new('RGBA',(2160,2160),(17,98,167,255))
        padded,offset=prepare_face(original)
        self.assertEqual((padded.size,offset),((2176,2176),8))
        self.assertEqual(padded.getpixel((8,8)),(17,98,167))
        self.assertEqual(padded.getpixel((0,0)),(255,255,255))
        np.testing.assert_array_equal(np.array(padded.crop((8,8,2168,2168))),np.array(original.convert('RGB')))

    def test_import_rejects_external_source_paths(self):
        self.p['source']='../../outside.png'
        with self.assertRaises(ValueError):self.store.save(self.p)


class RuntimeTests(unittest.TestCase):
    def test_mixed_ownership_and_timeout(self):
        r=Runtime()
        r.command({"type":"parameters","source":"human","values":{"headX":12,"mouthOpen":.1}},now=0)
        r.command({"type":"parameters","source":"ai","values":{"headX":-12,"mouthOpen":.8}},now=0)
        for _ in range(60):r.step(now=.5)
        self.assertAlmostEqual(r.values["headX"],12,places=3)
        self.assertAlmostEqual(r.values["mouthOpen"],.8,places=3)
        for _ in range(60):r.step(now=3)
        self.assertAlmostEqual(r.values["headX"],0,places=3)
        self.assertAlmostEqual(r.values["mouthOpen"],0,places=3)

    def test_bad_input_is_atomic(self):
        r=Runtime()
        for invalid in [float('nan'),float('inf'),True,"5"]:
            with self.assertRaises(ValueError):r.command({"type":"parameters","values":{"headX":4,"mouthOpen":invalid}})
            self.assertEqual(r.inputs["ai"],{})

    def test_registered_expression_holds_after_input_timeout(self):
        r=Runtime();r.project={"id":"p","expressions":[{"id":"happy","meshVersion":1}]}
        r.command({"type":"expression","expressionId":"happy","strength":.75,"duration":.25})
        for _ in range(300):r.step(now=50)
        self.assertEqual(r.expression_strength,.75)
        r.command({"type":"release"});self.assertEqual(r.expression,"happy")

    def test_pause_freezes_all_simulation(self):
        r=Runtime();r.step();r.command({"type":"pause"});before=r.snapshot()
        for _ in range(100):r.step()
        after=r.snapshot()
        for key in ('values','hair','time','strength'):self.assertEqual(before[key],after[key])

    def test_physics_consistent_at_30_and_60_hz(self):
        a,b=Runtime(),Runtime()
        for _ in range(600):a.step(1/60,now=0)
        for _ in range(300):b.step(1/30,now=0)
        for role in a.hair:self.assertAlmostEqual(a.hair[role][0],b.hair[role][0],delta=.004)


if __name__=="__main__":unittest.main()
