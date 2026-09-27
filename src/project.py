"""Versioned projects, non-destructive edits, atomic storage and portable archives."""
from __future__ import annotations

import copy
import io
import json
import math
import os
import re
import uuid
import zipfile
from pathlib import Path

SCHEMA = 1


def uid():
    return uuid.uuid4().hex


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError("Invalid identifier")
    return value


def validate(project):
    if project.get('parentMotion'):
        from .parent_motion import validate_settings
        validate_settings(project['parentMotion'])
    if project.get("schemaVersion") != SCHEMA:
        raise ValueError("Unsupported schema version; original file was not changed")
    safe_id(project["id"])
    def asset(value, extensions="png"):
        if not isinstance(value,str) or not re.fullmatch(r"assets/[a-zA-Z0-9_.-]+\.("+extensions+r")", value):
            raise ValueError("Invalid asset path")
    def number(value, low, high):
        if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid numeric project value")
    asset(project["source"])
    if len(project["size"]) != 2:
        raise ValueError("Invalid canvas")
    for value in project["size"]:number(value,1,20000)
    if not 1 <= len(project["parts"]) <= 100:
        raise ValueError("Invalid part count")
    ids = set()
    for part in project["parts"]:
        safe_id(part["id"])
        if part["id"] in ids:
            raise ValueError("Duplicate part id")
        ids.add(part["id"])
        for key in ("image",):
            asset(part[key])
        if part.get('hiddenOriginal'):asset(part['hiddenOriginal'])
        if len(part["pivot"]) != 2:
            raise ValueError("Invalid pivot")
        for value in part["pivot"]:number(value,0,1)
        number(part["depth"],-.5,.5)
        if "visible" in part and type(part["visible"]) is not bool:raise ValueError("Invalid visibility")
        if "rect" in part:
            from .segmentation import rectangle
            rectangle(part["rect"])
        for point in part["polygon"]:
            if len(point)!=2:raise ValueError("Invalid polygon point")
            for value in point:number(value,0,1)
        physics=part["physics"]
        number(physics["stiffness"],1,100)
        number(physics["damping"],1,30)
        number(physics["strength"],0,3)
        mesh=part["mesh"]
        if len(mesh["vertices"])>10000 or len(mesh["triangles"])>20000:raise ValueError("Mesh too large")
        for vertex in mesh["vertices"]:
            if len(vertex)!=3:raise ValueError("Invalid vertex")
            for value in vertex:number(value,-1,2)
        for triangle in mesh["triangles"]:
            if len(triangle)!=3 or any(type(i) is not int or not 0<=i<len(mesh["vertices"]) for i in triangle):raise ValueError("Invalid triangle")
    for source in project.get('expressionSources',[]):
        safe_id(source['id']);asset(source['image']);asset(source['original'])
        from .segmentation import rectangle
        rectangle(source['rect'])
        from scripts.run_face_job import FACE_TAGS
        if not source.get('tags') or any(t not in FACE_TAGS for t in source['tags']):raise ValueError('Invalid expression tags')
    for expression in project.get("expressions", []):
        asset(expression["image"])
        if len(expression["rect"])!=4:raise ValueError("Invalid expression rectangle")
        for value in expression["rect"]:number(value,0,1)
        if expression.get("meshVersion") != project["meshVersion"]:
            expression["stale"] = True
    if project.get("faceInput"):
        asset(project["faceInput"]["image"])
        if project["faceInput"].get("original"):asset(project["faceInput"]["original"])
    if project.get("depth"):asset(project["depth"]["image"],"npy")
    for path in project.get('depth',{}).get('arrays',{}).values():asset(path,'npy')
    if project.get('personMask'):asset(project['personMask']['image'])
    for scope,record in project.get('faceDepths',{}).items():
        if scope not in ('front','left','right'):raise ValueError('Invalid face depth scope')
        asset(record['sourceAsset'])
        for path in record['arrays'].values():asset(path,'npy')
    if project.get('multiViewFace'):
        m=project['multiViewFace'];asset(m['frontImage'])
        if m['cols']!=49 or m['rows']!=73 or len(m['depth'])!=49*73:raise ValueError('Invalid multiview grid')
        for v in m['depth']:number(v,.000001,100000)
        number(m['strength'],0,1);number(m['correction'],0,1)
        def matrix(value):
            if len(value)!=3 or any(len(row)!=3 for row in value):raise ValueError('Invalid camera matrix')
            for row in value:
                for n in row:number(n,-100000,100000)
        matrix(m['intrinsics'])
        if min(m['intrinsics'][0][0],m['intrinsics'][1][1])<=0:raise ValueError('Invalid focal length')
        for side,v in m['views'].items():
            if side not in ('left','right'):raise ValueError('Invalid face view')
            asset(v['image']);matrix(v['rotation']);matrix(v['intrinsics']);number(v['scale'],.0001,1000);number(v['angle'],5,30);number(v['correctionGain'],0,1)
            if len(v['translation'])!=3 or len(v['residualOffsets'])!=49*73:raise ValueError('Invalid view transform')
            for n in v['translation']:number(n,-100000,100000)
            for offset in v['residualOffsets']:
                if len(offset)!=2:raise ValueError('Invalid residual')
                for n in offset:number(n,-2,2)
    for side,view in project.get('directionalFace',{}).get('views',{}).items():
        if side not in ('left','right'):raise ValueError('Invalid direction')
        asset(view['image']);asset(view['original'])
        if view.get('fitted'):
            asset(view['frontImage']);number(view['angle'],5,30)
            if view['cols']!=49 or view['rows']!=73 or len(view['offsets'])!=49*73:raise ValueError('Invalid directional grid')
            for point in view['offsets']:
                if len(point)!=2:raise ValueError('Invalid directional offset')
                for value in point:number(value,-2,2)
            from .segmentation import rectangle
            rectangle(view['rect'])
            for key in ('basePoints','targetPoints'):
                if not 6<=len(view[key])<=512:raise ValueError('Invalid correspondence count')
                for point in view[key]:
                    if len(point)!=2:raise ValueError('Invalid correspondence point')
                    for value in point:number(value,0,1)
            if len(view['basePoints'])!=len(view['targetPoints']):raise ValueError('Mismatched correspondences')
    if project.get('headSurface'):
        surface=project['headSurface'];cols=surface['cols'];rows=surface['rows']
        if type(cols) is not int or type(rows) is not int or not 2<=cols<=100 or not 2<=rows<=100 or len(surface['depth'])!=cols*rows:raise ValueError('Invalid head surface')
        for value in surface['depth']:number(value,.000001,100000)
        number(surface.get('strength',1),0,1.5);number(surface['maxAngle'],1,20)
        if len(surface['pivot'])!=3 or len(surface['intrinsics'])!=3 or any(len(row)!=3 for row in surface['intrinsics']):raise ValueError('Invalid surface camera')
        for value in surface['pivot']:number(value,-100000,100000)
        for row in surface['intrinsics']:
            for value in row:number(value,-100000,100000)
        if surface['intrinsics'][0][0]<=0 or surface['intrinsics'][1][1]<=0 or surface['pivot'][2]<=0:raise ValueError('Invalid surface focal length')
    if project.get('eyeMask'):
        eye=project['eyeMask'];asset(eye['image']);asset(eye['irisImage'])
        from .segmentation import rectangle
        rectangle(eye['rect'])
        number(eye.get('scale',1),.5,1.5)
        if len(eye['eyes'])!=2:raise ValueError('Expected two eye regions')
        for region in eye['eyes']:
            if len(region['bbox'])!=4:raise ValueError('Invalid eye bbox')
            for value in region['bbox']:number(value,0,1)
    rig=project.get('rig')
    for name,joint in project.get('landmarks',{}).get('joints',{}).items():
        if len(joint['position'])!=2:raise ValueError('Invalid landmark')
        for value in joint['position']:number(value,0,1)
    if rig:
        bones=rig['bones']
        if not 1<=len(bones)<=100:raise ValueError('Invalid bone count')
        for i,bone in enumerate(bones):
            if type(bone['parent']) is not int or not -1<=bone['parent']<i:raise ValueError('Invalid bone hierarchy')
            if len(bone['position'])!=3:raise ValueError('Invalid bone position')
            for value in bone['position']:number(value,-1,2)
        for collider in rig.get('colliders',[]):
            if len(collider['center'])!=2:raise ValueError('Invalid collider')
            for value in collider['center']:number(value,0,1)
            number(collider['radius'],.001,1)
    for part in project['parts']:
        mesh=part['mesh'];count=len(mesh['vertices'])
        if 'weights' in mesh or 'joints' in mesh:
            if not rig or len(mesh['weights'])!=count or len(mesh['joints'])!=count:raise ValueError('Skin/mesh mismatch')
            for weights,joints in zip(mesh['weights'],mesh['joints']):
                if len(weights)!=4 or len(joints)!=4:raise ValueError('Expected four skin influences')
                for value in weights:number(value,0,1)
                if abs(sum(weights)-1)>1e-5 or any(type(j) is not int or not 0<=j<len(rig['bones']) for j in joints):raise ValueError('Invalid skin weights/joints')
        for morph in mesh.get('morphs',{}).values():
            if len(morph)!=count:raise ValueError('Morph/mesh mismatch')
            for delta in morph:
                if len(delta)!=3:raise ValueError('Invalid morph delta')
                for value in delta:number(value,-2,2)
    for expression in project.get('expressions',[]):
        if expression.get('fit'):
            fit=expression['fit'];part=next((x for x in project['parts'] if x['id']==fit['partId']),None)
            if not part or len(fit['uv'])!=len(part['mesh']['vertices']):raise ValueError('Expression fit/mesh mismatch')
            for uv in fit['uv']:
                if len(uv)!=2:raise ValueError('Invalid UV')
                for value in uv:number(value,-10,10)
    effects=project.get("faceEffects",[])
    if len(effects)>24:raise ValueError("Too many face effects")
    for effect in effects:
        safe_id(effect["id"])
        if effect["kind"] not in ("blush","star","heart","image"):raise ValueError("Unknown effect")
        number(effect["opacity"],0,1);number(effect["scale"],0,3)
        if type(effect["visible"]) is not bool:raise ValueError("Invalid visibility")
        if effect["kind"]=="image":
            asset(effect["image"])
            from .segmentation import rectangle
            rectangle(effect["rect"])
    return project


def merge_generated(project, candidate):
    """Only targeted, unlocked parts are changed; preserve stable IDs and manual edits."""
    updated = copy.deepcopy(project)
    affected = []
    for result in candidate["parts"]:
        for index, old in enumerate(updated["parts"]):
            if old["id"] == result["id"] and not old.get("locked"):
                replacement = copy.deepcopy(result)
                replacement["edits"] = old.get("edits", {})
                for key, value in replacement["edits"].items():
                    if key in ("name", "pivot", "depth", "physics", "polygon"):
                        replacement[key] = copy.deepcopy(value)
                updated["parts"][index] = replacement
                affected.append(old["id"])
    updated["revision"] += 1
    updated["stale"] = [f"{p}:mesh,rig" for p in affected]
    return validate(updated)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, project_id):
        return self.root / safe_id(project_id)

    def save(self, project, checkpoint=False):
        validate(project)
        folder = self.folder(project["id"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "project.json"
        if checkpoint and path.exists():
            target = folder / "checkpoints"
            target.mkdir(exist_ok=True)
            (target / f"{uid()}.json").write_bytes(path.read_bytes())
        temp = folder / f"{uid()}.tmp"
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(project, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    def load(self, project_id):
        return validate(json.loads((self.folder(project_id) / "project.json").read_text("utf-8")))

    def list(self):
        projects = []
        for path in self.root.glob("*/project.json"):
            try:
                p = self.load(path.parent.name)
                projects.append({"id": p["id"], "name": p["name"], "revision": p["revision"]})
            except (ValueError, KeyError):
                continue
        return projects

    def export(self, project_id, runtime=False):
        project = self.load(project_id)
        if runtime and (project.get("stale") or project.get('rig',{}).get('stale')):
            raise ValueError("Regenerate stale mesh/rig before export")
        if runtime:
            project.pop("history", None)
            project.pop("candidates", None)
            project.pop("faceInput", None)
        required = {project["source"]}
        required.update(p["image"] for p in project["parts"])
        required.update(e["image"] for e in project.get("expressions", []))
        for e in project.get('expressionSources',[]):required.update([e['image'],e['original']])
        required.update(e["image"] for e in project.get("faceEffects", []) if e.get("image"))
        if project.get("depth"):required.add(project["depth"]["image"])
        required.update(project.get('depth',{}).get('arrays',{}).values())
        if project.get('eyeMask'):required.update([project['eyeMask']['image'],project['eyeMask']['irisImage']])
        if project.get('personMask'):required.add(project['personMask']['image'])
        for record in project.get('faceDepths',{}).values():required.update([record['sourceAsset'],*record['arrays'].values()])
        if project.get('multiViewFace'):
            required.add(project['multiViewFace']['frontImage'])
            for view in project['multiViewFace']['views'].values():required.add(view['image'])
        for view in project.get('directionalFace',{}).get('views',{}).values():
            required.update([view['image'],view['original']])
            if view.get('frontImage'):required.add(view['frontImage'])
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", json.dumps(project, ensure_ascii=False))
            for path in (self.folder(project_id) / "assets").glob("*"):
                if path.is_file() and (not runtime or f"assets/{path.name}" in required):
                    archive.write(path, f"assets/{path.name}")
        return data.getvalue()

    def import_archive(self, payload):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 1000 or sum(x.file_size for x in entries) > 150_000_000:
                raise ValueError("Archive is too large")
            for entry in entries:
                if entry.filename != "project.json" and not re.fullmatch(r"assets/[a-zA-Z0-9_.-]+\.(png|npy|glb)", entry.filename):
                    raise ValueError("Unsafe archive path")
            project = validate(json.loads(archive.read("project.json")))
            project["id"] = uid()
            folder = self.folder(project["id"])
            (folder / "assets").mkdir(parents=True)
            for entry in entries:
                if entry.filename.startswith("assets/"):
                    (folder / entry.filename).write_bytes(archive.read(entry))
            for part in project["parts"]:
                if not (folder / part["image"]).exists():
                    raise ValueError("Missing part asset")
            self.save(project)
            return project
