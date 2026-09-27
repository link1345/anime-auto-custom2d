"""See-through jobs, strict imports, non-destructive adoption and layer splitting."""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from PIL import Image

from scripts.run_face_job import FACE_TAGS, BODY_TAGS, REPO_COMMIT
from .pipeline import image_from_bytes, mesh_for
from .project import uid, safe_id, validate

FACE_ORDER = ["ears", "face", "eyewhite", "irides", "nose", "mouth", "eyebrow", "eyelash"]
BODY_ORDER = ["back hair", "legwear", "footwear", "neck", "topwear", "bottomwear", "handwear", "front hair"]
ROLES = {"back hair":"hairBack", "front hair":"hairFront", "neck":"neck", "topwear":"chest",
         "handwear":"arms", "bottomwear":"skirt", "legwear":"legs", "footwear":"feet"}


def rectangle(value):
    if not isinstance(value,list) or len(value)!=4 or any(type(x) not in (int,float) or not np.isfinite(x) for x in value):
        raise ValueError("矩形は x,y,幅,高さの4数値です")
    x,y,w,h=value
    if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:
        raise ValueError("矩形はキャンバス内の0〜1で指定してください")
    return value


def make_job(store,project,scope,settings=None):
    if scope not in ("face","body"):raise ValueError("Invalid segmentation scope")
    if scope=="face" and not project.get("faceInput"):raise ValueError("顔素材を先に登録してください")
    path=project["faceInput"]["image"] if scope=="face" else project["source"]
    payload=(store.folder(project["id"])/path).read_bytes()
    settings=settings or {"seed":42,"steps":30}
    if type(settings.get("seed")) is not int or not 0<=settings["seed"]<2**32 or type(settings.get("steps")) is not int or not 1<=settings["steps"]<=100:
        raise ValueError("Invalid inference settings")
    job={"schemaVersion":1,"id":uid(),"projectId":project["id"],"revision":project["revision"],
         "scope":scope,"backend":"see-through","input":"input.png","sourceAsset":path,
         "inputSha256":hashlib.sha256(payload).hexdigest(),"settings":settings,"repoCommit":REPO_COMMIT,
         "allowedTags":list(FACE_TAGS if scope=="face" else BODY_TAGS)}
    if scope=="face":job["validation"]=project["faceInput"]["validation"]
    folder=store.folder(project["id"])/"jobs";folder.mkdir(exist_ok=True)
    (folder/f"{job['id']}.json").write_text(json.dumps(job),"utf-8")
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("job.json",json.dumps(job,ensure_ascii=False));z.writestr("input.png",payload)
    return job,buffer.getvalue()


def make_part(tag,label,path,scope,rect,bbox,size,job_id):
    x,y,w,h=rect;left,top,right,bottom=bbox;iw,ih=size
    bounds=[x+left/iw*w,y+top/ih*h,x+right/iw*w,y+bottom/ih*h]
    polygon=[[bounds[0],bounds[1]],[bounds[2],bounds[1]],[bounds[2],bounds[3]],[bounds[0],bounds[3]]]
    role=f"face:{tag}" if scope=="face" else ROLES[tag]
    order=(FACE_ORDER if scope=="face" else BODY_ORDER).index(tag)
    depth=.021+order*.001 if scope=="face" else {"back hair":-.02,"front hair":.045}.get(tag,order*.001)
    return {"id":f"st-{scope}-{tag.replace(' ','-')}","name":label,"role":role,"image":path,"rect":rect,
            "scope":scope,"tag":tag,"depth":depth,"pivot":[(bounds[0]+bounds[2])/2,bounds[1]],
            "polygon":polygon,"mesh":mesh_for(polygon,depth),"locked":False,"edits":{},"visible":True,
            "provenance":"see-through","completion":"inferred","jobId":job_id,
            "physics":{"enabled":role.startswith("hair"),"stiffness":36,"damping":9,"strength":.7}}


def preview_project(project,parts):
    result=copy.deepcopy(project)
    by_id={p["id"]:p for p in result["parts"]}
    for part in parts:
        old=by_id.get(part["id"])
        if old and old.get("locked"):continue
        replacement=copy.deepcopy(part)
        if old:
            replacement["edits"]=old.get("edits",{})
            for key,value in replacement["edits"].items():
                if key in ("name","pivot","depth","physics","visible","rect","role"):
                    replacement[key]=copy.deepcopy(value)
            result["parts"][result["parts"].index(old)]=replacement
        else:result["parts"].append(replacement)
    return result


def import_result(store,project,payload):
    folder=store.folder(project["id"])
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(infos)>40 or len(names)!=len(set(names)) or sum(i.file_size for i in infos)>250_000_000:
            raise ValueError("結果ZIPが大きすぎるか、重複エントリがあります")
        if any(n!="result.json" and not re.fullmatch(r"layers/[a-zA-Z0-9_-]+\.png",n) for n in names):
            raise ValueError("Unsafe result ZIP path")
        metadata=json.loads(z.read("result.json"))
        job_path=folder/"jobs"/f"{safe_id(metadata['jobId'])}.json"
        if not job_path.is_file():raise ValueError("このプロジェクトから出力したジョブではありません")
        job=json.loads(job_path.read_text("utf-8"));scope=job["scope"]
        if metadata.get("schemaVersion")!=1 or metadata.get("status")!="success" or metadata.get("backend")!="see-through" or metadata.get("projectId")!=project["id"] or metadata.get("scope")!=scope or metadata.get("inputSha256")!=job["inputSha256"]:
            raise ValueError("結果と入力ジョブの対応が不正です")
        current_path=project.get("faceInput",{}).get("image") if scope=="face" else project["source"]
        if current_path!=job["sourceAsset"] or hashlib.sha256((folder/current_path).read_bytes()).hexdigest()!=job["inputSha256"]:
            raise ValueError("入力素材が差し替えられています。新しいジョブを出力してください")
        allowed=FACE_TAGS if scope=="face" else BODY_TAGS
        layers=metadata.get("layers",[])
        if not layers or len(layers)>len(allowed):raise ValueError("Invalid layer count")
        seen=set();decoded=[]
        source=image_from_bytes((folder/current_path).read_bytes())
        for item in layers:
            tag=item["tag"]
            if tag not in allowed or tag in seen:raise ValueError("対象外または重複したレイヤーです")
            seen.add(tag);data=z.read(item["image"])
            if hashlib.sha256(data).hexdigest()!=item["sha256"]:raise ValueError("レイヤーのハッシュが一致しません")
            im=image_from_bytes(data)
            if im.size!=source.size:raise ValueError("レイヤーの寸法が元入力と一致しません")
            bbox=im.getchannel("A").getbbox()
            if not bbox:raise ValueError("透明のみのレイヤーはmissingとして扱ってください")
            decoded.append((tag,im,bbox))
        rect=rectangle(project["faceInput"]["alignment"]["rect"]) if scope=="face" else [0,0,1,1]
        parts=[]
        for tag,im,bbox in decoded:
            name=f"assets/st-{uid()}.png";im.save(folder/name)
            parts.append(make_part(tag,allowed[tag],name,scope,rect,bbox,im.size,job["id"]))
    updated=preview_project(project,parts)
    if updated.get('rig'):updated['rig']['stale']=True
    if scope=="face":
        for tag,im,bbox in decoded:
            if any(p.get("scope")=="face" and p.get("tag")==tag and p["locked"] for p in project["parts"]):continue
            alpha=np.asarray(im)[:,:,3].astype(float)
            if tag not in ("irides","mouth"):continue
            segments=[(0,im.width)] if tag=="mouth" else [(0,im.width//2),(im.width//2,im.width)]
            for index,(left,right) in enumerate(segments):
                a=alpha[:,left:right];mass=a.sum()
                if mass<=0:continue
                cx=float((a.sum(axis=0)*np.arange(left,right)).sum()/mass)/im.width
                cy=float((a.sum(axis=1)*np.arange(im.height)).sum()/mass)/im.height
                key="mouth" if tag=="mouth" else ("leftEye" if index==0 else "rightEye")
                updated["face"][key]=[rect[0]+cx*rect[2],rect[1]+cy*rect[3]]
    updated.setdefault("segmentation",{})[scope]={"jobId":job["id"],"execution":metadata.get("execution",{}),
        "missing":[tag for tag in allowed if tag not in seen],"inputSha256":job["inputSha256"]}
    if scope=='face':
        updated.pop('eyeMask',None)
        if {'irides','eyewhite','eyelash'}<=seen:
            from .character import build_eye_masks
            try:updated=build_eye_masks(store,updated)
            except ValueError:pass  # Partial/empty eyes remain explicitly unmasked until corrected.
    candidate={"id":uid(),"projectId":project["id"],"baseRevision":project["revision"],"backend":"see-through",
               "scope":scope,"project":validate(updated),"parts":parts,
               "warnings":[f"未検出: {allowed[t]}" for t in allowed if t not in seen]}
    candidate_folder=folder/"candidates";candidate_folder.mkdir(exist_ok=True)
    (candidate_folder/f"{candidate['id']}.json").write_text(json.dumps(candidate,ensure_ascii=False),"utf-8")
    return candidate


def split_part(store,project,part_id,axis,position):
    """Geometric cut, never presented as anatomical inference. Both outputs exact."""
    if axis not in ("x","y") or type(position) not in (int,float) or not 0<position<1:
        raise ValueError("切断位置は0〜1の範囲内です")
    updated=copy.deepcopy(project);part=next((p for p in updated["parts"] if p["id"]==part_id),None)
    if not part:raise ValueError("Unknown part")
    if part["locked"]:raise ValueError("保護を解除してから分割してください")
    image=Image.open(store.folder(project["id"])/part["image"]).convert("RGBA")
    arrays=[np.array(image),np.array(image)]
    rect=part.get("rect",[0,0,1,1]);idx=0 if axis=="x" else 1
    coordinate=(position-rect[idx])/rect[idx+2]
    cut=round(coordinate*image.size[idx])
    if not 0<cut<image.size[idx]:raise ValueError("切断線がパーツの範囲外です")
    if axis=="x":arrays[0][:,cut:,3]=0;arrays[1][:,:cut,3]=0
    else:arrays[0][cut:,:,3]=0;arrays[1][:cut,:,3]=0
    if any(not a[:,:,3].any() for a in arrays):raise ValueError("片側が空になります。切断位置を調整してください")
    for index,array in enumerate(arrays):
        child=copy.deepcopy(part);child["id"]=uid();child["name"]+=f" / {index+1}"
        child["image"]=f"assets/split-{child['id']}.png";child["derivedFrom"]=part["id"]
        child["locked"]=True;child["visible"]=True;child["provenance"]="manual-split";child["edits"]={}
        child_image=Image.fromarray(array);child_image.save(store.folder(project["id"])/child["image"])
        bbox=child_image.getchannel("A").getbbox();iw,ih=child_image.size
        x,y,w,h=rect;l,t,r,b=bbox
        child["polygon"]=[[x+l/iw*w,y+t/ih*h],[x+r/iw*w,y+t/ih*h],[x+r/iw*w,y+b/ih*h],[x+l/iw*w,y+b/ih*h]]
        child["pivot"]=[x+(l+r)/2/iw*w,y+t/ih*h];child["mesh"]=mesh_for(child["polygon"],child["depth"])
        updated["parts"].append(child)
    part["visible"]=False;part["locked"]=True
    if updated.get('rig'):updated['rig']['stale']=True
    return validate(updated)


def split_islands(store,project,part_id):
    """Split only disconnected column islands; connected strands need manual cuts."""
    part=next(p for p in project["parts"] if p["id"]==part_id)
    if part["locked"]:raise ValueError("保護を解除してから分割してください")
    image=Image.open(store.folder(project["id"])/part["image"]).convert("RGBA")
    active=np.any(np.asarray(image)[:,:,3]>0,axis=0)
    runs=np.flatnonzero(np.diff(np.r_[False,active,False])).reshape(-1,2)
    runs=[(int(a),int(b)) for a,b in runs if b-a>=2]
    if len(runs)<2:raise ValueError("独立した左右の領域がありません。手動の切断線で細分化してください。")
    # Make one conservative cut per action. Further cuts can be repeated explicitly.
    cut=(runs[0][1]+runs[1][0])/2/image.width
    rect=part.get("rect",[0,0,1,1])
    return split_part(store,project,part_id,"x",rect[0]+cut*rect[2])
