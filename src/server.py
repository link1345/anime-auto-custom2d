from __future__ import annotations

import asyncio
import copy
import io
import json
import math
import time
import zipfile
from pathlib import Path

from aiohttp import web, WSMsgType
from PIL import Image

from .pipeline import create_project, generate_parts, image_from_bytes, validate_face_image
from .project import Store, uid, safe_id, merge_generated
from .runtime import Runtime
from .media import upscale_face
from .segmentation import make_job, import_result as import_segmentation_result, rectangle, split_part, split_islands
from .colab import notebook
from . import compute, character
from .glb import export_glb

ROOT = Path(__file__).resolve().parent.parent


@web.middleware
async def local_only(request, handler):
    # Binding loopback is not sufficient: also reject browser cross-origin requests
    # and DNS rebinding. OBS and Python clients omit Origin; those remain local.
    if request.host not in (f"127.0.0.1:{request.app['port']}", f"localhost:{request.app['port']}"):
        raise web.HTTPForbidden(text="Invalid host")
    origin = request.headers.get("Origin")
    if origin and origin not in (f"http://127.0.0.1:{request.app['port']}", f"http://localhost:{request.app['port']}"):
        raise web.HTTPForbidden(text="Cross-origin access is disabled")
    try:
        return await handler(request)
    except (ValueError, KeyError, TypeError, StopIteration, zipfile.BadZipFile) as exc:
        return web.json_response({"error": str(exc)}, status=400)


def finite(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Expected number in [{low}, {high}]")
    return value


def active(request):
    project = request.app["runtime"].project
    if not project:
        raise ValueError("No active project")
    return project


async def announce(app):
    for socket in tuple(app["sockets"]):
        if not socket.closed:
            await socket.send_json({"type": "projectChanged"})


async def commit(request, project, checkpoint=False):
    app = request.app
    previous = app["runtime"].project
    app["undo"].append(copy.deepcopy(previous))
    del app["undo"][:-40]
    app["redo"].clear()
    project["revision"] += 1
    app["store"].save(project, checkpoint)
    app["runtime"].project = project
    await announce(app)
    return web.json_response(project)


async def projects(request):
    return web.json_response(request.app["store"].list())


async def current(request):
    return web.json_response(active(request))


async def new_project(request):
    data = await request.post()
    upload = data.get("image")
    if not upload or not hasattr(upload, "file"):
        raise ValueError("Choose a PNG/JPEG/WebP image")
    project = await asyncio.to_thread(create_project, request.app["store"], upload.file.read(), data.get("name", "新しいキャラクター"))
    request.app["runtime"].project = project
    request.app["undo"].clear()
    request.app["redo"].clear()
    await announce(request.app)
    return web.json_response(project)


async def select_project(request):
    request.app["runtime"].__init__()
    request.app["runtime"].project = request.app["store"].load(request.match_info["id"])
    request.app["undo"].clear()
    request.app["redo"].clear()
    await announce(request.app)
    return web.json_response(active(request))


async def edit_part(request):
    project = copy.deepcopy(active(request))
    part = next(p for p in project["parts"] if p["id"] == request.match_info["id"])
    data = await request.json()
    if part.get("locked") and set(data) != {"locked"}:
        raise ValueError("Unlock the part before editing")
    for key, value in data.items():
        if key == "name":
            part[key] = str(value)[:80]
        elif key == "locked":
            if type(value) is not bool:
                raise ValueError("Expected boolean")
            part[key] = value
        elif key == "visible":
            if type(value) is not bool:raise ValueError("Expected boolean")
            part[key]=value
        elif key == "role":
            if value not in ("head","neck","chest","armLeft","armRight","forearmLeft","forearmRight","handLeft","handRight","skirt","legs","feet","hairFront","hairBack","hairLeft","hairRight"):
                raise ValueError("Invalid role")
            part[key]=value
        elif key == "rect":
            part[key]=rectangle(value)
        elif key == "pivot":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Expected 2D pivot")
            part[key] = [finite(x, 0, 1) for x in value]
        elif key == "depth":
            part[key] = finite(value, -.5, .5)
        elif key == "polygon":
            if not isinstance(value, list) or not 3 <= len(value) <= 100:
                raise ValueError("Polygon requires 3–100 points")
            if any(not isinstance(v, list) or len(v) != 2 for v in value):
                raise ValueError("Expected 2D polygon")
            part[key] = [[finite(x, 0, 1), finite(y, 0, 1)] for x,y in value]
            project["stale"] = ["segmentation:mesh,rig"]
        elif key == "physics":
            part[key] = {"enabled": bool(value["enabled"]),
                         "stiffness": finite(value["stiffness"], 1, 100),
                         "damping": finite(value["damping"], 1, 30),
                         "strength": finite(value["strength"], 0, 3)}
        else:
            raise ValueError(f"Unsupported edit {key}")
    part["edits"].update({k: v for k,v in data.items() if k != "locked"})
    if project.get('rig') and set(data)&{'role','polygon','rect'}:project['rig']['stale']=True
    if project.get('eyeMask') and part.get('scope')=='face' and set(data)&{'rect','polygon'}:project.pop('eyeMask')
    return await commit(request, project)


async def regenerate(request):
    project = copy.deepcopy(active(request))
    folder = request.app["store"].folder(project["id"])
    templates=[p for p in project["parts"] if p["provenance"]=="template"]
    if not templates:raise ValueError("テンプレートパーツがありません。See-throughはジョブを再実行してください")
    parts = await asyncio.to_thread(generate_parts, Image.open(folder / project["source"]).convert("RGBA"), folder, templates)
    candidate = {"id": uid(), "projectId":project["id"], "parts": parts, "baseRevision": project["revision"], "backend": "editable-template"}
    request.app["candidates"][candidate["id"]] = candidate
    return web.json_response(candidate)


async def adopt(request):
    candidate_id = (await request.json())["id"]
    project = active(request)
    candidate=request.app["candidates"].get(candidate_id)
    if candidate is None:
        path=request.app["store"].folder(project["id"])/"candidates"/f"{safe_id(candidate_id)}.json"
        if not path.is_file():raise ValueError("候補がありません")
        candidate=json.loads(path.read_text("utf-8"))
    if candidate.get("projectId")!=project["id"] or candidate["baseRevision"] != project["revision"]:
        raise ValueError("Project changed; regenerate the candidate")
    updated = candidate["project"] if candidate["backend"]=="see-through" else merge_generated(project, candidate)
    # Only the template rebuild resolves all template mesh/rig invalidations.
    if candidate["backend"] != "see-through":
        updated["stale"] = []
        updated['meshVersion']+=1
        if updated.get('rig'):updated['rig']['stale']=True
    return await commit(request, updated, True)


async def history(request):
    direction = request.match_info["direction"]
    if direction == "checkpoint":
        request.app["store"].save(active(request), True)
        return web.json_response({"saved": True})
    if direction not in ("undo", "redo"):
        raise ValueError("Invalid history action")
    stack = request.app[direction]
    if not stack:
        raise ValueError("No history")
    opposite = "redo" if direction == "undo" else "undo"
    request.app[opposite].append(copy.deepcopy(active(request)))
    project = stack.pop()
    # Monotonic revisions also invalidate candidates after an undo.
    project["revision"] = active(request)["revision"] + 1
    request.app["runtime"].project = project
    request.app["store"].save(project)
    await announce(request.app)
    return web.json_response(project)


async def replace_image(request):
    project = copy.deepcopy(active(request))
    part = next(p for p in project["parts"] if p["id"] == request.match_info["id"])
    if part["locked"]:
        raise ValueError("Part is locked")
    data = await request.post()
    image = image_from_bytes(data["image"].file.read())
    if list(image.size) != project["size"]:
        raise ValueError("Replacement must use the same canvas dimensions to preserve alignment")
    filename = f"assets/{uid()}.png"
    image.save(request.app["store"].folder(project["id"]) / filename)
    part["image"], part["provenance"] = filename, "user"
    part["locked"] = True
    return await commit(request, project)


async def expressions(request):
    project = copy.deepcopy(active(request))
    data = await request.post()
    role = data.get("target", "face")
    if role not in ("face", "eyes", "mouth", "brows"):
        raise ValueError("Invalid expression target")
    image = image_from_bytes(data["image"].file.read())
    eid = uid()
    path = f"assets/expression-{eid}.png"
    image.save(request.app["store"].folder(project["id"]) / path)
    rect = json.loads(data.get("rect", "[0,0,1,1]"))
    if len(rect) != 4:
        raise ValueError("Expected x,y,width,height")
    rect = [finite(x, 0, 1) for x in rect]
    if rect[2] <= 0 or rect[3] <= 0 or rect[0]+rect[2] > 1 or rect[1]+rect[3] > 1:
        raise ValueError("Rectangle must fit the image")
    project["expressions"].append({"id": eid, "name": str(data.get("name", "表情"))[:80],
        "target": role, "image": path, "source": "user", "status": "adopted",
        "meshVersion": project["meshVersion"], "rect": rect,
        "method": "crossfade", "alignment": "manual", "stale": False})
    return await commit(request, project)


async def face_input(request):
    project = copy.deepcopy(active(request))
    data = await request.post()
    image = image_from_bytes(data["image"].file.read())
    folder=request.app["store"].folder(project["id"])
    original=f"assets/face-original-{uid()}.png";image.save(folder/original)
    scaling=None
    if data.get("upscale")=="true":
        scaled=f"assets/face-scaled-{uid()}.png"
        scaling=await asyncio.to_thread(upscale_face,folder/original,folder/scaled,int(data.get("size",2160)))
        image=Image.open(folder/scaled).convert("RGBA")
    metadata = validate_face_image(image, data.get("faceFillsFrame") == "true")
    if scaling:metadata.update(scaling)
    if active(request)["id"] != project["id"] or active(request)["revision"] != project["revision"]:
        raise ValueError("顔画像の処理中にプロジェクトが変更されました。もう一度登録してください")
    filename = f"assets/face-{uid()}.png"
    image.save(request.app["store"].folder(project["id"]) / filename)
    project["faceInput"] = {"image": filename, "original":original, "validation": metadata, "source": "user",
                            "alignment": {"rect": [.245, .01, .52, .3467], "confirmed": False}}
    return await commit(request, project)


async def face_job(request):
    scope=request.query.get("scope","face")
    _,payload=make_job(request.app["store"],active(request),scope)
    return web.Response(body=payload,content_type="application/zip",headers={"Content-Disposition":f'attachment; filename="see-through-{scope}-job.zip"'})


async def face_alignment(request):
    data=await request.json();p=copy.deepcopy(active(request))
    if not p.get("faceInput"):raise ValueError("顔素材を登録してください")
    rect=rectangle(data["rect"])
    previous=p["faceInput"]["alignment"]["rect"]
    p["faceInput"]["alignment"]={"rect":rect,"confirmed":True}
    if p.get("segmentation",{}).get("face"):
        for key in ("leftEye","rightEye","mouth"):
            tag="mouth" if key=="mouth" else "irides"
            if any(part.get("scope")=="face" and part.get("tag")==tag and part["locked"] for part in p["parts"]):
                continue
            x,y=p["face"][key]
            p["face"][key]=[rect[0]+(x-previous[0])/previous[2]*rect[2],rect[1]+(y-previous[1])/previous[3]*rect[3]]
    # Existing extracted layers follow the new alignment except locked/manual ones.
    for part in p["parts"]:
        if part.get("scope")=="face" and not part["locked"]:
            old=part.get("rect",[0,0,1,1]);pivot=part["pivot"]
            part["pivot"]=[rect[0]+(pivot[0]-old[0])/old[2]*rect[2],rect[1]+(pivot[1]-old[1])/old[3]*rect[3]]
            part["polygon"]=[[rect[0]+(x-old[0])/old[2]*rect[2],rect[1]+(y-old[1])/old[3]*rect[3]] for x,y in part["polygon"]]
            part["rect"]=rect
            from .pipeline import mesh_for
            part["mesh"]=mesh_for(part["polygon"],part["depth"])
    return await commit(request,p)


async def segmentation_import(request):
    data=await request.post()
    candidate=await asyncio.to_thread(import_segmentation_result,request.app["store"],copy.deepcopy(active(request)),data["file"].file.read())
    request.app["candidates"][candidate["id"]]=candidate
    return web.json_response(candidate)


async def segmentation_candidates(request):
    folder=request.app["store"].folder(active(request)["id"])/"candidates"
    return web.json_response([json.loads(p.read_text("utf-8")) for p in folder.glob("*.json") if json.loads(p.read_text("utf-8"))["baseRevision"]==active(request)["revision"]])


async def colab_download(request):
    return web.Response(text=json.dumps(notebook(),ensure_ascii=False),content_type="application/json",
                        headers={"Content-Disposition":'attachment; filename="Nagi_SeeThrough.ipynb"'})


async def split_layer(request):
    data=await request.json();p=active(request)
    if data.get("auto"):
        updated=split_islands(request.app["store"],p,request.match_info["id"])
    else:
        updated=split_part(request.app["store"],p,request.match_info["id"],data["axis"],data["position"])
    return await commit(request,updated,True)


async def face_effect(request):
    data=await request.json();p=copy.deepcopy(active(request))
    effects=p.setdefault("faceEffects",[])
    if data.get("delete"):
        p["faceEffects"]=[e for e in effects if e["id"]!=data["delete"]]
    elif data.get("id"):
        effect=next((e for e in effects if e["id"]==data["id"]),None)
        if not effect:raise ValueError("Unknown effect")
        for key in ("opacity","scale"):
            if key in data:effect[key]=finite(data[key],0,1 if key=="opacity" else 3)
        if "visible" in data:
            if type(data["visible"]) is not bool:raise ValueError("Expected boolean")
            effect["visible"]=data["visible"]
    else:
        kind=data.get("kind")
        if kind not in ("blush","star","heart"):raise ValueError("Unknown face effect")
        if len(effects)>=24:raise ValueError("顔装飾は24個までです")
        effects.append({"id":uid(),"kind":kind,"name":{"blush":"頬の赤み","star":"星の瞳","heart":"ハートの瞳"}[kind],
                        "opacity":.5 if kind=="blush" else .9,"scale":1,"visible":True,"source":"template"})
    return await commit(request,p)


async def effect_image(request):
    data=await request.post();p=copy.deepcopy(active(request))
    image=image_from_bytes(data["image"].file.read());path=f"assets/effect-{uid()}.png"
    image.save(request.app["store"].folder(p["id"])/path)
    rect=rectangle(json.loads(data.get("rect","[0.38,0.21,0.24,0.05]")))
    effects=p.setdefault("faceEffects",[])
    if len(effects)>=24:raise ValueError("顔装飾は24個までです")
    effects.append({"id":uid(),"kind":"image","image":path,"rect":rect,"opacity":1,"scale":1,"visible":True,
                    "name":str(data.get("name","追加装飾"))[:80],"source":"user"})
    return await commit(request,p)


async def display_settings(request):
    data=await request.json();p=copy.deepcopy(active(request))
    if type(data.get("templatesVisible")) is not bool:raise ValueError("Expected boolean")
    p.setdefault("display",{})["templatesVisible"]=data["templatesVisible"]
    return await commit(request,p)


async def export(request):
    payload = request.app["store"].export(active(request)["id"], request.match_info["kind"] == "model")
    ext = "aac2d" if request.match_info["kind"] == "model" else "zip"
    return web.Response(body=payload, content_type="application/zip", headers={"Content-Disposition": f'attachment; filename="avatar.{ext}"'})


async def import_archive(request):
    data = await request.post()
    project = request.app["store"].import_archive(data["file"].file.read())
    request.app["runtime"].__init__()
    request.app["runtime"].project = project
    request.app["undo"].clear()
    request.app["redo"].clear()
    await announce(request.app)
    return web.json_response(project)


async def job_export(request):
    backend=request.query.get('backend','moge3')
    _,payload=compute.make_job(request.app['store'],active(request),backend,request.query.get('scope'))
    return web.Response(body=payload,content_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{backend}-job.zip"'})


async def result_import(request):
    data=await request.post();p=copy.deepcopy(active(request))
    updated=await asyncio.to_thread(compute.integrate,request.app['store'],p,data['file'].file.read())
    if active(request)['id']!=p['id'] or active(request)['revision']!=p['revision']:raise ValueError('取り込み中に編集されました。再試行してください')
    return await commit(request,updated,True)


async def compute_book(request):
    book=compute.compute_notebook(request.match_info['backend'])
    return web.Response(text=json.dumps(book,ensure_ascii=False),content_type='application/json',headers={'Content-Disposition':f'attachment; filename="Nagi_{request.match_info["backend"]}.ipynb"'})


async def compute_settings(request):
    if request.method=='GET':return web.json_response(compute.configuration())
    data=await request.json();settings=compute.configuration()
    for key in settings:
        if key in data:settings[key]=data[key]
    if not Path(settings['python']).is_file() or not Path(settings['python']).name.lower().startswith('python'):raise ValueError('推論環境のPython実行ファイルを指定してください')
    if settings['device'] not in ('cpu','cuda'):raise ValueError('Invalid device')
    if type(settings['resolutionLevel']) is not int or not 0<=settings['resolutionLevel']<=9:raise ValueError('Invalid resolution level')
    if type(settings['refineSteps']) is not int or not 0<=settings['refineSteps']<=10:raise ValueError('Invalid refine steps')
    if type(settings['checkpoint']) is not str:raise ValueError('Invalid checkpoint')
    path=ROOT/'data/compute-settings.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(settings),'utf-8')
    return web.json_response(settings)


async def compute_local(request):
    state=request.app['compute_state']
    if request.method=='GET':
        value={k:v for k,v in state.items() if k!='process'}
        if state.get('log') and Path(state['log']).is_file():
            with Path(state['log']).open('rb') as f:
                f.seek(max(0,Path(state['log']).stat().st_size-6000));value['tail']=f.read().decode('utf-8','replace')
        return web.json_response(value)
    data=await request.json()
    if data.get('cancel'):
        task=request.app.get('compute_task')
        if task and not task.done():task.cancel()
        return web.json_response({'status':'cancelling'})
    if data.get('apply'):
        if state.get('status')!='ready' or state.get('projectId')!=active(request)['id']:raise ValueError('現在のプロジェクトの完了結果がありません')
        p=copy.deepcopy(active(request));updated=await asyncio.to_thread(compute.integrate,request.app['store'],p,Path(state['result']).read_bytes())
        if p['revision']!=active(request)['revision'] or p['id']!=active(request)['id']:raise ValueError('取り込み中に編集されました')
        result=await commit(request,updated,True);state['status']='applied'
        (request.app['store'].folder(p['id'])/'jobs'/(state['id']+'-status.json')).write_text(json.dumps(state),'utf-8')
        return result
    if request.app.get('compute_task') and not request.app['compute_task'].done():raise ValueError('既に処理中です')
    backend=data.get('backend')
    if backend not in ('moge3','mediapipe'):raise ValueError('Invalid backend')
    settings=compute.configuration()
    if data.get('faceScope'):
        if backend!='moge3' or data['faceScope'] not in ('front','left','right'):raise ValueError('Invalid face scope')
        settings['faceScope']=data['faceScope']
    state.clear();state.update(status='queued',backend=backend)
    request.app['compute_task']=asyncio.create_task(compute.execute_local(request.app['store'],copy.deepcopy(active(request)),backend,state,settings))
    return web.json_response({'status':'queued'})


async def direction_image(request):
    p=copy.deepcopy(active(request));data=await request.post();side=data.get('side')
    if side not in ('left','right'):raise ValueError('Invalid direction')
    if not p.get('faceInput'):raise ValueError('正面顔画像が必要です')
    image=image_from_bytes(data['image'].file.read());folder=request.app['store'].folder(p['id'])
    original=f'assets/direction-original-{uid()}.png';image.save(folder/original)
    path=f'assets/direction-{uid()}.png'
    metadata=await asyncio.to_thread(upscale_face,folder/original,folder/path,2160)
    if active(request)['id']!=p['id'] or active(request)['revision']!=p['revision']:raise ValueError('処理中に編集されました。再試行してください')
    group=p.setdefault('directionalFace',{'enabled':False,'views':{}})
    if p.get('multiViewFace'):p['multiViewFace']['enabled']=False
    group['views'][side]={'image':path,'original':original,'validation':metadata,'fitted':False}
    return await commit(request,p,True)


async def character_edit(request):
    data=await request.json();p=copy.deepcopy(active(request));store=request.app['store'];operation=data.get('operation')
    if operation=='parent-motion':
        from .parent_motion import configure
        updated=configure(store,p,data)
    elif operation=='multiview-fit':
        from .multiview import fit_multiview
        updated=await asyncio.to_thread(fit_multiview,store,p,finite(data.get('strength',.35),0,1),finite(data.get('correction',1),0,1))
        if active(request)['id']!=p['id'] or active(request)['revision']!=p['revision']:raise ValueError('調整中に編集されました。再試行してください')
    elif operation=='multiview-enable':
        updated=p
        if not p.get('multiViewFace'):raise ValueError('先に複数視点の共通顔面を構築してください')
        updated['multiViewFace']['enabled']=bool(data.get('enabled',True))
    elif operation=='direction-fit':updated=character.fit_direction(p,data['side'],data['basePoints'],data['targetPoints'],data.get('angle',18))
    elif operation=='direction-enable':
        updated=p;updated.setdefault('directionalFace',{'views':{}})['enabled']=bool(data.get('enabled',True))
    elif operation=='joint':updated=character.manual_joint(p,data['name'],data['position'])
    elif operation=='rig':updated=character.build_rig(p)
    elif operation=='split-arm':updated=character.split_arm(store,p,data['partId'],data['side'])
    elif operation=='fit-expression':updated=character.fit_expression(store,p,data['expressionId'],data['partId'],data.get('basePoints'),data.get('targetPoints'))
    elif operation=='head-surface':updated=character.build_head_surface(store,p)
    elif operation=='head-surface-settings':
        if not p.get('headSurface'):raise ValueError('先に共通立体顔面を構築してください')
        updated=p;updated['headSurface']['enabled']=bool(data.get('enabled',True));updated['headSurface']['strength']=finite(data.get('strength',1),0,1.5)
    elif operation=='eye-masks':updated=character.build_eye_masks(store,p,data.get('scale'),data.get('synchronize',False))
    elif operation=='hidden':updated=character.extend_hidden(store,p,data['partId'],data.get('pixels',8))
    elif operation=='person-mask':
        if not p.get('personMask'):raise ValueError('人物マスクがありません')
        updated=p;updated['personMask']['enabled']=bool(data.get('enabled',True))
    else:raise ValueError('Unknown character operation')
    return await commit(request,updated,True)


async def glb_export(request):
    payload=export_glb(request.app['store'],active(request))
    return web.Response(body=payload,content_type='model/gltf-binary',headers={'Content-Disposition':'attachment; filename="character.glb"'})


async def asset(request):
    project_id = safe_id(request.match_info["project"])
    name = request.match_info["name"]
    if Path(name).name != name or not name.endswith(".png"):
        raise web.HTTPNotFound()
    return web.FileResponse(request.app["store"].folder(project_id) / "assets" / name)


async def index(request):
    return web.FileResponse(ROOT / "web/index.html")


async def output(request):
    return web.FileResponse(ROOT / "web/output.html")


async def socket(request):
    ws = web.WebSocketResponse(max_msg_size=16_384, heartbeat=20)
    await ws.prepare(request)
    request.app["sockets"].add(ws)
    try:
        await ws.send_json(request.app["runtime"].snapshot())
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                command_id = None
                try:
                    message = json.loads(msg.data)
                    command_id = message.get("id") if isinstance(message, dict) else None
                    result = request.app["runtime"].command(message)
                    await ws.send_json({"type": "response", "id": command_id, "ok": True, "result": result})
                except (ValueError, TypeError, KeyError) as exc:
                    await ws.send_json({"type": "response", "id": command_id, "ok": False, "error": str(exc)})
    finally:
        request.app["sockets"].discard(ws)
    return ws


async def simulation(app):
    deadline = time.monotonic()
    tick = 0
    while True:
        app["runtime"].step()
        tick += 1
        if tick % 2 == 0:
            packet = json.dumps(app["runtime"].snapshot())
            for ws in tuple(app["sockets"]):
                if ws.closed:
                    continue
                try:
                    await asyncio.wait_for(ws.send_str(packet), timeout=.05)
                except (TimeoutError, ConnectionError, RuntimeError):
                    app["sockets"].discard(ws)
        deadline += 1/60
        if time.monotonic() - deadline > .25:
            deadline = time.monotonic()
        await asyncio.sleep(max(0, deadline-time.monotonic()))


async def lifecycle(app):
    task = asyncio.create_task(simulation(app))
    yield
    task.cancel()
    worker=app.get("compute_task")
    if worker and not worker.done():
        worker.cancel()
        try:await worker
        except asyncio.CancelledError:pass
    try:
        await task
    except asyncio.CancelledError:
        pass
    for ws in tuple(app["sockets"]):
        await ws.close()


def create_app(data=None, port=8765):
    app = web.Application(client_max_size=150_000_000, middlewares=[local_only])
    app.update(store=Store(data or ROOT / "data"), runtime=Runtime(), sockets=set(),
               undo=[], redo=[], candidates={}, jobs={}, compute_state={}, port=port)
    available = app["store"].list()
    if available:
        app["runtime"].project = app["store"].load(available[0]["id"])
    else:
        app["runtime"].project = create_project(app["store"], (ROOT / "assets/nagi.png").read_bytes(), "Nagi / ナギ")
    app['compute_state'].update(compute.restore_state(app['store'],app['runtime'].project))
    app.add_routes([
        web.get("/api/projects", projects), web.get("/api/project", current),
        web.post("/api/projects", new_project), web.post("/api/projects/{id}/select", select_project),
        web.patch("/api/parts/{id}", edit_part), web.post("/api/parts/{id}/image", replace_image),
        web.post("/api/regenerate", regenerate), web.post("/api/adopt", adopt),
        web.post("/api/history/{direction}", history), web.post("/api/expressions", expressions),
        web.post("/api/face", face_input), web.get("/api/face-job", face_job),
        web.patch("/api/face/alignment",face_alignment),web.get("/api/colab",colab_download),
        web.post("/api/segmentation/result",segmentation_import),web.get("/api/segmentation/candidates",segmentation_candidates),
        web.post("/api/parts/{id}/split",split_layer),web.post("/api/face/effects",face_effect),
        web.post("/api/face/effect-image",effect_image),
        web.patch("/api/display",display_settings),
        web.get("/api/export/{kind}", export), web.post("/api/import", import_archive),
        web.get("/api/job", job_export), web.post("/api/result", result_import),
        web.get("/api/compute/notebook/{backend}",compute_book),
        web.get("/api/compute/settings",compute_settings),web.patch("/api/compute/settings",compute_settings),
        web.get("/api/compute/local",compute_local),web.post("/api/compute/local",compute_local),
        web.post("/api/direction-image",direction_image),web.post("/api/character",character_edit),web.get("/api/character.glb",glb_export),
        web.get("/assets/{project}/{name}", asset), web.get("/ws", socket),
        web.get("/", index), web.get("/output", output),
        web.static("/web/", ROOT / "web")])
    app.cleanup_ctx.append(lifecycle)
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    web.run_app(create_app(args.data, args.port), host="127.0.0.1", port=args.port)
