"""Standalone Colab/local See-through runner and strict result packager."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from PIL import Image

REPO_COMMIT = "58a1cb11d13f85acec9bbddb8cd4b6487843d4cf"
FACE_TAGS = {"eyebrow":"眉毛", "eyelash":"まつげ", "mouth":"口", "nose":"鼻",
             "face":"肌", "ears":"耳", "irides":"瞳", "eyewhite":"白目"}
BODY_TAGS = {"back hair":"後ろ髪", "front hair":"前髪", "neck":"首", "topwear":"胸・上衣（袖を含む場合あり）",
             "handwear":"腕・手（モデル推定）", "bottomwear":"スカート・下衣", "legwear":"足・脚衣", "footwear":"靴"}


def prepare_face(image):
    if image.width != image.height or image.width < 2160:
        raise ValueError("Face input must be square and at least 2160px. Upscale with FFmpeg before export.")
    size = math.ceil(image.width/64)*64
    offset = (size-image.width)//2
    canvas = Image.new("RGBA", (size,size), (255,255,255,255))
    canvas.alpha_composite(image.convert("RGBA"), (offset,offset))
    return canvas.convert("RGB"), offset


def prepare_input(image, scope):
    if scope == "face":
        padded,offset=prepare_face(image)
        return padded,[offset,offset]
    size=math.ceil(max(image.size)/64)*64
    offset=[(size-image.width)//2,(size-image.height)//2]
    canvas=Image.new("RGBA",(size,size),(255,255,255,255))
    canvas.alpha_composite(image.convert("RGBA"),tuple(offset))
    return canvas.convert("RGB"),offset


def read_job(path):
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist())>30 or sum(x.file_size for x in archive.infolist())>100_000_000:
            raise ValueError("Oversized job")
        job=json.loads(archive.read("job.json"))
        if job.get("schemaVersion")!=1 or job.get("backend")!="see-through" or job.get("scope") not in ("face","body"):
            raise ValueError("Unsupported job")
        if not re.fullmatch(r"[a-f0-9]{32}",job.get("id","")):raise ValueError("Invalid job ID")
        if job.get("repoCommit")!=REPO_COMMIT:raise ValueError("Unsupported worker version")
        settings=job.get("settings",{})
        if type(settings.get("seed")) is not int or not 0<=settings["seed"]<2**32 or type(settings.get("steps")) is not int or not 1<=settings["steps"]<=100:
            raise ValueError("Invalid inference settings")
        source=archive.read("input.png")
        if hashlib.sha256(source).hexdigest()!=job["inputSha256"]:
            raise ValueError("Input hash mismatch")
        image=Image.open(io.BytesIO(source))
        if image.width*image.height>20_000_000:raise ValueError("Image too large")
        image=image.convert("RGBA")
    return job,image


def package_result(job, image, padded_size, offset, layer_dir, result_path, execution):
    """Whitelist output and map padded model coordinates back to source pixels."""
    tags=FACE_TAGS if job["scope"]=="face" else BODY_TAGS
    tags={k:v for k,v in tags.items() if k in job.get('allowedTags',tags)}
    result={"schemaVersion":1,"jobId":job["id"],"projectId":job["projectId"],
            "inputSha256":job["inputSha256"],"scope":job["scope"],"status":"success",
            "backend":"see-through","sourceSize":list(image.size),"inferenceResolution":padded_size,
            "paddingOffset":offset,"layers":[],"missing":[],"execution":execution}
    destination=Path(result_path);temp=destination.with_suffix(".tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as archive:
        for tag,label in tags.items():
            path=Path(layer_dir)/f"{tag}.png"
            if not path.is_file():
                result["missing"].append(tag);continue
            layer=Image.open(path).convert("RGBA")
            if layer.size!=(padded_size,padded_size):
                raise ValueError(f"Unexpected {tag} dimensions {layer.size}; no silent resizing")
            layer=layer.crop((offset[0],offset[1],offset[0]+image.width,offset[1]+image.height))
            if not layer.getchannel("A").getbbox():
                result["missing"].append(tag);continue
            filename=f"layers/{tag.replace(' ','_')}.png"
            output=io.BytesIO();layer.save(output,"PNG");payload=output.getvalue()
            archive.writestr(filename,payload)
            result["layers"].append({"tag":tag,"name":label,"image":filename,"sha256":hashlib.sha256(payload).hexdigest()})
        if not result["layers"]:raise ValueError("No requested nonempty layers were generated")
        archive.writestr("result.json",json.dumps(result,ensure_ascii=False))
    os.replace(temp,destination)
    return result


def run(job_path,result_path,repo,group_offload=False,work_dir=None):
    job,image=read_job(job_path);repo=Path(repo).resolve()
    script=repo/"inference/scripts/inference_psd.py"
    if not script.is_file():raise ValueError("Choose an official See-through checkout")
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if revision!=REPO_COMMIT:raise ValueError(f"See-through checkout must be pinned to {REPO_COMMIT}")
    work=Path(work_dir or Path(result_path).with_suffix(".work")).resolve()/job["id"]
    work.mkdir(parents=True,exist_ok=True)
    padded,offset=prepare_input(image,job["scope"])
    input_path=work/"input.png";padded.save(input_path)
    output=work/"output";marker=work/"inference-complete.json"
    parameters={"inputSha256":job["inputSha256"],"resolution":padded.width,
                "seed":job["settings"]["seed"],"steps":job["settings"]["steps"],"commit":revision}
    command=[sys.executable,str(script),"--srcp",str(input_path),"--save_dir",str(output),
             "--resolution",str(padded.width),"--resolution_depth","768",
             "--inference_steps",str(parameters["steps"]),"--seed",str(parameters["seed"]),"--save_to_psd"]
    if group_offload:command.append("--group_offload")
    start=time.monotonic()
    if not marker.exists() or json.loads(marker.read_text("utf-8"))!=parameters:
        print("Running:"," ".join(command),flush=True)
        env=os.environ.copy();env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
        with (work/"inference.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
            for line in process.stdout:
                print(line,end="",flush=True);log.write(line);log.flush()
            code=process.wait()
        if code:
            (work/"failure.json").write_text(json.dumps({"status":"failed","jobId":job["id"],"exitCode":code}),"utf-8")
            raise RuntimeError(f"See-through failed ({code}). Log: {work/'inference.log'}. Resolution was not reduced.")
        marker.write_text(json.dumps(parameters),"utf-8")
    import torch
    execution={**parameters,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
               "secondsThisRun":time.monotonic()-start,"groupOffload":group_offload,
               "model":"layerdifforg/seethroughv0.0.2_layerdiff3d","inferred":True}
    result=package_result(job,image,padded.width,offset,output/"input",result_path,execution)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job");parser.add_argument("result");parser.add_argument("--repo",required=True)
    parser.add_argument("--group-offload",action="store_true");parser.add_argument("--work-dir")
    args=parser.parse_args()
    run(args.job,args.result,args.repo,args.group_offload,args.work_dir)
