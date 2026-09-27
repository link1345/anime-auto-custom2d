"""Self-contained notebook; executes the pinned worker in fresh subprocesses."""
import json
from pathlib import Path
from scripts.run_face_job import REPO_COMMIT


def notebook():
    root=Path(__file__).resolve().parent.parent
    worker=(root/"scripts/run_face_job.py").read_text("utf-8")
    def cell(kind,text):
        data={"cell_type":kind,"metadata":{},"source":text.splitlines(keepends=True)}
        if kind=="code":data.update(execution_count=None,outputs=[])
        return data
    cells=[cell("markdown","""# Nagi Studio — See-through 顔・髪・体の分割

1. ランタイムをGPUへ変更します。2160px顔はL4 24GB以上、A100推奨です（達成保証ではありません）。
2. 上から実行し、アプリの「顔ジョブ ZIP」または「髪・体ジョブ ZIP」をアップロードします。
3. 完了した `see-through-result.zip` をアプリの「See-through結果 ZIP」から読み込みます。
4. 顔の位置、不要レイヤー、保護した編集を確認し「採用」を押します。

顔結果には眉毛・まつげ・口・鼻・肌・耳・瞳・白目の8種類のみを書き出します。
実装内部の推論は公式の全レイヤー処理ですが、その他の顔レイヤーは結果ZIPから除外します。
顔の2160px入力は2176pxへ余白追加し、結果では除去します。縮小フォールバックはありません。
初回は公式コードとモデルを取得します。Drive保存を選ぶと中間結果・ログ・ZIPを残せます。
"""),cell("code",f'''# 設定
USE_DRIVE = True
GROUP_OFFLOAD = True
REPO_COMMIT = "{REPO_COMMIT}"
DRIVE_FOLDER = "NagiStudio-SeeThrough"
'''),cell("code",'''import os, sys, subprocess, json, hashlib, zipfile, io
from pathlib import Path
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
print(gpu)
if int(gpu.splitlines()[0].rsplit(",", 1)[1]) < 22000:
    raise RuntimeError("2160px処理にはL4/A100等の22GB以上のGPUを選択してください。解像度は下げません。")
from google.colab import drive, files
if USE_DRIVE:
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive") / DRIVE_FOLDER
else:
    OUTPUT_ROOT = Path("/content/nagi-output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
'''),cell("code",'''# 準備。ノートブック本体でtorch/PILをimportせず、推論は新しいPythonプロセスで行います。
REPO = Path("/content/nagi-see-through")
if not REPO.exists():
    subprocess.run(["git", "clone", "https://github.com/shitagaki-lab/see-through.git", str(REPO)], check=True)
subprocess.run(["git", "fetch", "origin", REPO_COMMIT], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", REPO_COMMIT], cwd=REPO, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "torchaudio==2.8.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO, check=True)
assets = REPO / "assets"
if not assets.exists():
    assets.symlink_to(REPO / "common/assets", target_is_directory=True)
subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name())"], check=True)
'''),cell("code",'''# ジョブを選択。ZIP内のPythonコードを実行することはありません。
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("ジョブZIPを1つだけ選択してください")
payload = next(iter(uploaded.values()))
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    if sum(i.file_size for i in archive.infolist()) > 100_000_000:
        raise ValueError("ジョブが大きすぎます")
    job = json.loads(archive.read("job.json"))
    if job.get("backend") != "see-through" or job.get("scope") not in ("face", "body"):
        raise ValueError("Nagi StudioのSee-throughジョブではありません")
    if not __import__("re").fullmatch(r"[a-f0-9]{32}", job["id"]):
        raise ValueError("Invalid job ID")
JOB = OUTPUT_ROOT / (job["id"] + ".zip")
JOB.write_bytes(payload)
RESULT = OUTPUT_ROOT / (job["id"] + "-result.zip")
print("Scope:", job["scope"], "Settings:", job["settings"], "Result:", RESULT)
'''),cell("code",f'''# このノートブック生成時点のワーカーを固定。ユーザーZIP由来のコードは使いません。
WORKER = Path("/content/nagi-seethrough-worker.py")
WORKER.write_text({worker!r}, encoding="utf-8")
command = [sys.executable, str(WORKER), str(JOB), str(RESULT), "--repo", str(REPO), "--work-dir", str(OUTPUT_ROOT / "work")]
if GROUP_OFFLOAD:
    command.append("--group-offload")
subprocess.run(command, check=True)
'''),cell("code",'''# 成功した結果をローカルへダウンロード
with zipfile.ZipFile(RESULT) as archive:
    result = json.loads(archive.read("result.json"))
assert result["status"] == "success"
print("抽出:", [item["name"] for item in result["layers"]])
print("未検出:", result["missing"])
files.download(str(RESULT))
''')]
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"accelerator":"GPU",
        "colab":{"name":"Nagi_SeeThrough.ipynb","provenance":[]},
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":cells}


def write_notebook(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(notebook(),ensure_ascii=False,indent=2),"utf-8")


if __name__=="__main__":write_notebook("colab/Nagi_SeeThrough.ipynb")
