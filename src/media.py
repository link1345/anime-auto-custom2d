"""Lossless source retention and explicitly requested FFmpeg scaling."""
import os
from pathlib import Path
import shutil
import subprocess
from PIL import Image


def ffmpeg_path():
    configured = os.environ.get("AAC2D_FFMPEG")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", Path.home()/"AppData/Local"))/"Microsoft/WinGet/Packages"
    matches = sorted(packages.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
    if matches:
        return str(matches[-1])
    raise ValueError("FFmpegが見つかりません。PATHまたはAAC2D_FFMPEGに実行ファイルを設定してください。")


def upscale_face(source, destination, size=2160):
    with Image.open(source) as image:
        original = list(image.size)
        if image.width != image.height:
            raise ValueError("顔を正方形に切り出してから登録してください。縦横比は変えません。")
    if type(size) is not int or not 2160 <= size <= 4096:
        raise ValueError("拡大先は2160〜4096pxです")
    size = max(size, original[0])
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("原画像は上書きしません")
    command = [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
               "-vf", f"scale={size}:{size}:flags=lanczos,format=rgba", "-frames:v", "1", str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise ValueError("FFmpeg: " + exc.stderr.decode("utf-8", errors="replace")[-1500:]) from exc
    with Image.open(destination) as result:
        if result.size != (size,size) or result.mode != "RGBA":
            raise ValueError("FFmpeg output failed dimension/alpha validation")
    return {"tool":"ffmpeg", "filter":"lanczos", "originalSize":original,
            "outputSize":[size,size], "upscaledByApp":size>original[0]}
