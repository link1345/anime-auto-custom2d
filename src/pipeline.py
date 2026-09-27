"""Editable template decomposition. No claim of semantic AI segmentation/inpainting."""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .project import uid

# Normalized polygons for a front-facing upper-body illustration. These are a
# visible starting template, not inferred anatomy. All can be edited in the UI.
REGIONS = [
    ("body", "胴体・服（未分離）", [(0, .30), (1, .30), (1, 1), (0, 1)], [.5, .40], 0),
    ("armLeft", "右腕（画面左）", [(0, .35), (.29, .35), (.34, .65), (.16, 1), (0, 1)], [.27, .37], 1),
    ("armRight", "左腕（画面右）", [(.71, .35), (1, .35), (1, 1), (.84, 1), (.66, .65)], [.73, .37], 1),
    ("head", "頭・顔", [(0, 0), (1, 0), (1, .32), (.60, .32), (.59, .35), (.41, .35), (.40, .32), (0, .32)], [.5, .31], 2),
    ("hairLeft", "横髪・右", [(.20, .15), (.35, .10), (.39, .25), (.44, .36), (.28, .37), (.20, .30)], [.32, .15], 3),
    ("hairRight", "横髪・左", [(.65, .10), (.80, .15), (.80, .30), (.72, .37), (.56, .36), (.61, .25)], [.68, .15], 3),
]


def image_from_bytes(payload):
    image = Image.open(io.BytesIO(payload))
    if image.width * image.height > 20_000_000:
        raise ValueError("Image exceeds 20 megapixels")
    image = image.convert("RGBA")
    return image


def validate_face_image(image, face_fills_frame=False):
    """User's observed See-through constraint, not a universal model guarantee."""
    if image.width != image.height or image.width < 2160:
        raise ValueError("See-through顔入力は顔を大きく収めた正方形2160×2160以上が必要です。FFmpeg拡大を有効にしてください。")
    if not face_fills_frame:
        raise ValueError("顔が画面を大きく占め、目・眉・口・顎が見えていることを確認してください。")
    return {"width": image.width, "height": image.height, "faceFillsFrame": True,
            "upscaledByApp": False, "basis": "user-observed-see-through-requirement"}


def mesh_for(polygon, depth=0.0):
    # Fan triangulation is safe for the generated rectangular grid, not arbitrary masks.
    xs, ys = zip(*polygon)
    vertices = []
    for row in range(9):
        for col in range(9):
            x = min(xs) + (max(xs) - min(xs)) * col / 8
            y = min(ys) + (max(ys) - min(ys)) * row / 8
            vertices.append([x, y, depth + .025 * math.sin(col / 8 * math.pi)])
    triangles = []
    for row in range(8):
        for col in range(8):
            a = row * 9 + col
            triangles.extend([[a, a + 1, a + 9], [a + 1, a + 10, a + 9]])
    return {"vertices": vertices, "triangles": triangles, "version": 1, "source": "template"}


def generate_parts(image, folder, templates=None):
    folder = Path(folder)
    (folder / "assets").mkdir(parents=True, exist_ok=True)
    templates = templates or [dict(id=role, role=role, name=name, polygon=[list(v) for v in polygon],
        pivot=pivot, depth=depth * .01, locked=False, edits={})
        for role, name, polygon, pivot, depth in REGIONS]
    # Rendering proxies never replace the retained full-resolution original.
    image = image.copy()
    image.thumbnail((1024, 1536))
    w, h = image.size
    # Exclusive labels exactly preserve original pixel ownership at the rest pose.
    labels = Image.new("I", image.size, 0)
    draw = ImageDraw.Draw(labels)
    for index, part in enumerate(templates):
        draw.polygon([(int(x*w), int(y*h)) for x, y in part["polygon"]], fill=index+1)
    label_array = np.asarray(labels)
    rgba = np.asarray(image)
    parts = []
    for index, template in enumerate(templates):
        part = dict(template)
        pixels = rgba.copy()
        pixels[:, :, 3] = np.where(label_array == index+1, rgba[:, :, 3], 0)
        filename = f"{part['id']}-{uid()[:8]}.png"
        Image.fromarray(pixels).save(folder / "assets" / filename)
        part.update(image=f"assets/{filename}", mesh=mesh_for(part["polygon"], part["depth"]),
                    provenance="template", completion="not-inpainted",
                    physics={"enabled": part["role"].startswith("hair"), "stiffness": 36, "damping": 9, "strength": .7})
        parts.append(part)
    return parts


def create_project(store, payload, name="新しいキャラクター"):
    image = image_from_bytes(payload)
    project_id = uid()
    folder = store.folder(project_id)
    (folder / "assets").mkdir(parents=True)
    image.save(folder / "assets/source.png")
    p = {"schemaVersion": 1, "id": project_id, "name": name[:100], "revision": 1,
         "meshVersion": 1, "size": list(image.size), "source": "assets/source.png",
         "parts": generate_parts(image, folder), "expressions": [], "attachments": [],
         "outfit": [], "stale": [], "createdAt": time.time(),
         "face": {"leftEye": [.433, .194], "rightEye": [.568, .194], "mouth": [.502, .251], "skin": "#ffe4d2"},
         "warnings": ["仮の領域分割です。顔・関節位置と分割マスクを確認してください。",
                      "隠れ部分は未補完。動かすと境界に穴が開く場合があります。",
                      "深度はテンプレート。MoGe3推定は別ジョブで実行してください。",
                      "腕は一枚パーツの試作。肘・手首・指の追跡は未対応。"],
         "backend": {"name": "editable-template", "version": 1}}
    store.save(p)
    return p
