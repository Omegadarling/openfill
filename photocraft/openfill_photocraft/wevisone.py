# SPDX-License-Identifier: Apache-2.0
"""Get per-object alpha mattes from ComfyUI's built-in WeVisOne-VL node."""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import math
import re
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops

from openfill.backend import GenerationCancelled
from openfill.comfy_api import ComfyClient

DEFAULT_QUERY = "all distinct visible foreground objects"
MODEL = re.compile(r"wevisone[_-]vl[_-]([24])b[_-](bf16|int8_convrot)\.safetensors", re.I)


def encoder(client):
    if "WeVisOneDetect" not in client.object_info("WeVisOneDetect"):
        raise ValueError(
            "This ComfyUI needs the WeVisOne-VL Detect node. Update ComfyUI or choose a compatible source."
        )
    choices = []
    for name in client.combo_options("CLIPLoader", "clip_name"):
        match = MODEL.fullmatch(name.replace("\\", "/").split("/")[-1])
        if match:
            choices.append((int(match[1]), match[2].lower() == "bf16", name))
    if not choices:
        raise ValueError("Install a Comfy-Org WeVisOne-VL encoder in ComfyUI/models/text_encoders, then try again.")
    return max(choices)[2]  # pass the loader's exact path back, including its separators


def workflow(name, source, query):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": source}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": name, "type": "stable_diffusion"}},
        "3": {
            "class_type": "WeVisOneDetect",
            "inputs": {
                "clip": ["2", 0],
                "image": ["1", 0],
                "task": "segmentation",
                "query": query,
                "individual_masks": True,
                "max_length": 1024,
            },
        },
        "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
    }


def choose_matte(masks, point):
    """Prefer the smallest object actually under the click; a description selects all matches."""
    best, area = None, None
    for mask in masks:
        peak = mask.getextrema()[1]
        if peak < 8:
            continue
        if point is None:
            best = mask if best is None else ImageChops.lighter(best, mask)
        else:
            xy = tuple(min(n - 1, int(p * n)) for p, n in zip(point, mask.size, strict=True))
            if mask.getpixel(xy) < max(8, round(peak * 0.25)):
                continue
            occupied = sum(mask.histogram()[max(8, round(peak * 0.5)) :])
            if area is None or occupied < area:
                best, area = mask, occupied
    if best is None:
        if point is not None:
            raise ValueError("No object found at that point. Click inside it or describe what to select.")
        raise ValueError("No matching object found. Try a more specific description.")
    return clean_matte(best)


def clean_matte(mask):
    # Suppress barely visible distant specks that otherwise enlarge PhotoCraft's
    # selection bounds; make opaque cores fully editable. Preserve the soft edge.
    return mask.point(lambda alpha: 0 if alpha < 6 else 255 if alpha >= 250 else alpha)


def box_pixels(size, box):
    """Map normalized drag endpoints to a nonempty Pillow box (exclusive right/bottom)."""
    x1, x2 = sorted((box[0], box[2]))
    y1, y2 = sorted((box[1], box[3]))
    return math.floor(x1 * size[0]), math.floor(y1 * size[1]), math.ceil(x2 * size[0]), math.ceil(y2 * size[1])


def choose_box_matte(masks, rect, cut_edges=(False, False, False, False)):
    """Match a box around an object to its whole matte, including holes and soft edges."""
    left, top, right, bottom = rect
    box_area = (right - left) * (bottom - top)
    best = None
    best_score = 0
    for mask in masks:
        peak = mask.getextrema()[1]
        if peak < 8:
            continue
        threshold = max(8, round(peak * 0.5))
        core = mask.point([0] * threshold + [255] * (256 - threshold))
        x1, y1, x2, y2 = core.getbbox()
        # A matte cut off by the temporary crop needs a second look on the full image.
        touches = (x1 == 0, y1 == 0, x2 == mask.width, y2 == mask.height)
        if any(cut and touched for cut, touched in zip(cut_edges, touches, strict=True)):
            continue
        area = core.histogram()[255]
        inside = core.crop(rect).histogram()[255]
        if inside < area * 0.5:
            continue
        # Favor a tightly enclosed instance over a larger subject behind it. Use the
        # silhouette's bounds for fit so thin wires and hollow handles are eligible.
        overlap = (min(right, x2) - max(left, x1)) * (min(bottom, y2) - max(top, y1))
        union = box_area + (x2 - x1) * (y2 - y1) - overlap
        score = (inside / area) * (overlap / union)
        if score > best_score:
            best, best_score = mask, score
    return clean_matte(best) if best is not None else None


def model_frame(image):
    """Limit full-frame inference without losing original pixels for a smaller box."""
    if max(image.size) <= 1536:
        return image
    frame = image.copy()
    frame.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    return frame


class WeVisOne:
    """Use the configured edit server, or an optional dedicated loopback GPU for selection."""

    def __init__(self, profile):
        self.config = Path(profile) / "selection-engine.json"
        self.process = None
        self.cache_key, self.cached_masks = None, ()

    def _stop_owned(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _client(self, app, progress, cancel):
        if not getattr(app, "selection_helper", False):
            if app.settings.backend == "a1111":
                raise ValueError("Choose a ComfyUI engine to select objects with WeVisOne-VL.")
            url = app.server.ensure_running(progress, cancel)
            return ComfyClient(url, timeout=30)
        # This local, optional profile is written by the installer, never by the web API.
        cfg = json.loads(self.config.read_text(encoding="utf-8-sig"))
        port = int(cfg["port"])
        if not 1 <= port <= 65535:
            raise ValueError("Invalid local selection engine port.")
        client = ComfyClient(f"http://127.0.0.1:{port}", timeout=30)
        if client.is_alive():
            return client
        root, data, python = (Path(cfg[key]).resolve(strict=True) for key in ("comfy_dir", "data_dir", "python"))
        progress(0.05, "Starting the local selection engine…")
        if self.process is None or self.process.poll() is not None:
            args = [
                str(python),
                "-u",
                str(root / "main.py"),
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--base-directory",
                str(data),
                "--disable-all-custom-nodes",
                "--offline",
                "--disable-auto-launch",
            ]
            if "cuda_device" in cfg:
                args.extend(("--cuda-device", str(int(cfg["cuda_device"]))))
            with (self.config.parent / "selection-engine.log").open("ab") as log:
                self.process = subprocess.Popen(
                    args,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            atexit.unregister(self._stop_owned)
            atexit.register(self._stop_owned)
        until = time.monotonic() + 120
        while self.process.poll() is None and time.monotonic() < until:
            if cancel():
                raise GenerationCancelled("Selection cancelled.")
            if client.is_alive(timeout=1):
                return client
            time.sleep(0.25)
        raise RuntimeError(
            "The local selection engine could not start. See selection-engine.log in the OpenFill profile."
        )

    def mask(self, app, source, query, point, progress, cancel, *, box=None):
        client = self._client(app, progress, cancel)
        name = encoder(client)
        if box is None:
            masks = self._instances(client, name, model_frame(source), query, progress, cancel)
            return choose_matte(masks, point)

        # Keep surrounding context. For a small item this crop uses the original document
        # pixels, before the full image would be reduced to the model's input limit.
        width, height = source.size
        left, top, right, bottom = box_pixels(source.size, box)
        pad = max(16, math.ceil(max(right - left, bottom - top) / 2))
        x1, y1, x2, y2 = max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)
        crop_size = x2 - x1, y2 - y1
        if crop_size[0] * crop_size[1] < 0.75 * width * height:
            frame = model_frame(source.crop((x1, y1, x2, y2)))
            crop_box = (
                (left - x1) / crop_size[0],
                (top - y1) / crop_size[1],
                (right - x1) / crop_size[0],
                (bottom - y1) / crop_size[1],
            )
            masks = self._instances(client, name, frame, query, progress, cancel, cache=False)
            matte = choose_box_matte(masks, box_pixels(frame.size, crop_box), (x1 > 0, y1 > 0, x2 < width, y2 < height))
            if matte is not None:
                if matte.size != crop_size:
                    matte = matte.resize(crop_size, Image.Resampling.BILINEAR)
                result = Image.new("L", source.size)
                result.paste(matte, (x1, y1))
                return result
            progress(0.1, "Checking the full image for the boxed object…")

        frame = model_frame(source)
        masks = self._instances(client, name, frame, query, progress, cancel)
        matte = choose_box_matte(masks, box_pixels(frame.size, box))
        if matte is None:
            raise ValueError("No object found in that box. Draw a box around the whole object or describe it.")
        return matte

    def _instances(self, client, name, source, query, progress, cancel, *, cache=True):
        if cancel():
            raise GenerationCancelled("Selection cancelled.")
        stream = io.BytesIO()
        source.save(stream, "PNG", compress_level=1)
        data = stream.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        key = (client.base_url, name, digest, query)
        if key == self.cache_key:
            return self.cached_masks
        path = client.upload_image(f"{digest}.png", data, subfolder="openfill/selection")
        prompt_id = client.queue_prompt(workflow(name, path, query))
        history = client.wait_for(
            prompt_id,
            1,
            time.monotonic() + 360,
            progress=progress,
            cancel=cancel,
            wait_for_queue=True,
            shared_queue=True,
        )
        outputs = history.get("outputs", {}).get("5", {}).get("images", [])
        masks = []
        for item in outputs:
            if cancel():
                raise GenerationCancelled("Selection cancelled.")
            if item.get("type") != "temp":
                raise ValueError("The selection engine returned an unexpected preview.")
            raw = client.view(item["filename"], item.get("subfolder", ""), "temp")
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != source.size:
                    raise ValueError("The selection mask does not match the image dimensions.")
                masks.append(image.convert("L"))
        # Bound retained masks. Repeated clicks on the same composite normally need no GPU pass.
        if cache and source.width * source.height * len(masks) <= 48 * 1024 * 1024:
            self.cache_key, self.cached_masks = key, masks
        return masks
