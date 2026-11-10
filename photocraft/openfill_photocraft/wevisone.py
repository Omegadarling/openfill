# SPDX-License-Identifier: Apache-2.0
"""Get per-object alpha mattes from ComfyUI's built-in WeVisOne-VL node."""

from __future__ import annotations

import atexit
import hashlib
import io
import json
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
    # Suppress barely visible distant specks that otherwise enlarge PhotoCraft's
    # selection bounds; make opaque cores fully editable. Preserve the soft edge.
    return best.point(lambda alpha: 0 if alpha < 6 else 255 if alpha >= 250 else alpha)


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
        if not self.config.is_file():
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

    def mask(self, app, source, query, point, progress, cancel):
        client = self._client(app, progress, cancel)
        name = encoder(client)
        stream = io.BytesIO()
        source.save(stream, "PNG", compress_level=1)
        data = stream.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        key = (client.base_url, name, digest, query)
        if key == self.cache_key:
            return choose_matte(self.cached_masks, point)
        if cancel():
            raise GenerationCancelled("Selection cancelled.")
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
        if source.width * source.height * len(masks) <= 48 * 1024 * 1024:
            self.cache_key, self.cached_masks = key, masks
        return choose_matte(masks, point)
