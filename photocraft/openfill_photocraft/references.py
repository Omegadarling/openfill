# OpenFill for PhotoCraft
# SPDX-License-Identifier: Apache-2.0
"""Session-only reference attachments shared by the native and web panels."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_REFERENCES = 15
MAX_UPLOAD = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_PIXELS = 40_000_000


@dataclass(frozen=True)
class Reference:
    id: str
    name: str
    width: int
    height: int
    png: bytes

    def row(self):
        return {"id": self.id, "name": self.name, "width": self.width, "height": self.height}


def decode_reference(name: str, data: bytes) -> Reference:
    name = name.replace("\\", "/").rsplit("/", 1)[-1][:240] or "Reference image"
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError(f"{name} must be an image of up to 16 MB.")
    try:
        with Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f"{name} is larger than 40 megapixels.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"{name} is animated. Choose a still image.")
            oriented = ImageOps.exif_transpose(image)
            oriented.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
            cleaned = oriented.convert("RGBA" if has_alpha else "RGB")
            cleaned.info.clear()
            output = io.BytesIO()
            cleaned.save(output, format="PNG")
            return Reference(uuid4().hex, name, cleaned.width, cleaned.height, output.getvalue())
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not read {name}. Choose a valid PNG, JPEG or WebP image.") from exc


class ReferenceImages:
    def __init__(self):
        self._lock = Lock()
        self._items: list[Reference] = []

    def rows(self):
        with self._lock:
            return [item.row() for item in self._items]

    def snapshot(self) -> tuple[bytes, ...]:
        with self._lock:
            return tuple(item.png for item in self._items)

    def add(self, name, data):
        return self._append([decode_reference(name, data)])

    def add_files(self, paths):
        paths = tuple(Path(path) for path in paths)
        if len(paths) > MAX_REFERENCES:
            raise ValueError(f"Add at most {MAX_REFERENCES} reference images.")
        new = []
        for path in paths:
            try:
                with path.open("rb") as file:
                    data = file.read(MAX_UPLOAD + 1)
            except OSError as exc:
                raise ValueError(f"Could not open {path.name}.") from exc
            new.append(decode_reference(path.name, data))
        return self._append(new)

    def _append(self, new):
        with self._lock:
            items = self._items + new
            if len(items) > MAX_REFERENCES:
                raise ValueError(f"Add at most {MAX_REFERENCES} reference images.")
            if sum(len(item.png) for item in items) > MAX_TOTAL:
                raise ValueError("References exceed the combined size limit. Remove an image and retry.")
            self._items = items
            return [item.row() for item in items]

    def remove(self, reference_id):
        with self._lock:
            self._items = [item for item in self._items if item.id != reference_id]
            return [item.row() for item in self._items]

    def order(self, ids):
        with self._lock:
            by_id = {item.id: item for item in self._items}
            if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids)
                    or len(ids) != len(by_id) or set(ids) != by_id.keys()):
                raise ValueError("The reference list changed. Refresh it before reordering.")
            self._items = [by_id[reference_id] for reference_id in ids]
            return [item.row() for item in self._items]
