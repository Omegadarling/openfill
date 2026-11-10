# SPDX-License-Identifier: Apache-2.0
"""Snapshot the artist's canvas, then apply one native, undoable selection change."""

from __future__ import annotations

import base64
import io
import math

from PIL import Image

from .bridge import PCImage
from .selection_advice import document_key
from .wevisone import DEFAULT_QUERY, WeVisOne

STALE = "The document or selection changed. Use the updated preview and try again."
OPERATIONS = {"replace": "new", "add": "add", "subtract": "subtract"}


def key_for(index, doc):
    return document_key(
        index,
        doc.get("name"),
        doc.get("revision"),
        doc.get("width"),
        doc.get("height"),
        doc.get("selectionBounds") if doc.get("hasSelection") else None,
        doc.get("activeLayer"),
    )


def active(client, expected):
    view = client.call("ui.inspect")
    doc = view.get("document") or {}
    if not doc:
        raise ValueError("Open an image in PhotoCraft first.")
    index = (view.get("session") or {}).get("active")
    if not expected or key_for(index, doc) != expected:
        raise ValueError(STALE)
    return index, doc


class Selection:
    def __init__(self, bridge):
        self.bridge = bridge
        self.model = WeVisOne(bridge.profile)
        self.key, self.source, self.preview = "", None, None

    def capture(self, key):
        client = self.bridge.client
        index, doc = active(client, key)
        if self.key == key:
            return self.preview

        # The preview and inference receive exactly the same unmarked composite.
        image = PCImage(self.bridge, doc["name"], index)
        full, mask = self.bridge._snapshot(image)
        active(client, key)
        rgba = full.copy()
        rgba.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        source = Image.new("RGB", rgba.size, (112, 112, 112))
        source.paste(rgba, mask=rgba.getchannel("A"))
        thumb = source.copy()
        thumb.thumbnail((720, 500), Image.Resampling.LANCZOS)
        if mask is not None:
            overlay = mask.resize(thumb.size, Image.Resampling.LANCZOS).point(lambda a: round(a * 0.48))
            thumb.paste((55, 175, 255), (0, 0, *thumb.size), overlay)
        stream = io.BytesIO()
        thumb.save(stream, "JPEG", quality=90)
        preview = {
            "key": key,
            "width": doc["width"],
            "height": doc["height"],
            "image": base64.b64encode(stream.getvalue()).decode("ascii"),
        }
        self.key, self.source, self.preview = key, source, preview
        return preview

    def validate(self, params):
        action = params.get("action")
        _, doc = active(self.bridge.client, params.get("key"))
        if action == "grow":
            if not doc.get("hasSelection"):
                raise ValueError("Make a selection before growing it.")
            pixels = params.get("pixels")
            if type(pixels) is not int or not 1 <= pixels <= 64:
                raise ValueError("Choose a grow distance from 1 to 64 pixels.")
            return
        if action not in ("click", "describe") or params.get("operation", "replace") not in OPERATIONS:
            raise ValueError("Choose a valid selection action.")
        if self.key != params.get("key") or self.source is None:
            raise ValueError(STALE)
        query = params.get("query", "")
        if not isinstance(query, str) or len(query) > 500 or (action == "describe" and not query.strip()):
            raise ValueError("Describe what to select in 1–500 characters.")
        if action == "click":
            point = params.get("point")
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or not all(type(p) in (int, float) and math.isfinite(p) and 0 <= p < 1 for p in point)
            ):
                raise ValueError("Click inside the image preview.")
        if params.get("operation") == "subtract" and not doc.get("hasSelection"):
            raise ValueError("Make a selection before subtracting from it.")

    def run(self, app, params, progress, cancel, begin_commit):
        self.validate(params)
        key, action = params["key"], params["action"]
        client = self.bridge.client
        if action == "grow":
            begin_commit()
            active(client, key)
            client.run("select.modify.expand", radius=params["pixels"])
            return f"Selection grew by {params['pixels']} px."
        progress(0.05, "Finding the object with WeVisOne-VL…")
        mask = self.model.mask(
            app,
            self.source,
            params.get("query", "").strip() or DEFAULT_QUERY,
            params.get("point") if action == "click" else None,
            progress,
            cancel,
        )
        index, doc = active(client, key)
        size = (doc["width"], doc["height"])
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.BILINEAR)
        guide = Image.new("RGBA", size, "white")
        guide.putalpha(mask)
        operation = OPERATIONS[params.get("operation", "replace")] if doc.get("hasSelection") else "new"
        begin_commit()
        active(client, key)
        scratch, layer = self.bridge._open(guide, "mask")
        try:
            # Opening the scratch tab is deliberate. Do not redirect the result if the artist
            # has since changed the target, or switched away from our scratch tab.
            if (
                self.bridge.session().get("active") != scratch
                or key_for(index, client.run("document.inspect", document=index)) != key
            ):
                raise ValueError(STALE)
            client.run("document.activate", document=index)
            active(client, key)
            client.run(
                "select.loadSelection", channel="transparency", document=scratch, layer=layer, operation=operation
            )
        finally:
            self.bridge._close(scratch)
        return {
            "new": "Object selected.",
            "add": "Added to the selection.",
            "subtract": "Subtracted from the selection.",
        }[operation]
