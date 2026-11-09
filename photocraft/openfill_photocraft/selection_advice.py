# SPDX-License-Identifier: Apache-2.0
"""Read-only PhotoCraft pixel sampling for removal suggestions."""

from __future__ import annotations

import hashlib
import json
import threading
from statistics import median


def selection_key(index, name, revision, width, height, bounds, active_layer=None):
    if not bounds or index is None:
        return ""
    fields = [index, name, revision, active_layer, width, height, list(bounds)]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=True).encode()).hexdigest()[:24]


def live_selection_key(client):
    ui = client.call("ui.inspect")
    session = ui.get("session") or {}
    doc = ui.get("document") or {}
    return selection_key(
        session.get("active"),
        doc.get("name"),
        doc.get("revision"),
        doc.get("width"),
        doc.get("height"),
        doc.get("selectionBounds") if doc.get("hasSelection") else None,
        doc.get("activeLayer"),
    )


def inspect_pixels(size, bounds, pixel):
    """Sample the composite around the bounding box. Coordinates are always inside the canvas."""
    width, height = size
    left, top, w, h = bounds
    right, bottom = left + w, top + h
    inset = max(1, min(8, round(min(w, h) / 40)))
    step = max(1, min(5, round(max(width, height) / 450)))
    span = max(2, round(min(w, h) * 0.12))
    ring, differences, inside, jumps, bends = [], [], [], [], []
    previous = {}
    cache = {}

    def read(x, y):
        if not (0 <= x < width and 0 <= y < height):
            return None
        if (x, y) not in cache:
            cache[x, y] = pixel(x, y)
        rgba = cache[x, y]
        return rgba[:3] if rgba[3] >= 0.85 else None

    def outside(side, x, y, dx, dy, outward_x, outward_y):
        a = read(x, y)
        if a is None:
            return
        if side in previous:
            jumps.append(max(abs(v - k) for v, k in zip(a, previous[side], strict=True)))
        previous[side] = a
        ring.append(a)
        b, c = read(x + dx, y + dy), read(x - dy, y + dx)
        neighbors = [n for n in (b, c) if n is not None]
        if neighbors:
            differences.append(max(max(abs(v - k) for v, k in zip(a, n, strict=True)) for n in neighbors))
        # A defocused edge can look flat at pixel scale. Sample the lighting farther out.
        mid = read(x + outward_x, y + outward_y)
        far = read(x + 2 * outward_x, y + 2 * outward_y)
        if mid is not None and far is not None:
            bends.append(max(abs((u + v) / 2 - m) for u, v, m in zip(a, far, mid, strict=True)))

    for i in range(7):
        x = left + int((i + 0.5) * w / 7)
        y = top + int((i + 0.5) * h / 7)
        outside("top", x, top - inset - step, step, 0, 0, -span)
        outside("bottom", x, bottom + inset + step - 1, step, 0, 0, span)
        outside("left", left - inset - step, y, 0, step, -span, 0)
        outside("right", right + inset + step - 1, y, 0, step, span, 0)
    for j in range(4):
        for i in range(4):
            color = read(left + int((i + 0.5) * w / 4), top + int((j + 0.5) * h / 4))
            if color is not None:
                inside.append(color)

    def percentile(values, fraction):
        values = sorted(values)
        return values[round((len(values) - 1) * fraction)] if values else 0.0

    def luminance(c):
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    luminances = [luminance(c) for c in ring]
    return {
        "coverage": w * h / (width * height),
        "aspect": max(w, h) / min(w, h),
        "samples": len(differences),
        "detail": percentile(differences, 0.75),
        "edge": percentile(differences, 0.9),
        "bend": percentile(bends, 0.9),
        "jump": percentile(jumps, 0.9),
        "range": percentile(luminances, 0.9) - percentile(luminances, 0.1),
        "cool": sum(c[2] > c[0] + 0.14 and c[1] > c[0] + 0.09 for c in ring) / max(1, len(ring)),
        "highlights": sum(min(c) > 0.80 for c in inside) / max(1, len(inside)),
        "chroma": median(max(c) - min(c) for c in ring) if ring else 0.0,
    }


class SelectionSampler:
    """Only sample on demand. Coalesce panel polls and discard obsolete work."""

    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.wanted = ""
        self.running = False
        self.completed = ""
        self.result = None

    def request(self, snapshot):
        key = snapshot.get("selectionKey") or ""
        with self.lock:
            self.wanted = key
            if not key:
                self.completed, self.result = "", None
                return {"status": "empty", "tip": "Select the object and any shadow you want removed."}
            if key == self.completed:
                return dict(self.result, key=key)
            if not self.running:
                self.running = True
                threading.Thread(target=self._sample, args=(snapshot, key), name="openfill-advice", daemon=True).start()
        return {"status": "sampling", "tip": "Reading colors and edges around the selection…", "key": key}

    def _sample(self, snapshot, key):
        result = None
        try:
            if live_selection_key(self.client) != key:
                return
            doc = snapshot["document"]

            reads = 0

            def pixel(x, y):
                nonlocal reads
                reads += 1
                if self.wanted != key or (reads % 16 == 0 and live_selection_key(self.client) != key):
                    raise InterruptedError
                return self.client.run("document.sampleColor", x=x, y=y, size=1, sampleLayer="all")

            values = inspect_pixels((doc["width"], doc["height"]), snapshot["selection"], pixel)
            if live_selection_key(self.client) == key:
                result = {"status": "ready", "features": values}
        except InterruptedError:
            pass
        except Exception:  # A manual method is always usable if the host cannot sample.
            result = {"status": "unavailable", "tip": "PhotoCraft could not sample this selection. Choose a Method."}
        finally:
            with self.lock:
                if result is not None and self.wanted == key:
                    self.completed, self.result = key, result
                self.running = False
