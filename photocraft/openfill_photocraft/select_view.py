# SPDX-License-Identifier: Apache-2.0
"""Selection-only controls for the native companion."""

from __future__ import annotations

import base64
import io
import threading
import time
import tkinter as tk

from PIL import Image

from . import ui


class SelectView(tk.Frame):
    def __init__(self, bar, parent):
        super().__init__(parent, bg=ui.BG)
        self.bar = bar
        self.key = self.loading = ""
        self.retry_at = 0.0
        self.photo, self.rect = None, None
        self.drag_start = None
        self.dragged = False
        self.operation = "replace"
        self.query = tk.StringVar()
        self.radius = tk.IntVar(value=4)
        ui.small(
            self,
            "WeVisOne-VL · Click an object, drag a box around it, or describe it. Blue shows the current selection.",
        ).pack(anchor="w")
        self.canvas = tk.Canvas(
            self, bg=ui.WELL, width=ui.CW, height=184, bd=0, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(anchor="w", pady=(6, 8))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Escape>", self._cancel_drag)
        self.canvas.bind("<Unmap>", self._cancel_drag)
        self.ops = ui.Seg(
            self, (("replace", "Replace"), ("add", "Add"), ("subtract", "Subtract")), self._operation, ui.CW, 12
        )
        self.ops.pack(anchor="w")
        self.ops.set(self.operation)
        ui.caption(self, "Describe an object (optional)").pack(anchor="w", pady=(10, 3))
        self.entry = tk.Entry(
            self,
            textvariable=self.query,
            bg=ui.WELL,
            fg=ui.TEXT,
            insertbackground=ui.TEXT,
            relief="flat",
            font=bar.fonts.body,
        )
        self.entry.pack(fill="x", ipady=5)
        self.entry.bind("<Return>", lambda _: self.submit("describe"))
        self.query.trace_add("write", lambda *_: self.buttons())
        row = ui.frame(self)
        row.pack(fill="x", pady=(8, 0))
        self.grow = ui.Button(row, "Grow", lambda: self.submit("grow"), width=60)
        self.grow.pack(side="right")
        self.amount = ui.small(row, "4 px")
        self.amount.pack(side="right", padx=6)
        ui.small(row, "Grow selection").pack(side="left")
        tk.Scale(
            row,
            from_=1,
            to=64,
            orient="horizontal",
            variable=self.radius,
            showvalue=False,
            command=lambda v: self.amount.configure(text=f"{int(float(v))} px"),
            bg=ui.BG,
            fg=ui.TEXT,
            troughcolor=ui.WELL,
            highlightthickness=0,
            sliderlength=14,
        ).pack(side="left", fill="x", expand=True)

    def _operation(self, value):
        self.operation = value
        self.ops.set(value)
        self.buttons()

    def buttons(self):
        pc = self.bar.watcher.snapshot()
        idle = not self.bar.service.busy() and not self.bar.service.layers
        ready = bool(idle and self.key and self.key == pc.get("documentKey"))
        self.grow.set(disabled=not (idle and pc.get("selection")))
        self.bar.go.set(
            disabled=not (ready and self.query.get().strip() and (pc.get("selection") or self.operation != "subtract"))
        )

    def update_view(self):
        if self.bar.collapsed or "edit" in self.bar.sections.collapsed:
            return
        pc = self.bar.watcher.snapshot()
        key = pc.get("documentKey") or ""
        blocked = bool(self.bar.service.layers)
        if self.key != key or blocked:
            self._cancel_drag()
            self.key, self.rect = "", None
            self.canvas.delete("all")
        if not self.key:
            self.canvas.delete("hint")
            msg = (
                "Keep or discard the current choices first."
                if blocked
                else ("Reading the canvas…" if key else "Open an image in PhotoCraft.")
            )
            self.canvas.create_text(ui.CW / 2, 92, text=msg, fill=ui.DIM, tags="hint")
        self.buttons()
        if (
            not key
            or self.key == key
            or self.loading
            or blocked
            or self.bar.service.busy()
            or time.monotonic() < self.retry_at
        ):
            return
        self.loading = key

        def read():
            try:
                result = self.bar.service.selection_preview(key)
                with Image.open(io.BytesIO(base64.b64decode(result["image"]))) as im:
                    im.thumbnail((ui.CW, 184), Image.Resampling.LANCZOS)
                    picture = im.copy()
                self.bar.calls.put((self._ready, (key, picture, "")))
            except Exception as exc:
                self.bar.calls.put((self._ready, (key, None, str(exc))))

        threading.Thread(target=read, name="openfill-selection-preview", daemon=True).start()

    def _ready(self, key, picture, error):
        self.loading = ""
        if self.bar.tab.get() != "select" or self.bar.watcher.snapshot().get("documentKey") != key:
            return
        if error:
            self.retry_at = time.monotonic() + 2
            self.bar._say(error, ui.BAD)
            return
        self.canvas.delete("all")
        self.photo = ui.photo(picture)
        left, top = (ui.CW - picture.width) // 2, (184 - picture.height) // 2
        self.canvas.create_image(left, top, anchor="nw", image=self.photo)
        self.rect = left, top, picture.width, picture.height
        self.key = key
        self.buttons()

    def _point(self, event):
        if self.rect is None:
            return None
        left, top, width, height = self.rect
        point = [(event.x - left) / width, (event.y - top) / height]
        return point if all(0 <= p < 1 for p in point) else None

    def _click(self, event):
        point = self._point(event)
        if point is not None:
            self.submit("click", point)

    def _press(self, event):
        self._cancel_drag()
        if self.key and not self.bar.service.busy() and self._point(event) is not None:
            self.canvas.focus_set()
            self.drag_start = (event.x, event.y)

    def _drag(self, event):
        if self.drag_start is None or self.rect is None:
            return None
        x, y = self.drag_start
        if not self.dragged and max(abs(event.x - x), abs(event.y - y)) < 4:
            return None
        self.dragged = True
        left, top, width, height = self.rect
        end_x = max(left, min(left + width, event.x))
        end_y = max(top, min(top + height, event.y))
        x1, x2 = sorted((x, end_x))
        y1, y2 = sorted((y, end_y))
        self.canvas.delete("drag-box")
        self.canvas.create_rectangle(x1, y1, x2, y2, tags="drag-box", outline=ui.ACCENT, width=2, dash=(4, 2))
        if x1 == x2 or y1 == y2:
            return None
        return [(x1 - left) / width, (y1 - top) / height, (x2 - left) / width, (y2 - top) / height]

    def _release(self, event):
        if self.drag_start is None:
            return
        box = self._drag(event)
        dragged = self.dragged
        self._cancel_drag()
        if box is not None:
            self.submit("box", box)
        elif not dragged:
            self._click(event)

    def _cancel_drag(self, _event=None):
        if self.drag_start is None:
            return None
        self.drag_start, self.dragged = None, False
        self.canvas.delete("drag-box")
        return "break"

    def submit(self, action, target=None):
        if self.bar.service.busy():
            return
        params = {"action": action, "key": self.key, "operation": self.operation, "query": self.query.get().strip()}
        if action == "grow":
            params.update(key=self.bar.watcher.snapshot().get("documentKey"), pixels=self.radius.get())
        elif action == "click":
            params["point"] = target
        elif action == "box":
            params["box"] = target
        try:
            self.bar.service.start("select", params)
        except (ValueError, RuntimeError) as exc:
            self.bar._say(str(exc), ui.BAD)
