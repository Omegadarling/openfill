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
        self.operation = "replace"
        self.query = tk.StringVar()
        self.radius = tk.IntVar(value=4)
        ui.small(self, "WeVisOne-VL · Click an object or describe it. Blue shows the current selection.").pack(
            anchor="w"
        )
        self.canvas = tk.Canvas(
            self, bg=ui.WELL, width=ui.CW, height=184, bd=0, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(anchor="w", pady=(6, 8))
        self.canvas.bind("<Button-1>", self._click)
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

    def _click(self, event):
        if self.rect:
            left, top, width, height = self.rect
            x, y = (event.x - left) / width, (event.y - top) / height
            if 0 <= x < 1 and 0 <= y < 1:
                self.submit("click", [x, y])

    def submit(self, action, point=None):
        if self.bar.service.busy():
            return
        params = {"action": action, "key": self.key, "operation": self.operation, "query": self.query.get().strip()}
        if action == "grow":
            params.update(key=self.bar.watcher.snapshot().get("documentKey"), pixels=self.radius.get())
        elif action == "click":
            params["point"] = point
        try:
            self.bar.service.start("select", params)
        except (ValueError, RuntimeError) as exc:
            self.bar._say(str(exc), ui.BAD)
