# SPDX-License-Identifier: Apache-2.0
"""The native panel's per-mode engine assignments."""

import tkinter as tk
from tkinter import simpledialog

from . import ui
from .engine_routes import MODES

CUSTOM = "Other ComfyUI URL…"


class EngineAssignments(tk.Frame):
    def __init__(self, parent, bar):
        super().__init__(parent, bg=ui.BG, bd=0, highlightthickness=0)
        self.panel, self.bar = parent, bar
        self.multi = ui.Checkbox(self, "Multi-engine")
        self.multi.pack(anchor="w", pady=(8, 0))
        self.rows = ui.frame(self)
        self.pickers = {}
        self.state_key = None
        for mode in MODES:
            ui.caption(self.rows, mode.title()).pack(anchor="w", pady=(6, 0))
            pick = ui.Select(self.rows, ui.CW, command=lambda label, m=mode: self.choose(m, label))
            pick.pack(anchor="w", pady=(3, 0))
            self.pickers[mode] = pick
        self.multi.bind("<Button-1>", lambda _: self.save(not self.multi.get()))
        self.multi.configure(takefocus=True)
        self.multi.bind("<space>", lambda _: self.save(not self.multi.get()))
        self.refresh()

    def refresh(self):
        data = self.bar.service.routes.describe()
        busy = self.bar.service.busy()
        key = (data, busy)
        if self.state_key == key:
            return
        self.state_key = key
        self.multi.set(data["enabled"])
        if data["enabled"]:
            self.rows.pack(fill="x", pady=(0, 6))
        else:
            self.rows.pack_forget()
        self.names = {}
        for mode, picker in self.pickers.items():
            available = {s["name"]: s["id"] for s in data["choices"] if mode in s["modes"]}
            self.names[mode] = available
            picker.set_values([] if busy else [*available, CUSTOM])
            picker.set(next(name for name, key in available.items() if key == data["routes"][mode]))

    def choose(self, mode, label):
        if label == CUSTOM:
            key = simpledialog.askstring("ComfyUI engine", "ComfyUI server URL", parent=self.bar.root)
            if not key:
                self.state_key = None
                self.refresh()
                return
        else:
            key = self.names[mode][label]
        self.save(True, mode, key)

    def save(self, enabled, mode=None, key=None):
        routes = self.bar.service.routes.describe()["routes"]
        if mode:
            routes[mode] = key
        try:
            self.bar.service.configure_engines(enabled, routes)
            self.bar.refresher.wake.set()
            self.panel.say("")
        except (ValueError, RuntimeError, OSError) as err:
            self.panel.say(str(err), ui.BAD)
        self.state_key = None
        self.panel.sync_source()
