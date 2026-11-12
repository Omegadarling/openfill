# SPDX-License-Identifier: Apache-2.0
"""The native panel's per-mode engine assignments."""

import threading
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


class ServerFinder(tk.Frame):
    """Search in the background or save a known address, including an offline server."""

    def __init__(self, panel, bar):
        super().__init__(panel, bg=ui.BG, bd=0, highlightthickness=0)
        self.panel, self.bar = panel, bar
        buttons = ui.frame(self)
        buttons.pack(anchor="w")
        self.find_button = ui.Button(buttons, "Find servers", self.find)
        self.find_button.pack(side="left")
        ui.Button(buttons, "Add server", self.show_form).pack(side="left", padx=(8, 0))
        self.message = ui.small(self, "")
        self.results = ui.frame(self)
        self.form = ui.frame(self)
        ui.caption(self.form, "ComfyUI address").pack(anchor="w")
        self.address = ui.Field(self.form, ui.CW, "studio.local:8188 or ssh://studio:8188")
        self.address.pack(anchor="w", pady=(4, 8))
        ui.caption(self.form, "Name (optional)").pack(anchor="w")
        self.name = ui.Field(self.form, ui.CW)
        self.name.pack(anchor="w", pady=(4, 8))
        actions = ui.frame(self.form)
        actions.pack(anchor="w")
        ui.Button(actions, "Save server", self.save, "accent").pack(side="left")
        ui.Button(actions, "Cancel", self.form.pack_forget).pack(side="left", padx=(8, 0))

    def say(self, text, color=ui.DIM):
        self.message.configure(text=text, fg=color)
        self.message.pack(fill="x", pady=(6, 0))

    def show_form(self, address="", name=""):
        self.address.set(address)
        self.name.set(name)
        self.form.pack(fill="x", pady=(8, 0))
        self.address.entry.focus_set()

    def save(self):
        try:
            row = self.bar.service.add_server(self.name.get(), self.address.get())
        except (ValueError, OSError, RuntimeError) as err:
            self.say(str(err), ui.BAD)
            return
        self.form.pack_forget()
        self.results.pack_forget()
        self.say(f"{row['name']} is saved. Choose it in an engine menu.", ui.OK)
        self.panel.sync_source()

    def find(self):
        self.find_button.set(disabled=True)
        self.form.pack_forget()
        self.results.pack_forget()
        self.say("Looking for ComfyUI on this computer and the LAN…")

        def work():
            try:
                rows = self.bar.service.find_servers()
                self.bar.call_soon(self.show_results, rows)
            except Exception as err:
                self.bar.call_soon(self.say, str(err), ui.BAD)
            finally:
                self.bar.call_soon(lambda: self.find_button.set(disabled=False))

        threading.Thread(target=work, name="openfill-find-servers", daemon=True).start()

    def show_results(self, rows):
        for child in self.results.winfo_children():
            child.destroy()
        if not rows:
            self.say("No servers answered. Use Add server for another port, an SSH connection, or an offline server.")
            return
        self.say("ComfyUI servers that answered", ui.OK)
        for row in rows:
            line = ui.frame(self.results)
            line.pack(fill="x", pady=(4, 0))
            if not row["saved"]:
                ui.Button(line, "Add", lambda r=row: self.show_form(r["address"], r["name"])).pack(side="right")
            label = row["name"] + (" · saved" if row["saved"] else "")
            ui.small(line, label, wrap=ui.CW - 55).pack(side="left")
        self.results.pack(fill="x")
