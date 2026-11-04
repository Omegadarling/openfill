# OpenFill for PhotoCraft
# SPDX-License-Identifier: Apache-2.0
"""Models and settings, behind the panel's gear: three sections.

- Models: this computer in one line, the installed models (each with a trash button that
  uninstalls it), the ones to download (Download, Resume, or the progress and a button that
  stops it), and the Hugging Face key FLUX needs.
- Engine: ComfyUI's folder, when it starts and stops, extra options, and Check, Stop engine
  and Logs.
- Generation: steps, prompt strictness, repaint margin and edge softness.

The views only read the refresher's facts and the settings on the UI thread; downloads,
removals, checks and saves run in a worker thread and report back through ``Bar.call_soon``.
"""

from __future__ import annotations

import threading
import math
import tkinter as tk
from typing import TYPE_CHECKING, Any

from openfill.config import FIBO_CONTEXT_MODES, FIBO_EDIT_MODES, validate_fibo_settings

from . import ui
from .ui import BAD, BG, CW, DIM, OK

if TYPE_CHECKING:
    from .bar import Bar

# Settings the gear view edits: (key, label, kind). The rest stay in config.json.
SETTINGS = (
    ("autostart", "Start the AI engine with PhotoCraft", bool),
    ("idle_timeout_minutes", "Stop after idle", int),
    ("steps", "Steps", int),
    ("cfg", "Prompt strictness (CFG)", float),
    ("grow_percent", "Repaint margin (%)", float),
    ("feather_percent", "Edge softness (%)", float),
    ("auto_feather", "Auto-feather selection", bool),
    ("auto_feather_percent", "Auto-feather area (%)", float),
    ("fibo_edit_mode", "FIBO edit mode", str),
    ("fibo_context_mode", "FIBO image context", str),
    ("fibo_context_percent", "Surrounding context (%)", float),
    ("fibo_boundary_match", "Match boundary colors", bool),
    ("fibo_boundary_strength", "Boundary color matching (%)", float),
    ("extra_server_args", "Extra ComfyUI options", str),
)
SECTIONS = (("models", "Models"), ("engine", "Engine"), ("generation", "Generation"))
BUSY = ("Downloading", "Waiting", "Installing", "Removing", "Working")


def field(parent: tk.Misc, label: str, control_factory, gap: int = 6) -> tuple[tk.Frame, Any]:
    """An uppercase label above a control."""
    box = ui.frame(parent)
    ui.caption(box, label).pack(anchor="w")
    control = control_factory(box)
    control.pack(anchor="w", pady=(gap, 0))
    return box, control


class GearView(tk.Frame):
    """Models and settings: the section control and the three sections."""

    def __init__(self, bar: Bar, parent: tk.Misc) -> None:
        super().__init__(parent, bg=BG, bd=0, highlightthickness=0)
        self.bar = bar
        self.section = "models"
        # the section control stays under the header; only the section below it scrolls
        self.tabs = ui.Seg(bar.gear_bar, SECTIONS, self._section_picked, CW, 13)
        self.tabs.pack(anchor="w", padx=16, pady=(12, 0))
        self.models = ModelsView(bar, self)
        self.engine = EngineView(bar, self)
        self.generation = GenerationView(bar, self)
        self._loaded = False

    def _section_picked(self, section: str) -> None:
        with ui.frozen(self.bar.root):
            self.show_section(section)

    def show_section(self, section: str) -> None:
        if section == "settings":
            section = "engine"
        self.section = section
        self.tabs.set(section)
        for name, view in (("models", self.models), ("engine", self.engine), ("generation", self.generation)):
            if name == section:
                view.pack(fill="x", anchor="w")
            else:
                view.pack_forget()
        if section == "models":
            self.models.update_view(self.bar.refresher.get())  # built already: this only updates it
        self.bar._layout_footer()

    def update_view(self, facts: dict[str, Any]) -> None:
        """Keep the models list current, shown or not, so opening it shows it at once."""
        self.models.update_view(facts)
        self.engine.sync_source()

    def say(self, text: str, color: str = OK) -> None:
        {"models": self.models, "engine": self.engine, "generation": self.generation}[self.section].say(text, color)

    # the settings ------------------------------------------------------------------------

    def load(self) -> None:
        settings = self.bar.service.app.settings
        self.engine.folder.set(settings.comfy_dir or "")
        self.engine.autostart.set(bool(settings.autostart))
        self.engine.idle.set(str(settings.idle_timeout_minutes))
        self.engine.extra.set(str(settings.extra_server_args or ""))
        for key, entry in self.generation.fields.items():
            entry.set(str(getattr(settings, key)))
        self.generation.load_auto_feather(settings)
        self.generation.load_fibo(settings)

    def values(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "autostart": self.engine.autostart.get(),
            "idle_timeout_minutes": self.engine.idle.get(),
            "extra_server_args": self.engine.extra.get(),
        }
        raw.update({key: entry.get() for key, entry in self.generation.fields.items()})
        raw.update(self.generation.auto_feather_values())
        raw.update(self.generation.fibo_values())
        out = {}
        for key, label, kind in SETTINGS:
            try:
                out[key] = kind(raw[key]) if kind is not bool else bool(raw[key])
            except ValueError as err:
                raise ValueError(f"{label}: {raw[key]!r} is not a number.") from err
        if not math.isfinite(out["auto_feather_percent"]):
            raise ValueError("Auto-feather area must be a finite percentage.")
        out["auto_feather_percent"] = max(0.0, min(100.0, out["auto_feather_percent"]))
        return validate_fibo_settings(out)

    def save(self) -> None:
        app = self.bar.service.app
        try:
            values = self.values()
        except ValueError as err:
            self.say(str(err), BAD)
            return
        folder = self.engine.folder.get().strip()
        try:
            if folder != (app.settings.comfy_dir or ""):
                app.use_comfy_dir(folder)
                self.bar.service.rescan_models()
                self.bar.service.warm_up()
            for key, value in values.items():
                setattr(app.settings, key, value)
            if app.settings.remote_fibo:
                app.settings.autostart = False
            app.settings.save(app.paths.config)
            app.reload_settings()
        except Exception as err:  # noqa: BLE001 - says what is wrong
            self.say(str(err), BAD)
            return
        self.bar.refresher.wake.set()
        self.say("Settings saved.", OK)


class ModelsView(tk.Frame):
    """This computer, the installed models and the ones to download, and the Hugging Face key."""

    def __init__(self, bar: Bar, parent: tk.Misc) -> None:
        super().__init__(parent, bg=BG, bd=0, highlightthickness=0)
        self.bar = bar
        self.hardware = ui.small(self, "Looking...")
        self.hardware.pack(fill="x", pady=(12, 0))
        self.note = ui.small(self, "")
        self.installed_title = ui.caption(self, "Installed")
        self.installed = ui.Box(self)
        self.available_title = ui.caption(self, "Available")
        self.available = ui.Box(self)
        self.key_box = ui.frame(self)
        ui.caption(self.key_box, "Hugging Face key (for FLUX)").pack(anchor="w")
        row = ui.frame(self.key_box)
        row.pack(fill="x", pady=(6, 0))
        save = ui.Button(row, "Save", self._save_token)
        self.token = ui.Field(row, CW - 8 - save.w, "hf_...", secret=True, on_enter=self._save_token)
        self.token.pack(side="left")
        save.pack(side="left", padx=(8, 0))
        self.key_box.pack(fill="x", pady=(12, 0))
        if bar.service.app.settings.remote_fibo:
            # Keep the empty frame packed: _build uses it as the list insertion
            # anchor. Hiding the frame itself breaks Tk's pack(before=...).
            for child in self.key_box.winfo_children():
                child.pack_forget()
        self._shown: tuple = ()
        self._rows: dict[str, dict[str, Any]] = {}

    def say(self, text: str, color: str = OK) -> None:
        if not text:
            self.note.pack_forget()
            return
        self.note.configure(text=text, fg=color)
        self.note.pack(fill="x", pady=(8, 0), after=self.hardware)

    def update_view(self, facts: dict[str, Any]) -> None:
        hardware = facts.get("hardware") or ""
        line = hardware or facts.get("comfy_problem") or ("Looking..." if not facts.get("error") else facts["error"])
        if self.hardware.cget("text") != line:
            self.hardware.configure(text=line)
        rows = [r for r in facts.get("rows") or [] if r["runnable"] or r["installed"]]
        installed = [r for r in rows if r["installed"]]
        installed.sort(key=lambda r: (not r["runnable"], not r["active"], r["name"].lower()))
        available = sorted((r for r in rows if not r["installed"]), key=lambda r: r["name"].lower())
        order = (tuple(r["id"] for r in installed), tuple(r["id"] for r in available))
        if order != self._shown:  # which models are where: lay both lists out again
            self._build(installed, available)
            self._shown = order  # record only after the widgets were built successfully
        else:  # the same rows: rebuild only a row whose buttons change (a download starts, ...)
            for box, rows in ((self.installed, installed), (self.available, available)):
                for i, row in enumerate(rows):
                    if self._rows.get(row["id"], {}).get("key") != self._row_key(row):
                        old = self._rows[row["id"]]["frame"]
                        self._row(box.inner, row, first=i == 0, before=old)
                        old.destroy()
                box.lift_corners()
        current = facts.get("current") or {}
        for row in installed + available:
            parts = self._rows.get(row["id"])
            if not parts:
                continue
            status = row.get("statusText", "")
            if "percent" in parts:
                pct = status.split()[-1] if status.startswith("Downloading") else ("Waiting" if status else "")
                if parts["percent"].cget("text") != pct:
                    parts["percent"].configure(text=pct)
            if "bar" in parts:
                total = current.get("total") or 0
                if current.get("model_id") == row["id"] and total:
                    parts["bar"].set((current.get("done") or 0) / total)

    @staticmethod
    def _mode(row: dict[str, Any]) -> str:
        status = row.get("statusText", "")
        if status.startswith(("Downloading", "Waiting")) and not row["installed"]:
            return "busy"
        if status.startswith(BUSY):
            return "working"
        return "idle"

    def _row_key(self, row: dict[str, Any]) -> tuple:
        """What a row's widgets depend on (not the percent, which changes in place)."""
        status = row.get("statusText", "")
        mode = self._mode(row)
        return (
            row["name"],
            row.get("size"),
            row["installed"],
            row["partial"],
            mode,
            status if mode == "working" else status.startswith("Downloading"),
            bool(row.get("missing_models")),
            bool(row.get("missing_nodes")),
            row.get("neededBy", ""),
        )

    def _build(self, installed: list[dict[str, Any]], available: list[dict[str, Any]]) -> None:
        for box in (self.installed, self.available):
            for child in box.inner.winfo_children():
                child.destroy()
        self._rows = {}
        for title, box, rows in (
            (self.installed_title, self.installed, installed),
            (self.available_title, self.available, available),
        ):
            title.pack_forget()
            box.pack_forget()
            if not rows:
                continue
            title.pack(anchor="w", pady=(12, 0), before=self.key_box)
            box.pack(fill="x", pady=(6, 0), before=self.key_box)
            for i, row in enumerate(rows):
                self._row(box.inner, row, first=i == 0)
            box.lift_corners()

    def _row(self, parent: tk.Misc, row: dict[str, Any], first: bool, before: tk.Misc | None = None) -> None:
        """One model: its name and what it does, and its button(s) on the right. Sizes are
        worked out from the buttons' known widths, so building a row lays nothing out."""
        from .modelinfo import model_detail

        f = ui.fonts(self)
        line = ui.frame(parent)
        line.pack(fill="x", **({"before": before} if before is not None else {}))
        if not first:
            ui.divider(line).pack(fill="x")
        body = ui.frame(line)
        body.pack(fill="x", padx=12, pady=10)
        top = ui.frame(body)
        top.pack(fill="x")
        right = ui.frame(top)
        right.pack(side="right")
        parts: dict[str, Any] = {"frame": line, "key": self._row_key(row)}
        mode = self._mode(row)
        status = row.get("statusText", "")
        detail = model_detail(row, row.get("neededBy", ""))
        width = 0

        def add(widget, w: int, padx=(0, 0)) -> None:
            nonlocal width
            widget.pack(side="left", padx=padx)
            width += w + sum(padx)

        if row["installed"]:
            if row.get("missing_models"):
                detail += " · Needs one more download"
                b = ui.Button(right, "Download", lambda i=row["id"]: self._get(i))
                add(b, b.w)
            elif row.get("missing_nodes"):
                detail += " · Needs extra parts"
                b = ui.Button(right, "Install", lambda i=row["id"]: self._nodes(i))
                add(b, b.w)
            elif mode == "working":
                t = ui.small(right, status, wrap=0)
                add(t, f.small.measure(status))
            elif row["runnable"] and not row.get("remote"):
                add(ui.IconButton(right, "trash", lambda r=row: self._uninstall(r)), 28)
        elif mode == "busy":
            parts["percent"] = ui.small(right, "", wrap=0)
            add(parts["percent"], f.small.measure("Waiting"), (0, 4))
            add(ui.IconButton(right, "close", lambda i=row["id"]: self._stop(i)), 28)
            if status.startswith("Downloading"):
                parts["bar"] = ui.Progress(body, CW - 2 - 24, BG)
                parts["bar"].pack(fill="x", pady=(8, 0))
        elif mode == "working":
            t = ui.small(right, status, wrap=0)
            add(t, f.small.measure(status))
        else:
            b = ui.Button(right, "Resume" if row["partial"] else "Download", lambda i=row["id"]: self._get(i))
            add(b, b.w)
        wrap = CW - 2 - 24 - 12 - width
        names = ui.frame(top)
        names.pack(side="left", fill="x")
        ui.text(names, row["name"], f.body, wrap=wrap).pack(anchor="w")
        ui.small(names, detail, wrap=wrap).pack(anchor="w")
        self._rows[row["id"]] = parts

    def _run(self, fn, *args, done: str = "") -> None:
        bar = self.bar

        def work() -> None:
            try:
                fn(*args)
                if done:
                    bar.call_soon(self.say, done, OK)
            except Exception as err:  # noqa: BLE001 - shown in the view
                bar.call_soon(self.say, str(err), BAD)
            bar.refresher.wake.set()

        threading.Thread(target=work, daemon=True).start()

    def _get(self, model_id: str) -> None:
        self.say("")
        self._run(self.bar.service.app.queue_download, model_id)

    def _nodes(self, model_id: str) -> None:
        self.say("")
        self._run(self.bar.service.app.queue_node_install, model_id)

    def _stop(self, model_id: str) -> None:
        self._run(self.bar.service.stop_download, model_id)

    def _uninstall(self, row: dict[str, Any]) -> None:
        from .bar import short_name

        if not ui.ask(
            self.bar.root,
            f"Uninstall {short_name(row['name'])}?\n\n"
            f"Its files ({row['size']}) are deleted. Files other models use are kept. You can download it again later.",
            "Uninstall",
        ):
            return
        self.say("")
        self._run(self.bar.service.uninstall, row["id"])

    def _save_token(self) -> None:
        value = self.token.get().strip()
        self.token.set("")
        self._run(self.bar.service.app.save_token, value, done="Hugging Face key saved.")


class EngineView(tk.Frame):
    """ComfyUI's folder, starting and stopping it, and its extra options."""

    def __init__(self, bar: Bar, parent: tk.Misc) -> None:
        super().__init__(parent, bg=BG, bd=0, highlightthickness=0)
        self.bar = bar

        def folder_row(box):
            row = ui.frame(box)
            browse = ui.Button(row, "Browse", self._browse)
            self.folder = ui.Field(row, CW - 8 - browse.w)
            self.folder.pack(side="left")
            browse.pack(side="left", padx=(8, 0))
            return row

        box, _ = field(self, "ComfyUI folder", folder_row)
        box.pack(fill="x", pady=(12, 0))
        self.autostart = ui.Checkbox(self, "Start the AI engine with PhotoCraft")
        self.autostart.pack(anchor="w", pady=(12, 0))

        def idle_row(box):
            row = ui.frame(box)
            self.idle = ui.Field(row, 72)
            self.idle.pack(side="left")
            ui.small(row, "minutes, 0 = never", wrap=0).pack(side="left", padx=(8, 0))
            return row

        box, _ = field(self, "Stop after idle", idle_row)
        box.pack(fill="x", pady=(12, 0))
        box, self.extra = field(self, "Extra ComfyUI options", lambda b: ui.Field(b, CW, "for example --lowvram"))
        box.pack(fill="x", pady=(12, 0))
        ui.divider(self).pack(fill="x", pady=(12, 0))
        tools = ui.frame(self)
        tools.pack(fill="x", pady=(12, 0))
        ui.caption(tools, "Engine").pack(anchor="w")
        row = ui.frame(tools)
        row.pack(anchor="w", pady=(8, 0))
        for i, (label, command) in enumerate((("Check", self.check), ("Stop engine", self.stop), ("Logs", self.logs))):
            ui.Button(row, label, command).pack(side="left", padx=(8 if i else 0, 0))
        self.result = ui.small(tools, "")
        if bar.service.app.settings.remote_fibo:
            for child in self.winfo_children():
                child.pack_forget()
            from openfill.sources import list_sources
            self.sources = {item["label"]: item["id"] for item in list_sources()}
            ui.caption(self, "Run edits on").pack(anchor="w", pady=(12, 0))
            self.source = ui.Select(self, CW, command=self.choose_source)
            self.source.pack(anchor="w", pady=(8, 0))
            self.source_url = ui.small(self, "")
            self.source_url.pack(anchor="w", pady=(8, 0))
            self.sync_source()
            ui.small(self, "Start ComfyUI on the chosen computer. OpenFill connects to that source and lists its available Qwen, FIBO, and SDXL edit models.").pack(fill="x", pady=(8, 0))
            row = ui.frame(self)
            row.pack(anchor="w", pady=(12, 0))
            ui.Button(row, "Check", self.check).pack(side="left")
            ui.Button(row, "Logs", self.logs).pack(side="left", padx=(8, 0))
            self.result = ui.small(self, "")

    def sync_source(self) -> None:
        if not hasattr(self, "source"):
            return
        app = self.bar.service.app
        self.source.set_values([] if self.bar.service.busy() else list(self.sources))
        self.source.set(app.source_label())
        self.source_url.configure(text=app.settings.url)

    def choose_source(self, label: str) -> None:
        self.say(f"Connecting to {label}...", DIM)
        source_id = self.sources[label]

        def work() -> None:
            try:
                result = self.bar.service.select_source(source_id)
                text = (f"Using {result['name']}. " if result["ok"] else f"{result['name']} is selected but unavailable. ") + result["detail"]
                self.bar.call_soon(self.say, text, OK if result["ok"] else BAD)
            except Exception as err:
                self.bar.call_soon(self.say, str(err), BAD)
            finally:
                self.bar.refresher.wake.set()
                self.bar.call_soon(self.sync_source)
        threading.Thread(target=work, daemon=True).start()

    def say(self, text: str, color: str = OK) -> None:
        if not text:
            self.result.pack_forget()
            return
        self.result.configure(text=text, fg=color)
        self.result.pack(fill="x", pady=(8, 0))

    def _browse(self) -> None:
        from tkinter import filedialog

        folder = filedialog.askdirectory(parent=self.bar.root, title="Choose your ComfyUI folder", mustexist=True)
        if folder:
            self.folder.set(folder)

    def check(self) -> None:
        bar = self.bar
        self.say("Checking the selected ComfyUI source..." if bar.service.app.settings.remote_fibo else
                 "Checking the AI engine (starting it takes about a minute)...", DIM)

        def work() -> None:
            try:
                health = bar.service.check_backend()
                device = getattr(health, "device", "")
                text = ("Working: " if health.ok else "Not working: ") + str(health.detail)
                text += f" on {device}." if device else "."
                bar.call_soon(self.say, text, OK if health.ok else BAD)
                bar.refresher.wake.set()
            except Exception as err:  # noqa: BLE001
                bar.call_soon(self.say, str(err), BAD)

        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self.say(self.bar.service.app.stop_backend(), DIM)
        self.bar.service.engine = ""

    def logs(self) -> None:
        self.bar.service.app.open_log_folder()


class GenerationView(tk.Frame):
    """Generation settings and selection-area-based auto-feather."""

    def __init__(self, bar: Bar, parent: tk.Misc) -> None:
        super().__init__(parent, bg=BG, bd=0, highlightthickness=0)
        self.bar = bar
        half = (CW - 12) // 2
        self.fields: dict[str, ui.Field] = {}
        keys = (("steps", "cfg"), ("grow_percent", "feather_percent"))
        labels = {key: label for key, label, _kind in SETTINGS}
        for pair in keys:
            row = ui.frame(self)
            row.pack(fill="x", pady=(12, 0))
            for i, key in enumerate(pair):
                box, entry = field(row, labels[key], lambda b: ui.Field(b, half))
                box.pack(side="left", padx=(12 if i else 0, 0))
                self.fields[key] = entry
        ui.small(
            self,
            "Steps: more is slower and finer. Strictness: how closely the prompt is followed. "
            "Margin: how far past the selection the fill may repaint, as a share of its size.",
        ).pack(fill="x", pady=(12, 0))
        self.result = ui.small(self, "")
        if bar.service.app.settings.remote_fibo:
            for child in self.winfo_children():
                child.pack_forget()
            ui.small(self, "FIBO: describe the change; Turbo 4 steps / CFG 1, Dev 30 steps / CFG 5. SDXL: describe the desired selected content (for example, a blue ceramic teapot); 25 steps / CFG 6, DPM++ 2M/Karras, 75% change strength. Pixels outside the selection are preserved.").pack(fill="x", pady=(12, 0))
            ui.small(self, "Qwen Image 2.1: describe the change and what to keep. Lightning v2 RL uses its own four-step schedule.").pack(fill="x", pady=(8, 0))
            ui.small(self, "The PhotoCraft selection controls the visible edit. Auto-feather softens its edges inward without growing the selection.").pack(fill="x", pady=(12, 0))
            self.result = ui.small(self, "")
        self.auto_feather = ui.Checkbox(self, "Auto-feather selection")
        self.auto_feather.pack(fill="x", pady=(12, 0))
        area_row = ui.frame(self)
        area_row.pack(fill="x", pady=(4, 0))
        ui.caption(area_row, "Feather area").pack(side="left")
        self.auto_feather_amount = ui.small(area_row, "10%", wrap=0)
        self.auto_feather_amount.pack(side="right")
        self.auto_feather_percent = tk.DoubleVar(self, 10.0)
        self.auto_feather_slider = tk.Scale(
            self, from_=0, to=100, resolution=1, orient="horizontal", showvalue=False,
            variable=self.auto_feather_percent, length=CW, width=12,
            bg=BG, fg=ui.TEXT, troughcolor=ui.PANEL, activebackground=ui.ACCENT,
            highlightthickness=0, bd=0, sliderrelief="flat", relief="flat",
        )
        self.auto_feather_slider.pack(fill="x", pady=(2, 0))
        self.auto_feather_help = ui.small(
            self, "Soften edges over roughly the outer 10% of the selected area. Larger selections get a wider feather; the selection does not grow.",
        )
        self.auto_feather_help.pack(fill="x", pady=(4, 0))
        self.auto_feather.var.trace_add("write", self._sync_auto_feather)
        self.auto_feather_percent.trace_add("write", self._sync_auto_feather)
        self.load_auto_feather(bar.service.app.settings)
        self._build_fibo(bar.service.app.settings)

    def _build_fibo(self, settings) -> None:
        self.fibo_box = box = ui.frame(self)
        if settings.remote_fibo:
            box.pack(fill="x", pady=(16, 0))
        ui.caption(box, "FIBO editing").pack(anchor="w")
        self.fibo_edit_mode = ui.Select(box, CW, command=lambda _value: self._sync_fibo())
        self.fibo_edit_mode.set_values(list(FIBO_EDIT_MODES.values()))
        self.fibo_edit_mode.pack(fill="x", pady=(8, 0))
        self.fibo_mode_help = ui.small(box, "")
        self.fibo_mode_help.pack(fill="x", pady=(4, 0))
        ui.caption(box, "Image context").pack(anchor="w", pady=(12, 0))
        self.fibo_context_mode = ui.Select(box, CW, command=lambda _value: self._sync_fibo())
        self.fibo_context_mode.set_values(list(FIBO_CONTEXT_MODES.values()))
        self.fibo_context_mode.pack(fill="x", pady=(8, 0))
        self.fibo_context_percent = tk.DoubleVar(self, 100.0)
        self.fibo_context_amount, self.fibo_context_slider = self._fibo_slider(
            box, "Context margin", self.fibo_context_percent, 25, 200,
        )
        self.fibo_context_help = ui.small(box, "")
        self.fibo_context_help.pack(fill="x", pady=(4, 0))
        self.fibo_boundary_match = ui.Checkbox(box, "Match boundary colors")
        self.fibo_boundary_match.pack(fill="x", pady=(12, 0))
        self.fibo_boundary_strength = tk.DoubleVar(self, 50.0)
        self.fibo_boundary_amount, self.fibo_boundary_slider = self._fibo_slider(
            box, "Matching strength", self.fibo_boundary_strength, 0, 100,
        )
        ui.small(box, "Reduce color seams at selection edges. This does not reconnect mismatched shapes.").pack(fill="x", pady=(4, 0))
        self.fibo_context_percent.trace_add("write", self._sync_fibo)
        self.fibo_boundary_strength.trace_add("write", self._sync_fibo)
        self.fibo_boundary_match.var.trace_add("write", self._sync_fibo)
        self.load_fibo(settings)

    @staticmethod
    def _fibo_slider(parent, label, variable, low, high):
        row = ui.frame(parent)
        row.pack(fill="x", pady=(8, 0))
        ui.caption(row, label).pack(side="left")
        amount = ui.small(row, "", wrap=0)
        amount.pack(side="right")
        slider = tk.Scale(
            parent, from_=low, to=high, resolution=1, orient="horizontal", showvalue=False,
            variable=variable, length=CW, width=12, bg=BG, fg=ui.TEXT,
            troughcolor=ui.PANEL, activebackground=ui.ACCENT, highlightthickness=0,
            bd=0, sliderrelief="flat", relief="flat",
        )
        slider.pack(fill="x", pady=(2, 0))
        return amount, slider

    def _sync_fibo(self, *_args) -> None:
        preserve = self.fibo_edit_mode.get() == FIBO_EDIT_MODES["preserve"]
        self.fibo_mode_help.configure(text=(
            "Starts from the original pixels and limits regeneration to the selection, retaining more of the existing structure. Some details may still change."
            if preserve else "Rebuild the selected content using its surroundings. Original details inside may be replaced."
        ))
        full = self.fibo_context_mode.get() == FIBO_CONTEXT_MODES["full"]
        amount = self.fibo_context_percent.get()
        self.fibo_context_amount.configure(text=f"{amount:g}%")
        self.fibo_context_slider.configure(state="disabled" if full else "normal")
        self.fibo_context_help.configure(text=(
            "Use the full image as context. Small selections may lose fine detail at the model's working resolution."
            if full else f"Include a margin on each side equal to {amount:g}% of the selection's larger dimension. The editable area stays the same."
        ))
        self.fibo_boundary_amount.configure(text=f"{self.fibo_boundary_strength.get():g}%")
        self.fibo_boundary_slider.configure(state="normal" if self.fibo_boundary_match.get() else "disabled")

    def load_fibo(self, settings) -> None:
        self.fibo_edit_mode.set(FIBO_EDIT_MODES.get(settings.fibo_edit_mode, FIBO_EDIT_MODES["preserve"]))
        self.fibo_context_mode.set(FIBO_CONTEXT_MODES.get(settings.fibo_context_mode, FIBO_CONTEXT_MODES["local"]))
        self.fibo_context_percent.set(settings.fibo_context_percent)
        self.fibo_boundary_strength.set(settings.fibo_boundary_strength)
        self.fibo_boundary_match.set(settings.fibo_boundary_match)
        self._sync_fibo()

    def fibo_values(self) -> dict[str, Any]:
        return {
            "fibo_edit_mode": {label: key for key, label in FIBO_EDIT_MODES.items()}.get(self.fibo_edit_mode.get(), ""),
            "fibo_context_mode": {label: key for key, label in FIBO_CONTEXT_MODES.items()}.get(self.fibo_context_mode.get(), ""),
            "fibo_context_percent": self.fibo_context_percent.get(),
            "fibo_boundary_match": self.fibo_boundary_match.get(),
            "fibo_boundary_strength": self.fibo_boundary_strength.get(),
        }

    def _sync_auto_feather(self, *_args) -> None:
        amount = max(0.0, min(100.0, self.auto_feather_percent.get()))
        self.auto_feather_amount.configure(text=f"{amount:g}%")
        self.auto_feather_slider.configure(state="normal" if self.auto_feather.get() else "disabled")
        self.auto_feather_help.configure(
            text=f"Soften edges over roughly the outer {amount:g}% of the selected area. Larger selections get a wider feather; the selection does not grow.",
        )

    def load_auto_feather(self, settings) -> None:
        self.auto_feather_percent.set(float(getattr(settings, "auto_feather_percent", 10.0)))
        self.auto_feather.set(bool(getattr(settings, "auto_feather", True)))
        self._sync_auto_feather()

    def auto_feather_values(self) -> dict[str, Any]:
        return {"auto_feather": self.auto_feather.get(), "auto_feather_percent": self.auto_feather_percent.get()}

    def say(self, text: str, color: str = OK) -> None:
        if not text:
            self.result.pack_forget()
            return
        self.result.configure(text=text, fg=color)
        self.result.pack(fill="x", pady=(8, 0))
