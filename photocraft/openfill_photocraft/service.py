# OpenFill for PhotoCraft
# SPDX-License-Identifier: Apache-2.0
"""The companion's engine: one OpenFill job at a time against the active PhotoCraft document.

``JobState`` is what the web page polls; ``WebRunner`` is ``App.runner_cls`` (the GTK
progress window's stand-in); ``Service`` runs fill, expand and text to image in a
worker thread and keeps the variations of the last job for the picker.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from openfill.app import App, NothingSelected
from openfill.backend import GenerationCancelled, describe_error
from openfill.config import validate_generation_settings
from openfill.progress import parse_preview_frame
from openfill.remote_models import H3_IMAGE_ID
from openfill.supervisor import process_name

from .bridge import PCImage, PCLayer, PhotoCraftBridge
from .control import ControlClient, ControlError
from .engine_routes import EngineRoutes, mode_for
from .references import ReferenceImages
from .removal_advice import H3_TIP, TEXTURE_REMOVAL, recommend
from .selection import Selection, key_for
from .selection_advice import SelectionSampler, live_selection_key, selection_key

NO_SELECTION = "Select an area first: Marquee, Lasso or Object Selection. Select > All works on the whole image."
TEXTURE_LABEL = "Local background repair"
REMOVE_SELECTION = "Select the object and its shadow, leaving some background outside the selection."


@dataclass
class JobState:
    """Everything the page shows about the current or last job."""

    state: str = "idle"  # idle | running | done | error | cancelled
    kind: str = ""
    title: str = ""
    model: str = ""
    fraction: float = 0.0
    text: str = ""
    message: str = ""
    started: float = 0.0
    finished: float = 0.0
    preview: bytes = b""
    preview_mime: str = ""
    preview_seq: int = 0
    preview_rect: tuple[int, int, int, int] | None = None  # the region a preview frame shows (job's first canvas)
    cancel_requested: bool = False
    local_removal: bool = False
    committing: bool = False  # the final PhotoCraft history operation cannot be cancelled halfway
    variations: list[str] = field(default_factory=list)  # names, best first
    shown: int = 0
    removal_summary: str = ""
    variation_set: int = 0
    notices: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "kind": self.kind,
                "title": self.title,
                "model": self.model,
                "fraction": round(self.fraction, 4),
                "text": self.text,
                "message": self.message,
                "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else 0,
                "previewSeq": self.preview_seq,
                "previewRect": self.preview_rect,
                "variations": list(self.variations),
                "shown": self.shown,
                "variationSet": self.variation_set,
                "removalSummary": self.removal_summary,
                "notices": list(self.notices[-5:]),
                "committing": self.committing,
            }


class WebRunner:
    """``App.runner_cls``: the job runs in the service's worker thread; progress, preview
    frames and Cancel go through ``JobState`` instead of a GTK window."""

    def __init__(self, app: App, title: str, model_name: str, interactive: bool = True) -> None:
        self.app = app
        self.state: JobState = app.job_state  # set by Service
        with self.state.lock:
            self.state.title, self.state.model = title, model_name
            if self.state.state != "running":
                self.state.state, self.state.started, self.state.fraction = "running", time.time(), 0.0

    def _progress(self, fraction: float, text: str) -> None:
        with self.state.lock:
            self.state.fraction, self.state.text = max(0.0, min(1.0, float(fraction))), str(text)

    def _preview(self, payload: bytes) -> None:
        frame = parse_preview_frame(payload)
        if frame is None:
            return
        with self.state.lock:
            self.state.preview, self.state.preview_mime = frame.data, frame.mime
            self.state.preview_seq += 1

    def _cancelled(self) -> bool:
        return self.state.cancel_requested

    def next_round(self) -> None:
        pass

    def run(self, job: Any) -> object:
        result = job(self._progress, self._preview, self._cancelled)
        if self.state.cancel_requested:
            raise GenerationCancelled("cancelled by the user after the result arrived")
        return result

    def done(self, message: str = "Done", close_after_ms: int = 1000) -> None:
        with self.state.lock:
            self.state.text = message

    def fail(self, error: BaseException) -> None:
        pass  # Service records the outcome


class Service:
    """Owns the PhotoCraft connection, OpenFill's App and the one running job."""

    def __init__(self, client: ControlClient, bridge: PhotoCraftBridge) -> None:
        self.client = client
        self.bridge = bridge
        self.state = JobState()
        bridge.notify = self._notice
        bridge.on_region = self._region
        self.app = App(interactive=True, io=bridge, runner_cls=WebRunner)
        self.app.job_state = self.state
        self.routes = EngineRoutes(self.app, bridge.profile)
        if not self.routes.app_for("edit").settings.remote_fibo:
            self._find_comfy()
        # ComfyUI stops when this companion ends (it is what the backend serves).
        import os

        self.app.server.watch_pid = os.getpid()
        self.app.server.watch_name = process_name(os.getpid()) or "python"
        self._thread: threading.Thread | None = None
        self._operation_lock = threading.Lock()
        self._discovery_lock = threading.Lock()
        self.references = ReferenceImages()
        self.advice_sampler = SelectionSampler(client)
        self.selection = Selection(bridge)
        self._job_app = self.app
        self._switching = False
        self.watcher = None  # menu.Watcher, set by the launcher: keeps the document's facts current
        self.engine = ""  # "" not started by us, "starting", "ready", or why it failed
        self._warm: threading.Thread | None = None
        self.image: PCImage | None = None
        self.layers: list[PCLayer] = []
        self._choice_kind = ""
        self.setup_task = ""  # the ComfyUI download this companion queued
        # The document's history before the last job, and as the job and the picker left it:
        # while it is still that, Keep and Discard can take the result back and leave one
        # clean step (Keep) or none (Discard) instead of piling deletes on top.
        self._job_base: list[str] = []
        self._expected_history: list[str] = []
        self._expected_revision: int | None = None
        self._collapsed = False  # the job's several steps were made into one (an expand)

    def _find_comfy(self) -> None:
        """First start: use the ComfyUI OpenFill for GIMP uses, or one already on this computer."""
        from openfill.comfy_setup import detect_comfy
        from openfill.config import comfy_layout

        if self.app.settings.remote_fibo:
            return
        if comfy_layout(self.app.settings.comfy_dir) is not None:
            return
        for folder in gimp_comfy_dirs():
            if comfy_layout(folder) is not None:
                self.app.use_comfy_dir(folder)
                self.app.log.info("using the ComfyUI OpenFill for GIMP uses: %s", folder)
                self.rescan_models()
                return
        found = detect_comfy()
        if found:
            self.app.use_comfy_dir(str(found[0].path))
            self.app.log.info("using the ComfyUI found at %s (%s)", found[0].path, found[0].source)
            self.rescan_models()

    def rescan_models(self) -> None:
        """The catalog plus the user's own model files in the (new) ComfyUI folder."""
        from openfill import catalog as cat
        from openfill import local_models

        app = self.app
        if app.settings.remote_fibo:
            app.refresh_remote_models(force=True)
            return
        try:
            entries = cat.load_merged(plugin_dir=app.plugin_dir, base=app.paths.base)
            found, _skipped = local_models.scan(app.models_root, entries, app.paths.base)
            app.catalog = cat.merge_entries(entries, found)
        except Exception as err:  # noqa: BLE001 - the catalog stays as it was
            app.log.warning("models not rescanned: %s", err)

    def _region(self, rect: tuple[int, int, int, int]) -> None:
        with self.state.lock:
            self.state.preview_rect = rect

    def _notice(self, text: str) -> None:
        with self.state.lock:
            self.state.notices.append(text)
        if hasattr(self, "app"):
            self.app.log.info("notice: %s", text)

    def warm_up(self) -> None:
        """Start ComfyUI in the background so the first Generate does not wait for it (only
        when it is set up and a model is there to use)."""
        from openfill.config import comfy_layout

        app = self.app_for("edit")
        if app.settings.remote_fibo or comfy_layout(app.settings.comfy_dir) is None or not app.settings.autostart:
            return
        if not (app.has_model_for("fill") or app.has_model_for("remove")):
            return

        def run() -> None:
            self.engine = "starting"
            try:
                health = app.check_backend()
                self.engine = "ready" if health.ok else (health.detail or "not running")
            except Exception as err:  # noqa: BLE001 - shown on the bar; a job tries again
                self.engine = describe_error(err)
            app.log.info("warm-up: %s", self.engine)

        self._warm = threading.Thread(target=run, name="openfill-warm-up", daemon=True)
        self._warm.start()

    # status -------------------------------------------------------------------------

    def photocraft(self) -> dict[str, Any]:
        """The active document and its selection, or why there is none."""
        if self.watcher is not None:
            return self.watcher.snapshot()
        try:
            ui = self.client.call("ui.inspect")
            session = ui.get("session") or {}
            d = ui.get("document") or {}
            if not d:
                return {"connected": True, "document": None}
            sel = d.get("selectionBounds") if d.get("hasSelection") else None
            return {
                "connected": True,
                "document": {
                    "name": d.get("name"),
                    "width": d.get("width"),
                    "height": d.get("height"),
                    "mode": d.get("mode"),
                },
                "selection": sel,
                "documentKey": key_for(session.get("active"), d),
                "selectionKey": selection_key(
                    session.get("active"),
                    d.get("name"),
                    d.get("revision"),
                    d.get("width"),
                    d.get("height"),
                    sel,
                    d.get("activeLayer"),
                ),
            }
        except ControlError as err:
            return {"connected": False, "error": str(err)}

    def removal_advice(self, available):
        """Read pixels locally and resolve the result against this panel's visible remover inventory."""
        if self.busy() or self.layers:
            return {
                "status": "paused",
                "tip": "Choose a Method to compare another repair, or keep or discard the current result.",
            }
        snapshot = self.photocraft()
        app = self.app_for("remove")
        if app.settings.remote_fibo and H3_IMAGE_ID in available and H3_IMAGE_ID not in app.settings.hidden_models:
            key = snapshot.get("selectionKey") or ""
            return {
                "status": "ready" if key else "empty",
                "key": key,
                "method": H3_IMAGE_ID,
                "displayMethod": H3_IMAGE_ID,
                "source": app.settings.fibo_source,
                "tip": H3_TIP if key else "Select the object and any shadow you want removed.",
            }
        result = self.advice_sampler.request(snapshot)
        if result["status"] == "ready":
            existing = app.settings.remote_fibo
            result.update(recommend(result.pop("features"), available, existing))
            result["source"] = app.settings.fibo_source if existing else ""
        return result

    def selection_preview(self, key, *, show_mask=False):
        with self._operation_lock:
            if self.busy() or self.layers:
                raise RuntimeError("Finish the current job and keep or discard its choices before selecting.")
            return self.selection.capture(key, show_mask=show_mask)

    # jobs -------------------------------------------------------------------------------

    def busy(self) -> bool:
        return getattr(self, "_switching", False) or (self._thread is not None and self._thread.is_alive())

    def app_for(self, mode):
        return self.routes.app_for(mode) if hasattr(self, "routes") else self.app

    def configure_engines(self, enabled, routes):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("OpenFill is reading the canvas or checking an engine. Try again in a moment.")
        try:
            if self.busy():
                raise RuntimeError("Wait for the current job to finish before changing its engines.")
            return self.routes.save(enabled, routes)
        finally:
            self._operation_lock.release()

    def add_server(self, name, address):
        with self._operation_lock:
            source = self.routes.add_server(name, address)
        return {"id": source.id, "name": source.label, "address": source.address}

    def find_servers(self):
        from .server_discovery import discover

        if not self._discovery_lock.acquire(blocking=False):
            raise RuntimeError("A server search is already running.")
        try:
            return discover(tuple(self.routes.sources.values()))
        finally:
            self._discovery_lock.release()

    def select_source(self, source_id: str) -> dict[str, Any]:
        """Change destinations only between edits, including the connection check."""
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("A source change or connection check is already running.")
        try:
            if self.busy():
                raise RuntimeError("Wait for the current edit to finish before changing its source.")
            self._switching = True
            if self.app.settings.remote_fibo:
                source = self.app.select_source(source_id)
                target, name = self.app, source.label
            else:
                target = self.routes.select_shared(source_id)
                name = self.routes.describe()["shared"]
            self.engine = ""
            health = target.check_backend()
            return {
                "source": source_id,
                "name": name,
                "ok": health.ok,
                "detail": health.detail,
                "device": getattr(health, "device", ""),
            }
        finally:
            self._switching = False
            self._operation_lock.release()

    def check_backend(self, mode="edit"):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("A source change or connection check is already running.")
        try:
            if self.busy():
                raise RuntimeError("Wait for the current edit to finish before checking its source.")
            self._switching = True
            app = self.app_for(mode)
            if mode_for(mode) == "select":
                from openfill.backend import HealthStatus

                from .wevisone import encoder

                try:
                    client = self.selection.model._client(app, lambda *_: None, lambda: False)
                    return HealthStatus(True, f"WeVisOne-VL is available: {encoder(client)}")
                except Exception as err:
                    return HealthStatus(False, str(err))
            return app.check_backend()
        finally:
            self._switching = False
            self._operation_lock.release()

    def start(self, kind: str, params: dict[str, Any]) -> None:
        if not self._operation_lock.acquire(blocking=False):
            if self._switching:
                raise RuntimeError("Wait for the source connection check to finish.")
            raise RuntimeError("OpenFill is reading the canvas. Try again in a moment.")
        try:
            self._start(kind, params)
        finally:
            self._operation_lock.release()

    def reference_rows(self):
        return self.references.rows()

    def add_reference_files(self, paths):
        return self.references.add_files(paths)

    def _start(self, kind: str, params: dict[str, Any]) -> None:
        params = dict(params)
        removing = kind == "fill" and bool(params.get("remove"))
        app = self.app_for(mode_for(kind, removing))
        expected = params.pop("engineSource", None)
        if expected is not None and expected != (app.settings.fibo_source if app.settings.remote_fibo else ""):
            raise ValueError("The engine changed. Review the current mode and try again.")
        if app.settings.remote_fibo and kind != "select":
            if kind != "fill":
                raise RuntimeError("Choose Select, Edit, or Remove for this ComfyUI source.")
            if removing and params.get("removal", "default") in ("", "default", None):
                params["removal"] = TEXTURE_REMOVAL
                if H3_IMAGE_ID not in app.settings.hidden_models and any(
                    entry.id == H3_IMAGE_ID for entry in app.installed_models("remove")
                ):
                    params["removal"] = H3_IMAGE_ID
            if not removing and not str(params.get("prompt", "")).strip():
                raise RuntimeError("Describe the desired edit or selected content.")
        if self.busy():
            raise RuntimeError("A job is already running.")
        if kind == "select":
            if self.layers:
                raise RuntimeError("Keep or discard the current choices before changing the selection.")
            self.selection.validate(params)
            params = {
                k: params[k] for k in ("action", "key", "pixels", "query", "point", "box", "operation") if k in params
            }
        if removing and "adviceKey" in params:
            key = params["adviceKey"]
            if not key or live_selection_key(self.client) != key:
                raise RuntimeError("The selection changed. Wait for the updated suggestion and press Remove again.")
            visible = []
            if params.get("removal") != TEXTURE_REMOVAL:
                visible = [e.id for e in app.installed_models("remove") if e.id not in app.settings.hidden_models]
            fresh = self.removal_advice(visible)
            if (fresh.get("key"), fresh.get("method"), fresh.get("source")) != (
                key,
                params.get("removal"),
                params.get("adviceSource"),
            ):
                raise RuntimeError("The removal suggestion changed. Review the current Method and press Remove again.")
        use_references = params.get("useReferences", False)
        if not isinstance(use_references, bool):
            raise ValueError("Reference use must be on or off.")
        if use_references and (kind != "fill" or params.get("remove")):
            raise ValueError("Reference images are available for prompted edits.")
        # Freeze the image bytes and their order before the background worker starts.
        params["_reference_images"] = self.references.snapshot() if use_references else ()
        local_removal = removing and params.get("removal") == TEXTURE_REMOVAL
        if removing and (local_removal or app.settings.remote_fibo):
            params.pop("generation_settings", None)
        if removing and not 1 <= int(params.get("variations", 1)) <= 5:
            raise ValueError("Choose between one and five removal variations.")
        if "generation_settings" in params:
            values = validate_generation_settings(params.pop("generation_settings"))
            settings = self.app.settings
            previous = {key: getattr(settings, key) for key in values if getattr(settings, key) != values[key]}
            if previous:
                for key in previous:
                    setattr(settings, key, values[key])
                try:
                    settings.save(self.app.paths.config)
                except OSError:
                    for key, value in previous.items():
                        setattr(settings, key, value)
                    raise
        with self.state.lock:
            self.state.state, self.state.kind, self.state.message, self.state.text = "running", kind, "", "Starting..."
            self.state.title = {
                "select": "Grow selection" if params.get("action") == "grow" else "Object selection",
                "fill": "Object Removal" if params.get("remove") else "Generative Fill",
                "expand": "Generative Expand",
                "txt2img": "Text to Image",
            }.get(kind, kind)
            self.state.model = "WeVisOne-VL" if kind == "select" and params.get("action") != "grow" else ""
            self.state.removal_summary = ""
            self.state.local_removal = local_removal
            self.state.fraction, self.state.started, self.state.finished = 0.0, time.time(), 0.0
            self.state.cancel_requested = False
            self.state.committing = False
            self.state.preview, self.state.preview_seq = b"", self.state.preview_seq + 1
            self.state.preview_rect = None
        self._job_app = app  # immutable destination for this job, including any cancellation
        self._thread = threading.Thread(target=self._job, args=(kind, params), name="openfill-job", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        with self.state.lock:
            committing = self.state.committing
            local_removal = self.state.local_removal
            selecting = self.state.kind == "select"
            if not committing:
                self.state.cancel_requested = True
        if committing:
            self._notice("The result is being added to PhotoCraft. Use Undo when it finishes to remove the edit.")
            return
        if local_removal or selecting:
            return  # The scratch fill checks cancellation before the result reaches the document.
        app = getattr(self, "_job_app", self.app)
        if app.settings.remote_fibo:
            backend = getattr(app, "_active_backend", None)
            if backend is not None:
                backend.cancel()
            return  # never interrupt another backend user or clear its shared queue
        try:
            app.server.request_cancel()
        except Exception:  # noqa: BLE001
            pass
        try:
            from openfill.comfy_api import ComfyClient

            ComfyClient(app.server.url(), timeout=3).interrupt()
        except Exception:  # noqa: BLE001
            pass

    def _job(self, kind: str, p: dict[str, Any]) -> None:
        image = None
        app = getattr(self, "_job_app", self.app)
        if (
            app is self.app
            and kind != "select"
            and not self.state.local_removal
            and self._warm is not None
            and self._warm.is_alive()
        ):
            with self.state.lock:
                self.state.text = "Starting the AI engine (the first time takes about a minute)..."
            self._warm.join()  # one start of ComfyUI, not two
        try:
            if kind == "select":

                def progress(fraction, message):
                    with self.state.lock:
                        self.state.fraction, self.state.text = fraction, message

                message = self.selection.run(app, p, progress, lambda: self.state.cancel_requested, self._begin_commit)
                with self.state.lock:
                    self.state.state, self.state.fraction = "done", 1.0
                    self.state.text = self.state.message = message
                return
            if p.get("adviceKey") and live_selection_key(self.client) != p["adviceKey"]:
                raise RuntimeError("The selection changed before removal started. Review the updated suggestion.")
            image = self.bridge.active_image() if kind != "txt2img" or p.get("target") == "layer" else None
            if kind != "txt2img" and image is None:
                raise RuntimeError("Open an image in PhotoCraft first.")
            previous = getattr(self, "layers", [])
            if previous and not self.bridge.has_image(self.image):
                self._clear_choices()
                previous = []
            if previous:
                if kind != "fill" or self._choice_kind != "fill" or image is None:
                    raise RuntimeError("Keep or discard the current choices before starting a different tool.")
                if (image.index, image.name) != (self.image.index, self.image.name):
                    raise RuntimeError("Keep or discard the current choices before generating on another document.")
                image.excluded_layers = tuple(layer.id for layer in previous if layer.id is not None)
            base: list[str] | None = None
            if image is not None:
                self.bridge.track(image)
                base = self.bridge.history(image)
            revision = self.bridge._revision(image) if previous or (kind == "fill" and p.get("remove")) else None
            if p.get("adviceKey") and live_selection_key(self.client) != p["adviceKey"]:
                raise RuntimeError("The selection changed before removal started. Review the updated suggestion.")
            if previous and (
                not self._expected_history or base != self._expected_history or revision != self._expected_revision
            ):
                raise RuntimeError("The document changed. Keep or discard the current choices before generating again.")
            seed = int(p.get("seed", -1))
            variations = int(p.get("variations", 1))
            model = str(p.get("model") or "active")
            if kind == "fill" and p.get("remove"):
                layers = self._remove(image, p, seed, variations)
                if 0 < len(layers) < variations:
                    label = "repair" if len(layers) == 1 else "repairs"
                    with self.state.lock:
                        self.state.removal_summary = (
                            f"{variations} requested; {len(layers)} distinct {label} for this selection."
                        )
            elif kind == "fill":
                references = p.get("_reference_images", ())
                layers = app.generative_fill(
                    image,
                    str(p.get("prompt", "")),
                    str(p.get("negative", "")),
                    variations,
                    None if app.settings.remote_fibo and p.get("strength") is None else float(p.get("strength", 1.0)),
                    seed,
                    model,
                    remove=bool(p.get("remove")),
                    **({"reference_images": references} if references else {}),
                )
            elif kind == "expand":
                layers = app.generative_expand(
                    image,
                    str(p.get("prompt", "")),
                    str(p.get("negative", "")),
                    int(p.get("left", 0)),
                    int(p.get("top", 0)),
                    int(p.get("right", 0)),
                    int(p.get("bottom", 0)),
                    variations,
                    seed,
                    model,
                )
            elif kind == "txt2img":
                app.text_to_image(
                    image,
                    str(p.get("prompt", "")),
                    str(p.get("negative", "")),
                    int(p.get("width", 0)),
                    int(p.get("height", 0)),
                    variations,
                    seed,
                    model,
                    str(p.get("target", "image")),
                )
                layers = []
            else:
                raise RuntimeError(f"unknown job {kind}")
            done_text = self.state.text or "Done"
            if image is not None:
                self._begin_commit()
                if revision is not None and (
                    self.bridge.history(image) != base or self.bridge._revision(image) != revision
                ):
                    raise RuntimeError("The document changed while generating. Review your selection before retrying.")
                if previous:
                    for number, layer in enumerate(layers or (), start=len(previous) + 1):
                        label = re.sub(r" \(\d+\)$", "", layer.name)
                        layer.name = f"{label} ({number})"
                        layer.visible = False
                with self.state.lock:
                    self.state.text = "Adding the result to PhotoCraft..."
                self.bridge.finish(image)  # the results reach the document here, in one step
                original_base = self._job_base if previous else []
                self._job_base, self._expected_history, self._collapsed = [], [], False
                self._expected_revision = None
                self._one_step(image, base)
                if previous and self._expected_history:
                    self._job_base = original_base
            self.image = image
            self.layers = previous + list(layers or [])
            self._choice_kind = kind if self.layers else ""
            with self.state.lock:
                self.state.state = "done"
                self.state.fraction = 1.0
                self.state.text = self.state.message = done_text
                self.state.variations = [layer.name for layer in self.layers]
                if not previous:
                    self.state.variation_set += 1
                    self.state.shown = 0
        except BaseException as err:  # noqa: BLE001 - reported on the page
            cancelled = isinstance(err, GenerationCancelled) or self.state.cancel_requested
            self.app.log.warning("job %s failed: %s\n%s", kind, err, traceback.format_exc())
            with self.state.lock:
                self.state.state = "cancelled" if cancelled else "error"
                if cancelled:
                    self.state.message = "Cancelled - the image was left unchanged."
                elif isinstance(err, NothingSelected):  # the core's message names GIMP's tools
                    self.state.message = REMOVE_SELECTION if p.get("remove") else NO_SELECTION
                elif kind == "select" and isinstance(err, ValueError):
                    self.state.message = str(err)
                else:
                    self.state.message = describe_error(err)
        finally:
            if image is not None:
                self.bridge.discard(image)  # whatever did not reach the document is dropped
            with self.state.lock:
                self.state.finished = time.time()
                self.state.committing = False

    def _begin_commit(self) -> None:
        """Atomically accept a late cancel or enter the indivisible history commit."""
        with self.state.lock:
            if self.state.cancel_requested:
                raise GenerationCancelled("cancelled before the result was added to PhotoCraft")
            self.state.committing = True

    # Remove values of older menus (saved in the panel's settings): LaMa or MAT alone, and the
    # two-pass "LaMa + a generation model", which LaMa at full resolution replaced.
    OLD_REMOVALS = {"lama": "lama-big", "mat": "mat-places512"}

    def _remove(self, image: PCImage, p: dict[str, Any], seed: int, variations: int) -> list:
        """Stage texture or model-based removal through the usual variation and commit path."""
        app = getattr(self, "_job_app", self.app)
        how = str(p.get("removal") or "default")
        if how == TEXTURE_REMOVAL:
            runner = WebRunner(app, "Object Removal", TEXTURE_LABEL)
            layers = runner.run(
                lambda progress, _preview, cancel: self.bridge.remove_with_texture(
                    image, variations, seed, progress, cancel
                )
            )
            runner.done("Background repaired.")
            return layers
        if how.startswith("refine:") or how == "default":
            how = "active"
        how = self.OLD_REMOVALS.get(how, how)
        layers = app.generative_fill(image, "", "", variations, 1.0, seed, how, remove=True, as_picked=how != "active")
        for index, layer in enumerate(layers):
            layer.name = "Object Removal" if len(layers) == 1 else f"Object Removal ({index + 1})"
        if image.pending_group:
            image.pending_group = "Object Removal"
        return layers

    # ComfyUI ----------------------------------------------------------------------------

    def comfy_preflight(self) -> dict[str, Any]:
        """What a ComfyUI download would be (Windows): its size, where it goes, and the checks
        (disk space, graphics card). ``blocked`` when it cannot run."""
        from openfill import comfy_setup

        install_dir = comfy_setup.default_install_dir()
        manifest, pre = self.app.setup_preflight(install_dir)
        chosen = manifest.chosen(pre.older_gpu, getattr(pre, "amd_build", False))
        size = comfy_setup.download_notice(chosen)
        return {
            "folder": str(install_dir),
            "size": size,
            "url": chosen.archive_url,
            "bytes": chosen.archive_size,
            "blocked": pre.blocked,
            "cpu": pre.needs_cpu_choice,
            "older_gpu": pre.older_gpu,
            "amd": bool(getattr(pre, "amd_build", False)),
            "gpu": getattr(pre, "gpu", ""),
            "text": pre.text(),
        }

    def setup_comfy(self) -> str:
        self.app._require_local_management()
        """Download and install ComfyUI's portable build (the version OpenFill for GIMP
        installs, checked against its SHA-256) as a background task. The supervisor saves the
        folder in the settings when it is done (see ``setup_progress``)."""
        from openfill import comfy_setup

        pre = self.comfy_preflight()
        if pre["blocked"]:
            raise RuntimeError(pre["text"] or "ComfyUI cannot be installed here.")
        self.setup_task = self.app.queue_comfy_setup(
            comfy_setup.default_install_dir(), bool(pre["cpu"]), bool(pre["older_gpu"]), bool(pre["amd"])
        )
        return self.setup_task

    def setup_progress(self) -> dict[str, Any] | None:
        """The ComfyUI download while it runs ({fraction, text}), its error when it failed, or
        None. When it has finished, the new folder is taken into use here."""
        from openfill import comfy_setup

        status = self.app.task_status()
        current = status.get("current") or {}
        if current.get("type") == "setup_comfy":
            state = str(current.get("state") or "")
            done, total = int(current.get("done") or 0), int(current.get("total") or 0)
            fraction = float(current.get("fraction") or comfy_setup.overall_fraction(state, done, total))
            return {"fraction": fraction, "text": setup_step_text(state, done, total)}
        task = getattr(self, "setup_task", "")
        if not task:
            return None
        last = next((h for h in status.get("history") or [] if h.get("task_id") == task), None)
        if last is None:
            return {"fraction": 0.0, "text": "Getting ready..."}
        self.setup_task = ""
        if last.get("state") != "done":
            return {"error": str(last.get("message") or last.get("state") or "the download stopped")}
        self.app.reload_settings()
        self.rescan_models()
        self.warm_up()
        return None

    # models -----------------------------------------------------------------------------

    def uninstall(self, model_id: str) -> str:
        """Delete one installed model's files (a task for the download worker). Files another
        installed model also uses stay; companion entries (the Fooocus patch) are their own
        entries and stay too. The model set active by hand falls back to the automatic choice."""
        app = self.app
        task = app.remove_model(model_id)
        if app.settings.active_model == model_id:
            app.settings.active_model = ""
            app.settings.save(app.paths.config)
        return task

    def stop_download(self, model_id: str) -> list[str]:
        """Stop downloading one model: drop its queued tasks and cancel it when it is the one
        running. Its parts (the Fooocus patch) stop too, unless another queued model needs
        them. What was downloaded stays on disk, so Download resumes it. Returns the ids
        stopped."""
        from openfill import catalog as cat

        server = self.app.server
        queued = [t for t in server.pending_tasks() if t.get("type") == "download"]
        try:
            needs = set(cat.by_id(self.app.catalog, model_id).requires_models)
        except cat.CatalogError:
            needs = set()
        others = {t.get("model_id") for t in queued} - {model_id} - needs
        for other in others:
            try:
                needs -= set(cat.by_id(self.app.catalog, other).requires_models)
            except cat.CatalogError:
                pass
        stop = {model_id} | needs
        folder = server.tasks_dir
        if folder.is_dir():
            for path in folder.glob("*.json"):
                if path.name.endswith(".running.json"):
                    continue
                try:
                    task = json.loads(path.read_text(encoding="utf-8"))
                    if task.get("type") == "download" and task.get("model_id") in stop:
                        path.unlink()
                except (OSError, ValueError):
                    pass
        current = server.task_status().get("current") or {}
        if current.get("type") == "download" and current.get("model_id") in stop:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "cancel").write_text(str(time.time()), encoding="utf-8")
        return sorted(stop)

    def show_variation(self, index: int) -> None:
        with self._operation_lock:
            if self.busy():
                raise RuntimeError("Wait for generation to finish before picking a choice.")
            if not self.image or not (0 <= index < len(self.layers)):
                raise RuntimeError("No variations to pick from.")
            unchanged = self._unchanged_since_job()
            self.bridge.show_variation(self.image, self.layers, index)
            if unchanged:
                self._remember_history()
            else:
                self._expected_history = []
                self._expected_revision = None
            with self.state.lock:
                self.state.shown = index

    def _one_step(self, image: PCImage, base: list[str] | None) -> None:
        """After a job: when it left several history steps (an expand crops the canvas, then adds
        its layers, once per round), make them one step named after the job, so one Undo takes
        it all back. Remembers the history for Keep and Discard."""
        try:
            after = self.bridge.history(image)
            if base is None or after[: len(base)] != base or len(after) == len(base):
                return
            if len(after) - len(base) > 1:
                self.bridge.collapse(image, len(base), self.state.title or "OpenFill")
                self._collapsed = True
                after = self.bridge.history(image)
            self._expected_revision = self.bridge._revision(image)
            self._job_base, self._expected_history = base, after
        except ControlError as err:
            self.app.log.warning("the job's steps were not made one: %s", err)

    def keep(self, which: str) -> None:
        """'selected': delete the hidden variations; 'all': keep them (hidden); 'none': delete all."""
        with self._operation_lock:
            if self.busy():
                raise RuntimeError("Wait for generation to finish before keeping or discarding choices.")
            self._keep(which)

    def _clear_choices(self) -> None:
        self.layers, self.image, self._choice_kind = [], None, ""
        self._job_base, self._expected_history, self._collapsed = [], [], False
        self._expected_revision = None
        with self.state.lock:
            self.state.variations = []
            self.state.shown = 0
            self.state.variation_set += 1

    def _keep(self, which: str) -> None:
        if which not in ("selected", "all", "none"):
            raise ValueError("Choose Keep, Keep all, or Discard.")
        if not self.image or not self.layers:
            return
        if which == "all" or not self.bridge.has_image(self.image):
            self._clear_choices()
            return
        shown = self.state.shown
        if self._unchanged_since_job():
            # Everything here works from the job's stored layers (pixels, mask, selection), not
            # from PhotoCraft's live selection, and the job is only forgotten once it worked: a
            # failure leaves the variations in the document and in the picker.
            image, base = self.image, self._job_base
            if which == "none":
                self.bridge.rewind(image, len(base))  # as if the job had not run: no step
            elif not self._collapsed or self._choice_kind == "fill":
                chosen = self.layers[shown]
                try:
                    self.bridge.keep_one(image, len(base), self.layers, chosen, re.sub(r" \(\d+\)$", "", chosen.name))
                except BaseException:
                    self._remember_history()
                    raise
            else:  # an expand: its canvas stays; the hidden variations go, still one step
                for i, layer in enumerate(self.layers):
                    if i != shown and layer.id is not None:
                        self.client.run("layer.delete", layer=layer.id)
                self.bridge.collapse(image, len(base), self.state.title or "OpenFill")
        else:
            # Other document history belongs to the user. Remove only existing
            # result layers by ID, even if they were moved out of their group.
            live_ids = {layer["id"] for layer in self.bridge._flat_layers(self.image)}
            if which == "selected" and self.layers[shown].id not in live_ids:
                raise RuntimeError("The selected variation is no longer in the document.")
            for i, layer in enumerate(self.layers):
                if layer.id in live_ids and (which == "none" or i != shown):
                    self.client.run("layer.delete", layer=layer.id)
        self._clear_choices()

    def _remember_history(self) -> None:
        """After a Keep that failed: the document's history as it is now, so Keep can be tried
        again when the variations are still (or again) the job's last step."""
        try:
            self._expected_history = self.bridge.history(self.image)
            self._expected_revision = self.bridge._revision(self.image)
        except ControlError:
            self._expected_history = []
            self._expected_revision = None

    def _unchanged_since_job(self) -> bool:
        """True when the document's history is still exactly as the last job and the picker
        left it (nothing else was done since), so the job's step can be taken back."""
        if not self._expected_history or self.image is None:
            return False
        try:
            return (
                self.bridge.history(self.image) == self._expected_history
                and self.bridge._revision(self.image) == self._expected_revision
            )
        except ControlError:
            return False


SETUP_STEPS = {
    "verifying": "Checking the download...",
    "unpacking": "Unpacking ComfyUI (a few minutes)...",
    "nodes": "Installing the inpaint nodes...",
    "packages": "Installing packages...",
}


def setup_step_text(state: str, done: int, total: int) -> str:
    """What the ComfyUI download is doing, for the panel."""
    from openfill import catalog as cat

    if state == "downloading":
        return f"Downloading ComfyUI: {cat.human_size(done)} of {cat.human_size(total)}"
    if state == "unpacking" and total:
        return f"Unpacking ComfyUI: {cat.human_size(done)} of about {cat.human_size(total)}"
    return SETUP_STEPS.get(state, "Getting ready...")


def gimp_comfy_dirs() -> list[str]:
    """ComfyUI folders set in OpenFill for GIMP's settings (newest GIMP first)."""
    import json
    import os
    import sys
    from pathlib import Path

    home = Path.home()
    if sys.platform == "win32":
        bases = [Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming") / "GIMP"]
    elif sys.platform == "darwin":
        bases = [home / "Library" / "Application Support" / "GIMP"]
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        bases = [config / "GIMP", home / ".var" / "app" / "org.gimp.GIMP" / "config" / "GIMP"]
    found = []
    for base in bases:
        for cfg in sorted(base.glob("*/openfill/config.json"), reverse=True):
            try:
                folder = json.loads(cfg.read_text(encoding="utf-8")).get("comfy_dir")
            except (OSError, ValueError):
                continue
            if isinstance(folder, str) and folder and folder not in found:
                found.append(folder)
    return found
