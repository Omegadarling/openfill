# OpenFill for PhotoCraft
# SPDX-License-Identifier: Apache-2.0
"""The companion's page and JSON API on 127.0.0.1 (standard library only).

Every request must carry the session key the launcher put in the page's URL, so
another local program cannot drive PhotoCraft through it.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import secrets
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from openfill import catalog as cat
from openfill.backend import describe_error
from openfill.config import normalize_panel_layout, validate_fibo_settings

from . import __version__
from .references import MAX_UPLOAD
from .service import Service

STATIC = Path(__file__).with_name("static")

# Settings the page may change (the rest stay in config.json).
EDITABLE = {
    "comfy_dir": str,
    "autostart": bool,
    "idle_timeout_minutes": int,
    "extra_server_args": str,
    "steps": int,
    "cfg": float,
    "grow_percent": float,
    "feather_percent": float,
    "auto_feather": bool,
    "auto_feather_percent": float,
    "fibo_edit_mode": str,
    "fibo_context_mode": str,
    "fibo_context_percent": float,
    "fibo_boundary_match": bool,
    "fibo_boundary_strength": float,
    "debug": bool,
}


class Api:
    def __init__(self, service: Service) -> None:
        self.service = service
        self.key = secrets.token_urlsafe(24)
        self.on_show = None  # brings the OpenFill bar up (set by the launcher)
        self._pick = {"seq": 0, "action": ""}  # the last Filter menu pick, for the page
        self._preferences_lock = threading.Lock()

    def pick(self, action: str) -> None:
        self._pick = {"seq": self._pick["seq"] + 1, "action": action}

    # GET ------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        app = self.service.app
        return {
            "version": __version__,
            "job": self.service.state.snapshot(),
            "photocraft": self.service.photocraft(),
            "busy": self.service.busy(),
            "remoteFibo": app.settings.remote_fibo,
            "backendUrl": app.settings.url if app.settings.remote_fibo else "",
            "backendSource": app.settings.fibo_source if app.settings.remote_fibo else "",
            "backendName": app.source_label() if app.settings.remote_fibo else "",
            "activeModel": app.active_model_id(),
            "comfyReady": _comfy_set(app),
            "comfyProblem": "" if _comfy_set(app) else _comfy_problem(app),
            "setup": None if _comfy_set(app) else self.service.setup_progress(),
            "pick": self._pick,
        }

    def models(self) -> dict[str, Any]:
        from .removal_advice import removal_scenes

        app = self.service.app
        source_id = app.settings.fibo_source if app.settings.remote_fibo else ""
        try:
            rows = app.model_rows(app.server_model_files())
        except Exception as err:  # noqa: BLE001
            return {
                "rows": [],
                "removalScenes": removal_scenes([], app.settings.remote_fibo),
                "error": describe_error(err),
                "source": source_id,
            }
        available = [
            r["id"]
            for r in rows
            if r.get("visible", True)
            and r["installed"]
            and r["runnable"]
            and "remove" in r["capabilities"]
            and not r["missing_nodes"]
            and not r["missing_models"]
        ]
        scenes = removal_scenes(available, app.settings.remote_fibo)
        status = app.task_status()
        current = status.get("current") or {}
        pending = [t.get("model_id") for t in app.server.pending_tasks()]
        from .modelinfo import hardware_line, needed_by

        for row in rows:
            row["statusText"] = cat.row_status(row, current, pending)
            if not row["runnable"]:
                row["neededBy"] = needed_by(list(app.catalog), row["id"])
        if app.settings.remote_fibo:
            from openfill.sources import get_source

            if source_id != app.settings.fibo_source:
                rows, source_id = [], app.settings.fibo_source
                scenes = removal_scenes([], True)
            return {
                "rows": rows,
                "removalScenes": scenes,
                "current": current,
                "pending": pending,
                "source": source_id,
                "hardware": f"{get_source(source_id).label} · ComfyUI",
            }
        hw = None
        try:
            from openfill import hardware

            detected = hardware.detect(app.models_root, app.server.url())
            hw = hardware_line(detected)
            if detected.is_amd:
                cat.add_amd_notes(rows, app.catalog)  # what may not work on an AMD card
        except Exception:  # noqa: BLE001
            hw = None
        return {"rows": rows, "removalScenes": scenes, "current": current, "pending": pending, "hardware": hw}

    def thumbnail(self, index: int) -> bytes:
        """A small JPEG of variation ``index`` of the last job (for the page's picker)."""
        layers = self.service.layers
        if not 0 <= index < len(layers):
            return b""
        import io

        from PIL import Image

        layer = layers[index]
        im = Image.frombytes("RGBA", (layer.origin.width, layer.origin.height), layer.rgba).convert("RGB")
        im.thumbnail((320, 240))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def settings(self) -> dict[str, Any]:
        s = self.service.app.settings
        return {k: getattr(s, k) for k in EDITABLE}

    def layout(self):
        return normalize_panel_layout(self.service.app.settings.panel_layout)

    def _save_preference(self, key, value):
        app = self.service.app
        settings = app.settings
        previous = getattr(settings, key)
        setattr(settings, key, value)
        try:
            settings.save(app.paths.config)
        except OSError:
            setattr(settings, key, previous)
            raise

    def sources(self) -> dict[str, Any]:
        from openfill.sources import list_sources

        selected = self.service.app.settings.fibo_source
        return {
            "selected": selected,
            "sources": [dict(item, name=item["label"], selected=item["id"] == selected) for item in list_sources()],
        }

    # POST -----------------------------------------------------------------------------

    def post(self, path: str, body: dict[str, Any]) -> Any:
        svc, app = self.service, self.service.app
        if path == "/api/removal-advice":
            available = body.get("available")
            if (
                not isinstance(available, list)
                or len(available) > 512
                or not all(isinstance(i, str) for i in available)
            ):
                raise ValueError("Expected a list of available removal model IDs.")
            return svc.removal_advice(available)
        if path in ("/api/fill", "/api/expand", "/api/txt2img"):
            svc.start(path.rsplit("/", 1)[1], body)
            return {"started": True}
        if path == "/api/show":
            if self.on_show:
                self.on_show(str(body.get("action") or "panel"))
            return {"ok": True}
        if path == "/api/open":
            return svc.bridge.open_file(str(body["path"]))
        if path == "/api/cancel":
            svc.cancel()
            return {"ok": True}
        if path == "/api/variation":
            svc.show_variation(int(body.get("index", 0)))
            return {"ok": True}
        if path == "/api/keep":
            svc.keep(str(body.get("which", "selected")))
            return {"ok": True}
        if path == "/api/models/download":
            return {"tasks": app.queue_download(str(body["id"]))}
        if path == "/api/models/activate":
            entry = app.set_active(str(body["id"]))
            return {"active": entry.id}
        if path == "/api/models/visibility":
            model_id, visible = body.get("id"), body.get("visible")
            if not isinstance(model_id, str) or not 0 < len(model_id) <= 256 or not isinstance(visible, bool):
                raise ValueError("Provide a model ID and choose whether it is visible.")
            with self._preferences_lock:
                hidden = set(app.settings.hidden_models)
                if visible:
                    hidden.discard(model_id)
                else:
                    hidden.add(model_id)
                self._save_preference("hidden_models", sorted(hidden))
            return {"id": model_id, "visible": visible}
        if path == "/api/layout":
            layout = normalize_panel_layout(body)
            with self._preferences_lock:
                self._save_preference("panel_layout", layout)
            return layout
        if path == "/api/references":
            name, data = body.get("name"), body.get("data")
            if not isinstance(name, str) or not isinstance(data, str) or len(data) > 4 * ((MAX_UPLOAD + 2) // 3):
                raise ValueError("Choose a PNG, JPEG or WebP image of up to 16 MB.")
            try:
                decoded = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("The reference image could not be decoded.") from exc
            return svc.references.add(name, decoded)
        if path == "/api/references/remove":
            return svc.references.remove(body.get("id"))
        if path == "/api/references/order":
            return svc.references.order(body.get("ids"))
        if path == "/api/models/remove":
            return {"task": svc.uninstall(str(body["id"]))}
        if path == "/api/models/stop":
            return {"stopped": svc.stop_download(str(body["id"]))}
        if path == "/api/models/nodes":
            return {"task": app.queue_node_install(str(body["id"]))}
        if path == "/api/models/token":
            app.save_token(str(body.get("token", "")))
            return {"ok": True}
        if path == "/api/settings":
            # Validate before changing settings so a rejected amount cannot
            # partially save the toggle or other form values.
            if "fibo_boundary_match" in body and not isinstance(body["fibo_boundary_match"], bool):
                raise ValueError("Boundary color matching must be on or off.")
            values = {key: kind(body[key]) for key, kind in EDITABLE.items() if key in body}
            values = validate_fibo_settings(values)
            if "auto_feather_percent" in values:
                amount = values["auto_feather_percent"]
                if not math.isfinite(amount):
                    raise ValueError("Auto-feather area must be a finite percentage.")
                values["auto_feather_percent"] = max(0.0, min(100.0, amount))
            with self._preferences_lock:
                for key, value in values.items():
                    setattr(app.settings, key, value)
                if app.settings.remote_fibo:
                    app.settings.autostart = False
                app.settings.save(app.paths.config)
                app.reload_settings()
            return self.settings()
        if path == "/api/comfy/setup":
            return {"task": svc.setup_comfy()}
        if path == "/api/backend/check":
            health = svc.check_backend()
            return {"ok": health.ok, "detail": health.detail, "device": getattr(health, "device", "")}
        if path == "/api/backend/source":
            return svc.select_source(str(body["source"]))
        if path == "/api/backend/stop":
            return {"message": app.stop_backend()}
        if path == "/api/logs":
            app.open_log_folder()
            return {"ok": True}
        raise KeyError(path)


def _comfy_set(app) -> bool:
    """A ComfyUI folder OpenFill can use (models and nodes live there), not just a server answering."""
    from openfill.config import comfy_layout

    return app.comfy_ready() if app.settings.remote_fibo else comfy_layout(app.settings.comfy_dir) is not None


def _comfy_problem(app) -> str:
    from openfill.config import comfy_problem

    if app.settings.remote_fibo:
        return app.remote_models.error or f"Start ComfyUI on {app.source_label()}, then press Check."
    return comfy_problem(app.settings.comfy_dir) or "ComfyUI is not set up."


def make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenFillPhotoCraft"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet console
            pass

        def _authorised(self) -> bool:
            query = parse_qs(urlparse(self.path).query)
            return secrets.compare_digest((query.get("key") or [""])[0], api.key) or secrets.compare_digest(
                self.headers.get("X-OpenFill-Key", ""), api.key
            )

        def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: Any) -> None:
            self._send(code, json.dumps(data, default=str).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not self._authorised():
                self._send(403, b"Open OpenFill from its launcher (the address carries a key).", "text/plain")
                return
            try:
                if path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/ping":
                    self._json(200, {"ok": True, "version": __version__})
                elif path == "/api/state":
                    self._json(200, api.state())
                elif path == "/api/models":
                    self._json(200, api.models())
                elif path == "/api/sources":
                    self._json(200, api.sources())
                elif path == "/api/settings":
                    self._json(200, api.settings())
                elif path == "/api/layout":
                    self._json(200, api.layout())
                elif path == "/api/references":
                    self._json(200, api.service.reference_rows())
                elif path == "/api/comfy/preflight":  # what Download ComfyUI would fetch (read-only)
                    self._json(200, api.service.comfy_preflight())
                elif path == "/api/thumb":
                    data = api.thumbnail(int((parse_qs(urlparse(self.path).query).get("i") or ["0"])[0]))
                    if data:
                        self._send(200, data, "image/jpeg")
                    else:
                        self._send(204, b"", "text/plain")
                elif path == "/api/preview":
                    st = api.service.state
                    with st.lock:
                        data, mime = st.preview, st.preview_mime or "image/jpeg"
                    if not data:
                        self._send(204, b"", "text/plain")
                    else:
                        self._send(200, data, mime)
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as err:  # noqa: BLE001
                self._json(500, {"error": describe_error(err)})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorised():
                self._send(403, b"forbidden", "text/plain")
                return
            try:
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length") or 0)
                limit = 4 * ((MAX_UPLOAD + 2) // 3) + 4096 if path == "/api/references" else 1_000_000
                if not 0 <= length <= limit:
                    self.close_connection = True
                    self._json(413, {"error": "The request is too large."})
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                self._json(200, api.post(path, body))
            except KeyError as err:
                self._json(404, {"error": f"unknown request {err}"})
            except Exception as err:  # noqa: BLE001
                self._json(400, {"error": describe_error(err) if not isinstance(err, RuntimeError) else str(err)})

    return Handler


class _LoopbackHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer resolves the host name. The local panel only uses its numeric address.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def serve(api: Api, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the page on 127.0.0.1 in a background thread; returns (server, url with key)."""
    server = _LoopbackHTTPServer(("127.0.0.1", port), make_handler(api))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="openfill-web", daemon=True).start()
    host, real_port = server.server_address[:2]
    return server, f"http://{host}:{real_port}/?key={api.key}"
