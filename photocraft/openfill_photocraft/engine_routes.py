# SPDX-License-Identifier: Apache-2.0
"""Resolve each PhotoCraft operation to its engine without changing the shared source."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from openfill.app import App
from openfill.backend import BackendUnavailable, HealthStatus
from openfill.comfy_api import ComfyClient
from openfill.config import Paths
from openfill.remote_models import RemoteModels
from openfill.server import ServerManager
from openfill.sources import Source, new_source, sources

from .comfy_connection import ensure_connection

MODES = ("select", "edit", "remove")
SHARED = "shared"
HELPER = "selection-helper"
MANAGED = "managed"


def mode_for(tool, removing=False):
    if removing or tool == "remove":
        return "remove"
    if tool in ("fill", "expand", "txt2img", "edit"):
        return "edit"
    if tool == "select":
        return tool
    raise ValueError("Choose Select, Edit or Remove.")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as file:
            temp = Path(file.name)
            json.dump(value, file, indent=2)
            file.write("\n")
        os.replace(temp, path)
    finally:
        if temp:
            temp.unlink(missing_ok=True)


def custom_source(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Enter a ComfyUI server URL.")
    value = value.strip().rstrip("/")
    url = urlsplit(value)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or any(c.isspace() for c in value)
    ):
        raise ValueError("Enter an HTTP or HTTPS ComfyUI URL without credentials, a query or a fragment.")
    if url.port == 0:
        raise ValueError("Choose a ComfyUI server port from 1 to 65535.")
    return Source(value, url.netloc + url.path, value)


class RoutedSettings:
    """Share editing preferences; remember the active edit model per additional engine."""

    remote_fibo = True
    backend = "comfyui"
    autostart = False

    def __init__(self, base, source, path):
        self.base = base
        self.fibo_source = source.id
        self.url = self.active_url = source.url
        self.active_model = base.active_model
        if path.is_file():
            self.active_model = json.loads(path.read_text(encoding="utf-8"))["active_model"]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def save(self, path):
        write_json(path, {"active_model": self.active_model})


class ExternalServer(ServerManager):
    def __init__(self, settings, paths, log, source, profile):
        super().__init__(settings, paths, log)
        self.source, self.profile = source, profile

    def ensure_running(self, progress=None, cancel=None):
        if self.source.ssh_alias:
            ensure_connection(self.source.url, self.profile, source_id=self.source.id)
        elif not ComfyClient(self.source.url, timeout=3).is_alive(timeout=3):
            raise BackendUnavailable(f"{self.source.label} is not answering at {self.source.url}.")
        return self.source.url


class RoutedApp(App):
    """One existing ComfyUI endpoint, using the base app's editor, progress and preferences."""

    def __init__(self, base, source, profile):
        self.io, self.runner_cls, self.interactive = base.io, base.runner_cls, base.interactive
        self.log, self.plugin_dir, self.job_state = base.log, base.plugin_dir, base.job_state
        key = hashlib.sha256(source.url.encode()).hexdigest()[:20]
        self.paths = Paths(profile / "engine-cache" / key)
        self.settings = RoutedSettings(base.settings, source, self.paths.config)
        self.source = source
        self.selection_helper = source.id == HELPER
        self.server = ExternalServer(self.settings, self.paths, self.log, source, profile)
        self.remote_models = RemoteModels(source.url, source_label=source.label)
        self.catalog = []
        self._source_lock = threading.RLock()
        self._active_backend = None
        self._opencv_noted = self._device_checked = self._attempted_tunnel = False

    def source_label(self):
        return self.source.label

    def refresh_remote_models(self, force=False):
        with self._source_lock:
            if self.source.ssh_alias and not self._attempted_tunnel:
                self._attempted_tunnel = True
                try:
                    self.server.ensure_running()
                except Exception as err:
                    self.log.info("Engine %s: %s", self.source.label, err)
        super().refresh_remote_models(force=force)

    def check_backend(self):
        try:
            backend = self.backend()
            health = backend.health()
            self.refresh_remote_models(force=True)
            if health.ok and not self.catalog:
                return HealthStatus(False, self.remote_models.error)
            return health
        except Exception as err:
            return HealthStatus(False, str(err))


class EngineRoutes:
    def __init__(self, base, profile):
        self.base, self.profile = base, Path(profile)
        self.path = self.profile / "engine-routing.json"
        self.lock = threading.RLock()
        self.sessions = {}
        self.managed_shared = MANAGED
        self.helper_config = self.profile / "selection-engine.json"
        self.enabled = False
        self.routes = dict.fromkeys(MODES, SHARED)
        if self.helper_config.is_file():
            self.routes["select"] = HELPER
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8-sig"))
                self._validate(saved["enabled"], saved["routes"])
                shared = saved.get("shared", MANAGED)
                if shared != MANAGED and shared not in self.sources:
                    raise ValueError("The saved shared server is unknown.")
                self.enabled, self.routes, self.managed_shared = saved["enabled"], saved["routes"], shared
            except (KeyError, ValueError, TypeError, OSError) as err:
                base.log.warning("Could not read engine assignments: %s", err)

    @property
    def sources(self):
        return {s.id: s for s in sources(self.base.settings.comfy_servers)}

    def add_server(self, name, address):
        with self.lock:
            existing = tuple(self.sources.values())
            source = new_source(name, address, existing)
            if source not in existing:
                settings = self.base.settings
                settings.save_server(self.base.paths.config, asdict(source))
            return source

    @property
    def shared_id(self):
        return self.base.settings.fibo_source if self.base.settings.remote_fibo else self.managed_shared

    def select_shared(self, key):
        """A managed profile can use an external shared engine and return to its original setup."""
        with self.lock:
            if key != MANAGED and key not in self.sources:
                raise ValueError("Choose a saved ComfyUI server.")
            self._write(self.enabled, self.routes, key)
            self.managed_shared = key
            return self._app_for_key(key, "edit")

    def _write(self, enabled, routes, shared):
        write_json(self.path, {"enabled": enabled, "routes": routes, "shared": shared})

    def _source(self, key, mode):
        if key == SHARED:
            return None
        if key == HELPER:
            if mode != "select" or not self.helper_config.is_file():
                raise ValueError("The local WeVisOne engine is available for Select only.")
            port = int(json.loads(self.helper_config.read_text(encoding="utf-8-sig"))["port"])
            if not 1 <= port <= 65535:
                raise ValueError("Invalid local selection engine port.")
            return Source(HELPER, "Local WeVisOne", f"http://127.0.0.1:{port}")
        return self.sources.get(key) or custom_source(key)

    def _validate(self, enabled, routes):
        if type(enabled) is not bool or not isinstance(routes, dict) or set(routes) != set(MODES):
            raise ValueError("Choose one engine for each of Select, Edit and Remove.")
        for mode, key in routes.items():
            if not isinstance(key, str):
                raise ValueError("Choose an engine or enter its ComfyUI URL.")
            self._source(key, mode)

    def save(self, enabled, routes):
        with self.lock:
            self._validate(enabled, routes)
            cleaned = {}
            for mode, key in routes.items():
                source = self._source(key, mode)
                cleaned[mode] = source.id if source else SHARED
            self._write(enabled, cleaned, self.managed_shared)
            self.enabled, self.routes = enabled, cleaned
            return self.describe()

    def describe(self):
        with self.lock:
            registered = self.sources
            shared_id = self.shared_id
            shared = "Managed ComfyUI" if shared_id == MANAGED else registered[shared_id].label
            servers = [{"id": s.id, "name": s.label, "address": s.address} for s in registered.values()]
            if not self.base.settings.remote_fibo:
                servers.insert(0, {"id": MANAGED, "name": "Managed ComfyUI", "address": ""})
            choices = [{"id": SHARED, "name": f"Shared · {shared}", "modes": MODES}]
            choices.extend({"id": s.id, "name": s.label, "modes": MODES} for s in registered.values())
            if self.helper_config.is_file():
                choices.append({"id": HELPER, "name": "Local WeVisOne", "modes": ["select"]})
            for key in dict.fromkeys(self.routes.values()):
                if key not in (*registered, HELPER, SHARED):
                    choices.append({"id": key, "name": custom_source(key).label, "modes": MODES})
            return {
                "enabled": self.enabled,
                "routes": dict(self.routes),
                "choices": choices,
                "shared": shared,
                "sharedId": shared_id,
                "servers": servers,
            }

    def app_for(self, tool):
        with self.lock:
            mode = mode_for(tool)
            key = self.routes[mode] if self.enabled else SHARED
            return self._app_for_key(self.shared_id if key == SHARED else key, mode)

    def _app_for_key(self, key, mode):
        if key == MANAGED or (self.base.settings.remote_fibo and key == self.base.settings.fibo_source):
            return self.base
        source = self._source(key, mode)
        if source.id not in self.sessions:
            self.sessions[source.id] = RoutedApp(self.base, source, self.profile)
        app = self.sessions[source.id]
        app.settings.base = self.base.settings
        return app
