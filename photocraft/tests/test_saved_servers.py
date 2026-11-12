"""Saved ComfyUI endpoints, discovery, and managed-profile switching."""

import json
import logging
import tempfile
import threading
import unittest
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openfill_photocraft import server_discovery
from openfill_photocraft.engine_routes import EngineRoutes, RoutedApp
from openfill_photocraft.service import Service
from openfill_photocraft.web import Api

from openfill.app import App
from openfill.backend import HealthStatus
from openfill.config import Paths, Settings
from openfill.sources import Source, get_source, new_source, server_address, sources


class SavedServerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.profile = Path(folder.name)
        base = App.__new__(App)
        base.settings = Settings(remote_fibo=True)
        base.paths = Paths.for_profile(self.profile).ensure()
        base.log = logging.getLogger("openfill-test")
        base._source_lock = threading.RLock()
        base.io, base.runner_cls, base.interactive, base.plugin_dir, base.job_state = (
            Mock(),
            Mock(),
            True,
            self.profile,
            None,
        )
        base.check_backend = Mock(return_value=HealthStatus(False, "offline"))
        self.svc = Service.__new__(Service)
        self.svc.app, self.svc._thread = base, None
        self.svc._operation_lock = threading.Lock()
        self.svc.routes = EngineRoutes(base, self.profile)
        self.api = Api(self.svc)

    def test_saved_server_drives_shared_and_mode_routes_after_restart(self):
        first = self.api.post("/api/servers/add", {"name": "Studio", "address": "studio.local:8189/"})
        second = self.api.post("/api/servers/add", {"name": "Studio B", "address": "studio.local:8190"})
        self.assertEqual(first, self.api.post("/api/servers/add", {"address": "http://studio.local:8189"}))
        self.assertEqual(len(self.api.sources()["sources"]), 3)
        result = self.api.post("/api/backend/source", {"source": first["id"]})
        self.assertFalse(result["ok"])
        self.assertEqual(self.svc.app.settings.active_url, "http://studio.local:8189")
        choices = dict(select="shared", edit="shared", remove=second["id"])
        self.svc.configure_engines(True, choices)
        self.svc.app.reload_settings()
        restarted = EngineRoutes(self.svc.app, self.profile)
        self.assertIs(restarted.app_for("edit"), self.svc.app)
        self.assertEqual(restarted.app_for("remove").settings.active_url, "http://studio.local:8190")
        self.assertEqual(restarted.describe()["shared"], "Studio")

    def test_managed_profile_can_use_a_server_and_return(self):
        self.svc.app.settings.remote_fibo = False
        original = self.svc.app.settings.comfy_dir
        added = self.svc.add_server("Studio", "studio.local")
        with patch.object(RoutedApp, "check_backend", return_value=HealthStatus(True, "ready")):
            self.svc.select_source(added["id"])
        restarted = EngineRoutes(self.svc.app, self.profile)
        self.assertEqual(restarted.app_for("select").settings.active_url, "http://studio.local:8188")
        self.svc.select_source("managed")
        self.assertIs(self.svc.app_for("edit"), self.svc.app)
        self.assertEqual(self.svc.app.settings.comfy_dir, original)
        self.assertTrue(self.svc.app.settings.autostart)

    def test_damaged_entry_duplicate_and_failed_write(self):
        self.svc.app.settings.comfy_servers = [None, {"id": "../bad"}]
        self.assertEqual(len(self.svc.routes.sources), 1)
        self.svc.add_server("Bench", "bench.local")
        before = self.svc.app.paths.config.read_bytes()
        with self.assertRaisesRegex(ValueError, "name"):
            self.svc.add_server("bench", "elsewhere.local")
        with patch.object(Settings, "save", side_effect=OSError("disk")), self.assertRaises(OSError):
            self.svc.add_server("Other", "other.local")
        self.assertEqual(len(self.svc.routes.sources), 2)
        self.assertEqual(self.svc.app.paths.config.read_bytes(), before)

    def test_safe_addresses_and_distinct_forwards(self):
        for address in (
            "http://user:secret@host",
            "ssh://-x:22",
            "http://host:99999",
            "http://0.0.0.0:8188",
            "ftp://host",
            "http://host?key=x",
            "ssh://studio:8188/path",
        ):
            with self.subTest(address=address), self.assertRaises(ValueError):
                server_address(address)
        self.assertEqual(server_address("https://STUDIO.local:443/comfy/"), "https://studio.local/comfy")
        self.assertEqual(server_address("[::1]:8188"), "http://[::1]:8188")
        one = new_source("Bench", "ssh://studio:8188", sources())
        two = new_source("Second", "ssh://studio:8000", (*sources(), one))
        self.assertNotEqual(one.url, two.url)
        saved = get_source(two.id, [asdict(s) for s in (one, two)])
        self.assertEqual(saved.address, "ssh://studio:8000")
        self.assertEqual(saved, two)

    def test_search_lists_only_comfy_and_keeps_saved_names(self):
        good = {"system": {"comfyui_version": "test"}, "devices": []}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(good if self.path == "/system_stats" else {"ok": True}).encode())

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"
        self.assertEqual(server_discovery.probe(url), "test")
        self.assertIsNone(server_discovery.probe(url + "/unrelated"))
        with (
            patch.object(server_discovery, "local_network", return_value=None),
            patch.object(server_discovery, "PORTS", ()),
        ):
            discovered = server_discovery.discover([Source("studio", "Studio", url)])
        self.assertEqual(discovered, [{"address": url, "name": "Studio", "saved": True, "version": "test"}])

    def test_active_job_cannot_be_rerouted(self):
        self.svc._thread = SimpleNamespace(is_alive=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "current edit"):
            self.svc.select_source("local")
