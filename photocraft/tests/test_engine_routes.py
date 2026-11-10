"""Per-mode engine changes cannot redirect a job, its cancellation or its inventory."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from openfill_photocraft.engine_routes import HELPER, SHARED, EngineRoutes
from openfill_photocraft.service import JobState, Service

from openfill.config import Settings


class RoutingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.profile = Path(temp.name)
        (self.profile / "selection-engine.json").write_text('{"port": 18777}')
        self.app = Mock()
        self.app.settings = Settings(remote_fibo=True)
        self.routes = EngineRoutes(self.app, self.profile)
        self.choices = dict(select=HELPER, edit="http://localhost:19901", remove="http://localhost:19902")

    def test_default_uses_one_engine_even_with_old_selection_config(self):
        self.assertFalse(self.routes.enabled)
        for mode in ("select", "fill", "remove", "expand", "txt2img"):
            self.assertIs(self.routes.app_for(mode), self.app)
        self.assertFalse(self.routes.sessions)

    def test_independent_servers_survive_switches_and_persist(self):
        self.routes.save(True, self.choices)
        pick, edit, remove = [self.routes.app_for(k) for k in ("select", "edit", "remove")]
        self.assertTrue(pick.selection_helper)
        self.assertEqual(remove.server.url(), self.choices["remove"])
        self.assertEqual(edit.server.url(), self.choices["edit"])
        self.routes.save(False, self.choices)
        self.assertIs(self.routes.app_for("edit"), self.app)
        self.routes.save(True, self.choices)
        self.assertIs(self.routes.app_for("edit"), edit)
        self.assertIs(self.routes.app_for("remove"), remove)
        self.assertEqual(EngineRoutes(self.app, self.profile).describe(), self.routes.describe())
        self.assertEqual(self.app.settings, Settings(remote_fibo=True))

    def test_atomic_validation_keeps_old_assignments(self):
        saved = self.routes.save(True, self.choices)
        old = self.routes.path.read_bytes()
        for value in (HELPER, "not-a-source", "http://someone:secret@host", "https://host:99999"):
            with self.assertRaises(ValueError):
                self.routes.save(False, dict(self.choices, remove=value))
            self.assertEqual(self.routes.path.read_bytes(), old)
            self.assertEqual(self.routes.describe(), saved)

    def test_model_choice_and_shared_preferences(self):
        settings = self.routes.base.settings
        self.routes.save(True, self.choices)
        edit = self.routes.app_for("edit")
        edit.settings.active_model = "a-model-on-the-edit-server"
        edit.settings.save(edit.paths.config)
        self.assertEqual(json.loads(edit.paths.config.read_text()), {"active_model": edit.settings.active_model})
        self.assertNotEqual(settings.active_model, edit.settings.active_model)
        settings.fibo_edit_mode = "replace"
        self.assertEqual(edit.settings.fibo_edit_mode, "replace")
        restarted = EngineRoutes(self.app, self.profile)
        self.assertEqual(restarted.app_for("fill").settings.active_model, edit.settings.active_model)

    def test_busy_assignment_and_cancel_stay_with_running_engine(self):
        self.routes.save(True, self.choices)
        service = Service.__new__(Service)
        service.app, service.routes = self.app, self.routes
        service._operation_lock = threading.Lock()
        service._thread = Mock()
        service._thread.is_alive.return_value = True
        service.state = JobState()
        service._job_app = self.routes.app_for("remove")
        service._job_app._active_backend = Mock()
        with self.assertRaisesRegex(RuntimeError, "current job"):
            service.configure_engines(False, dict.fromkeys(self.choices, SHARED))
        service.cancel()
        service._job_app._active_backend.cancel.assert_called_once_with()
        self.app._active_backend.cancel.assert_not_called()
