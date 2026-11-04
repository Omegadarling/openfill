# SPDX-License-Identifier: Apache-2.0
import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.references import ReferenceImages  # noqa: E402
from openfill_photocraft.web import Api  # noqa: E402

from openfill.config import PANEL_SECTIONS, Settings  # noqa: E402


def png(color):
    buffer = io.BytesIO()
    Image.new("RGBA", (12, 8), color).save(buffer, "PNG")
    return buffer.getvalue()


class PanelPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        self.app = SimpleNamespace(settings=Settings(active_model="qwen"), paths=SimpleNamespace(config=self.path))
        self.service = SimpleNamespace(app=self.app, references=ReferenceImages())
        self.api = Api(self.service)

    def test_layout_and_visibility_are_independent_preferences(self):
        self.api.post("/api/layout", {"order": ["models", "edit", "models", "stale", 9], "collapsed": ["edit"]})
        self.api.post("/api/models/visibility", {"id": "qwen", "visible": False})
        saved = Settings.load(self.path)
        self.assertEqual(
            saved.panel_layout,
            {
                "order": ["models", "edit", "references", "engine", "generation"],
                "collapsed": ["edit"],
            },
        )
        self.assertEqual(set(saved.panel_layout["order"]), set(PANEL_SECTIONS))
        self.assertEqual(saved.hidden_models, ["qwen"])
        self.assertEqual(saved.active_model, "qwen")
        self.api.post("/api/models/visibility", {"id": "qwen", "visible": True})
        self.assertEqual(Settings.load(self.path).hidden_models, [])

    def test_failed_preference_write_restores_state(self):
        self.path.mkdir()  # rename onto a directory fails
        with self.assertRaises(OSError):
            self.api.post("/api/models/visibility", {"id": "qwen", "visible": False})
        self.assertEqual(self.app.settings.hidden_models, [])

    def test_web_attachments_can_be_reordered_and_removed(self):
        def attach(name, color):
            return self.api.post("/api/references", {"name": name, "data": base64.b64encode(png(color)).decode()})

        attach("C:\\private\\first.png", "red")
        first, second = attach("second.png", "blue")
        self.assertEqual(first["name"], "first.png")
        frozen = self.service.references.snapshot()
        with self.assertRaises(ValueError):
            self.api.post("/api/references/order", {"ids": [second["id"], second["id"]]})
        self.api.post("/api/references/order", {"ids": [second["id"], first["id"]]})
        self.assertEqual(self.service.references.snapshot(), frozen[::-1])
        self.api.post("/api/references/remove", {"id": second["id"]})
        self.assertEqual(self.service.references.snapshot(), frozen[:1])
        self.assertFalse(self.path.exists())

    def test_native_multi_file_attachment_is_atomic(self):
        good = Path(self.tmp.name) / "valid.png"
        bad = Path(self.tmp.name) / "invalid.png"
        good.write_bytes(png("red"))
        bad.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            self.service.references.add_files([good, bad])
        self.assertEqual(self.service.references.rows(), [])
        self.assertEqual(self.service.references.add_files([good])[0]["width"], 12)
