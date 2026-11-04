# SPDX-License-Identifier: Apache-2.0
"""Exercise the native sections and attachments without connecting to PhotoCraft or ComfyUI."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

try:
    import tkinter as tk
except ImportError as err:
    raise unittest.SkipTest("Tk is unavailable") from err

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill.config import Settings  # noqa: E402
from openfill_photocraft import ui  # noqa: E402
from openfill_photocraft.bar import Bar  # noqa: E402
from openfill_photocraft.references import ReferenceImages  # noqa: E402
from openfill_photocraft.web import Api  # noqa: E402


class InlinePanelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = Path(tmp.name)
        self.qwen, self.sdxl = "qwen-model", "sdxl-model"
        self.qwen_name, self.sdxl_name = "Qwen Lightning · Local ComfyUI", "SDXL · Local ComfyUI"
        self.service = service = Mock()
        service.app.settings = Settings(remote_fibo=True, active_model=self.qwen)
        service.app.paths.config = self.profile / "openfill" / "config.json"
        service.app.source_label.return_value = "Local ComfyUI"
        service.engine = "ready"
        service.busy.return_value = False
        service.state.snapshot.return_value = {"state": "idle", "variations": []}
        service.references = ReferenceImages()
        service.reference_rows.side_effect = service.references.rows
        service.add_reference_files.side_effect = service.references.add_files
        watcher = Mock()
        watcher.snapshot.return_value = {"connected": True, "document": {"name": "Edit.png"},
                                         "selection": [0, 0, 64, 64]}
        rows = [{"id": model_id, "name": name, "runnable": True, "installed": True,
                 "active": model_id == self.qwen, "partial": [], "remote": True,
                 "max_references": 15 if model_id == self.qwen else 0,
                 "reference_syntax": "<imageN>" if model_id == self.qwen else ""}
                for model_id, name in ((self.qwen, self.qwen_name), (self.sdxl, self.sdxl_name))]
        refresher = Mock()
        refresher.get.return_value = {"comfy_ok": True, "models": {"fill": [(r["id"], r["name"]) for r in rows]},
                                      "rows": rows, "active": self.qwen, "active_name": self.qwen_name}
        native_tk = tk.Tk

        def hidden_root(*args, **kwargs):
            try:
                root = native_tk(*args, **kwargs)
            except tk.TclError as err:
                self.skipTest(f"Tk cannot open a test window: {err}")
            root.withdraw()
            self.addCleanup(self.close_root, root)
            return root

        self.api = Api(service)
        with patch("tkinter.Tk", side_effect=hidden_root), \
             patch("openfill_photocraft.bar.Refresher", return_value=refresher), \
             patch("openfill_photocraft.tether.Tether"), \
             patch("openfill_photocraft.ui.make_borderless", return_value=False), \
             patch.object(Bar, "_tick", lambda self: None), \
             patch.object(Bar, "_drain", lambda self: None), \
             patch.object(Bar, "_place", lambda self: self.root.withdraw()):
            self.bar = Bar(service, watcher, "http://127.0.0.1/test", self.profile, api=self.api)
        self.bar.refresh_model_visibility()

    @staticmethod
    def close_root(root):
        root.update_idletasks()
        for callback in root.tk.splitlist(root.tk.call("after", "info")):
            root.after_cancel(callback)
        ui._cache.clear()
        ui.FONTS = None
        root.destroy()

    def test_drag_and_collapse_persist_and_restore(self):
        from openfill_photocraft.panels import SectionStack

        bar = self.bar
        self.assertIs(bar.api, self.api)
        root, stack = bar.root, bar.sections
        root.attributes("-alpha", 0.0)
        root.attributes("-topmost", False)
        root.deiconify()
        root.update()
        self.assertIn("engine", stack.collapsed)
        engine, edit = stack.grips["engine"], stack.headers["edit"]
        stack._start_drag("engine", SimpleNamespace(widget=engine, y_root=engine.winfo_rooty() + 10))
        stack._drop(SimpleNamespace(y_root=edit.winfo_rooty() + 2))
        self.assertEqual(stack.order[0], "engine")
        self.assertIn("engine", stack.collapsed)
        stack.toggles["edit"].invoke()
        self.assertFalse(stack.bodies["edit"].winfo_manager())
        loaded = Settings.load(self.service.app.paths.config).panel_layout
        new_stack = SectionStack(bar.inner, bar.scroller, loaded, Mock(), Mock())
        self.assertEqual(new_stack.order[0], "engine")
        self.assertIn("edit", new_stack.collapsed)
        self.assertFalse(new_stack.bodies["edit"].winfo_manager())
        self.assertTrue(new_stack.bodies["references"].winfo_manager())

    def test_model_visibility_changes_only_the_dropdown(self):
        bar = self.bar
        self.assertEqual(bar.model_box.values, [self.qwen_name, self.sdxl_name])
        bar.models_view._rows[self.qwen]["visibility"].set(False)
        bar.refresh_model_visibility()
        self.assertEqual(bar.model_box.values, [self.sdxl_name])
        self.assertEqual(bar._model(), self.sdxl)
        self.assertIn(self.qwen, bar.models_view._rows)
        self.assertFalse(bar.models_view._rows[self.qwen]["visibility"].get())
        loaded = Settings.load(self.service.app.paths.config)
        self.assertEqual(loaded.hidden_models, [self.qwen])
        self.assertEqual(loaded.active_model, self.qwen)
        bar.models_view._rows[self.sdxl]["visibility"].set(False)
        bar.refresh_model_visibility()
        self.assertTrue(bar.go.disabled)
        self.assertEqual(bar.model_box.get(), "No models shown")
        bar.models_view._rows[self.qwen]["visibility"].set(True)
        bar.refresh_model_visibility()
        self.assertEqual(bar.model_box.values, [self.qwen_name])
        self.assertFalse(bar.go.disabled)

    def test_attach_reorder_and_send_only_to_a_supported_model(self):
        bar = self.bar
        files = [self.profile / "first.png", self.profile / "second.png"]
        for path, color in zip(files, ("red", "blue")):
            Image.new("RGB", (16, 24), color).save(path)
        with patch("tkinter.filedialog.askopenfilenames", return_value=tuple(map(str, files))):
            bar.references.add.invoke()
            callback, args = bar.calls.get(timeout=5)
            callback(*args)
        self.assertEqual([r["name"] for r in bar.references.rows], ["first.png", "second.png"])
        bar.references._move(1, -1)
        self.assertEqual([r["name"] for r in bar.references.rows], ["second.png", "first.png"])
        top_row = bar.references.list.winfo_children()[0]
        self.assertIn("<image2>  second.png", top_row.winfo_children()[-1].cget("text"))
        bar.prompt.set("Add the object from <image2>.")
        bar.generate()
        self.assertTrue(self.service.start.call_args.args[1]["useReferences"])
        bar.model_choice.set(self.sdxl_name)
        bar._fill_models()
        self.assertTrue(bar.references.add.disabled)
        self.assertIn("runs without them", bar.references.help.cget("text"))
        bar.generate()
        self.assertNotIn("useReferences", self.service.start.call_args.args[1])
        bar.references._remove(bar.references.rows[0]["id"])
        self.assertEqual([r["name"] for r in bar.references.rows], ["first.png"])


if __name__ == "__main__":
    unittest.main()
