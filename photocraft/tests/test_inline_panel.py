# SPDX-License-Identifier: Apache-2.0
"""Exercise the native sections and attachments without connecting to PhotoCraft or ComfyUI."""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

try:
    import tkinter as tk
except ImportError as err:
    raise unittest.SkipTest("Tk is unavailable") from err

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft import ui  # noqa: E402
from openfill_photocraft.bar import Bar  # noqa: E402
from openfill_photocraft.references import ReferenceImages  # noqa: E402
from openfill_photocraft.web import Api  # noqa: E402

from openfill.config import Settings  # noqa: E402


class InlinePanelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = Path(tmp.name)
        self.qwen, self.sdxl = "qwen-model", "sdxl-model"
        self.qwen_name, self.sdxl_name = "Qwen Lightning · Local ComfyUI", "SDXL · Local ComfyUI"
        self.service = service = Mock()
        service.app.settings = Settings(remote_fibo=True, active_model=self.qwen)
        from openfill_photocraft.engine_routes import EngineRoutes

        service.routes = EngineRoutes(service.app, self.profile)
        service.app_for.side_effect = service.routes.app_for
        service.configure_engines.side_effect = service.routes.save
        service.app.paths.config = self.profile / "openfill" / "config.json"
        service.app.source_label.return_value = "Local ComfyUI"
        service.engine = "ready"
        service.busy.return_value = False
        service.removal_advice.return_value = {
            "status": "ready",
            "method": "photocraft-texture",
            "displayMethod": "photocraft-texture",
            "key": "test-key",
            "source": "local",
            "tip": "Gradual lighting.",
        }
        service.state.snapshot.return_value = {"state": "idle", "variations": []}
        service.references = ReferenceImages()
        service.reference_rows.side_effect = service.references.rows
        service.add_reference_files.side_effect = service.references.add_files
        watcher = Mock()
        watcher.snapshot.return_value = {
            "connected": True,
            "document": {"name": "Edit.png"},
            "selection": [0, 0, 64, 64],
        }
        rows = [
            {
                "id": model_id,
                "name": name,
                "runnable": True,
                "installed": True,
                "active": model_id == self.qwen,
                "partial": [],
                "remote": True,
                "max_references": 15 if model_id == self.qwen else 0,
                "reference_syntax": "<imageN>" if model_id == self.qwen else "",
            }
            for model_id, name in ((self.qwen, self.qwen_name), (self.sdxl, self.sdxl_name))
        ]
        refresher = Mock()
        refresher.get.return_value = {
            "comfy_ok": True,
            "models": {"fill": [(r["id"], r["name"]) for r in rows]},
            "rows": rows,
            "active": self.qwen,
            "active_name": self.qwen_name,
        }
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
        with (
            patch("tkinter.Tk", side_effect=hidden_root),
            patch("openfill_photocraft.bar.Refresher", return_value=refresher),
            patch("openfill_photocraft.tether.Tether"),
            patch("openfill_photocraft.ui.make_borderless", return_value=False),
            patch.object(Bar, "_tick", lambda self: None),
            patch.object(Bar, "_drain", lambda self: None),
            patch.object(Bar, "_place", lambda self: self.root.withdraw()),
        ):
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

    def test_select_click_uses_image_area_and_never_submits_edit_settings(self):
        bar, view = self.bar, self.bar.select_view_panel
        self.service.layers = []
        bar.tab_buttons["select"].invoke()
        pc = bar.watcher.snapshot.return_value
        pc.update(documentKey="canvas", document={"name": "portrait", "width": 100, "height": 200})
        view._ready("canvas", Image.new("RGB", (80, 160)), "")
        left, top, width, height = view.rect
        self.assertTrue(view.winfo_manager())
        self.assertFalse(bar.options.winfo_manager() or bar.prompt.winfo_manager())
        view._click(SimpleNamespace(x=left - 1, y=top))
        self.service.start.assert_not_called()
        view._operation("add")
        view._click(SimpleNamespace(x=left + width // 2, y=top + height // 4))
        self.service.start.assert_called_once_with(
            "select", {"key": "canvas", "action": "click", "query": "", "operation": "add", "point": [0.5, 0.25]}
        )
        bar.tab_buttons["remove"].invoke()
        self.assertFalse(view.winfo_manager())
        self.assertTrue(bar.options.winfo_manager())
        self.assertEqual(list(bar.tab_buttons), ["select", "fill", "remove"])

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

    def test_disclosure_arrows_keep_the_same_size_in_both_states(self):
        stack = self.bar.sections
        toggle = stack.toggles["edit"]
        header = stack.headers["edit"]
        self.assertEqual(int(header.cget("height")), 32)
        expanded = toggle.cget("image")
        self.assertEqual(int(toggle.tk.call("image", "width", expanded)), 12)
        toggle.invoke()
        collapsed = toggle.cget("image")
        self.assertNotEqual(collapsed, expanded)
        self.assertEqual(int(toggle.tk.call("image", "width", collapsed)), 12)
        self.assertFalse(stack.bodies["edit"].winfo_manager())
        toggle.invoke()
        self.assertEqual(toggle.cget("image"), expanded)
        self.assertTrue(stack.bodies["edit"].winfo_manager())
        self.assertEqual(int(header.cget("height")), 32)

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

    def test_remove_defaults_to_texture_offline_and_preserves_pending_edit_controls(self):
        bar = self.bar
        bar.prompt.set("An edit I am still writing")
        bar.refresher.get.return_value = {"error": "ComfyUI is offline", "comfy_ok": False}
        bar.tab_buttons["remove"].invoke()
        self.assertEqual(bar.model_box.values, ["Local background repair"])
        self.assertFalse(bar.go.disabled)
        self.assertFalse(bar.prompt.winfo_manager())
        self.assertEqual(bar.model_label.get(), "Method")
        with patch.object(bar.settings_view, "generation_options", side_effect=AssertionError("editing controls read")):
            bar.generate()
        kind, request = self.service.start.call_args.args
        self.assertEqual((kind, request["removal"], request["prompt"]), ("fill", "photocraft-texture", ""))
        self.assertTrue(request["remove"])
        self.assertNotIn("useReferences", request)
        bar.tab_buttons["fill"].invoke()
        self.assertEqual(bar.prompt.get(), "An edit I am still writing")
        self.assertTrue(bar.go.disabled)

    def test_remove_only_offers_visible_reconstruction_models_and_keeps_the_edit_model(self):
        bar = self.bar
        facts = bar.refresher.get.return_value
        facts["models"]["remove"] = [("fibo", "FIBO Dev"), ("hidden", "Hidden FIBO")]
        self.service.app.settings.hidden_models = ["hidden"]
        bar.select_tab("remove")
        self.assertEqual(bar.model_box.values, ["Local background repair", "FIBO Dev"])
        bar._model_picked("FIBO Dev")
        bar.generate()
        self.assertEqual(self.service.start.call_args.args[1]["removal"], "fibo")
        self.service.app.set_active.assert_not_called()
        bar.select_tab("fill")
        self.assertEqual(bar._model(), self.qwen)

    def test_scene_suggestion_then_manual_override_uses_the_chosen_model(self):
        bar = self.bar
        facts = bar.refresher.get.return_value
        facts["models"]["remove"] = [
            ("remote-lama-rmbg", "LaMa (Big-LaMa)"),
            ("remote-qwen-image-2.1-lightning-v2-rl", "Qwen Image 2.1 Lightning"),
        ]
        bar.select_tab("remove")
        bar._scene_picked("Wire or repeating pattern")
        self.assertEqual(bar._model(), "remote-lama-rmbg")
        self.assertIn("LaMa", bar.remove_line.cget("text"))
        bar._model_picked("Qwen Image 2.1 Lightning")
        self.assertEqual(bar.remove_scene, "")
        bar.generate()
        self.assertEqual(self.service.start.call_args.args[1]["removal"], "remote-qwen-image-2.1-lightning-v2-rl")
        self.service.app.set_active.assert_not_called()
        bar._scene_picked("Covered lettering")
        self.assertIn("reference", bar.remove_line.cget("text"))
        self.assertEqual(bar._model(), "remote-qwen-image-2.1-lightning-v2-rl")

    def test_auto_advice_updates_method_and_keeps_a_manual_override(self):
        from openfill_photocraft.removal_advice import AUTO

        bar = self.bar
        self.service.removal_advice.return_value = {"status": "sampling", "tip": "Reading colors…"}
        facts = bar.refresher.get.return_value
        facts["models"]["remove"] = [("remote-lama-rmbg", "Big-LaMa")]
        bar.select_tab("remove")
        self.assertTrue(bar.go.disabled)
        self.assertEqual(bar.remove_scene_box.get(), AUTO)
        self.service.removal_advice.return_value = {
            "status": "ready",
            "method": "remote-lama-rmbg",
            "displayMethod": "remote-lama-rmbg",
            "source": "local",
            "key": "correct-selection",
            "tip": "Nearby texture.",
        }
        bar._fill_models()
        self.assertEqual(bar._model(), "remote-lama-rmbg")
        bar.pick_variations(3)
        bar.generate()
        self.assertEqual(self.service.start.call_args.args[1]["adviceKey"], "correct-selection")
        self.assertEqual(self.service.start.call_args.args[1]["variations"], 3)
        bar._model_picked("Big-LaMa")
        bar.pick_variations(5)
        bar.generate()
        self.assertNotIn("adviceKey", self.service.start.call_args.args[1])
        self.assertEqual(self.service.start.call_args.args[1]["variations"], 5)
        bar._model_picked("Local background repair")
        bar._fill_models()
        self.assertEqual(bar._model(), "photocraft-texture")
        bar.generate()
        self.assertNotIn("adviceKey", self.service.start.call_args.args[1])
        bar._scene_picked(AUTO)
        self.assertEqual(bar._model(), "remote-lama-rmbg")

    def test_attach_reorder_and_send_only_to_a_supported_model(self):
        bar = self.bar
        files = [self.profile / "first.png", self.profile / "second.png"]
        for path, color in zip(files, ("red", "blue"), strict=False):
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

    @unittest.skipUnless(sys.platform == "win32", "Windows shell file drops")
    def test_native_windows_file_drop_adds_ordered_unicode_paths(self):
        import ctypes as ct
        from ctypes import wintypes as wt

        class DropFiles(ct.Structure):
            _fields_ = [("pFiles", wt.DWORD), ("pt", wt.POINT), ("fNC", wt.BOOL), ("fWide", wt.BOOL)]

        user = ct.WinDLL("user32", use_last_error=True)
        memory = ct.WinDLL("kernel32", use_last_error=True)
        post = user.PostMessageW
        post.argtypes, post.restype = (wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM), wt.BOOL
        user.GetWindowLongW.argtypes, user.GetWindowLongW.restype = (wt.HWND, ct.c_int), wt.LONG
        memory.GlobalAlloc.argtypes, memory.GlobalAlloc.restype = (wt.UINT, ct.c_size_t), wt.HGLOBAL
        memory.GlobalLock.argtypes, memory.GlobalLock.restype = (wt.HGLOBAL,), ct.c_void_p
        memory.GlobalUnlock.argtypes, memory.GlobalUnlock.restype = (wt.HGLOBAL,), wt.BOOL
        memory.GlobalFree.argtypes, memory.GlobalFree.restype = (wt.HGLOBAL,), wt.HGLOBAL

        bar = self.bar
        target = bar.references.add.winfo_id()
        self.assertEqual(bar.references.add.cget_text(), "Drop images here or click to add")
        self.assertTrue(user.GetWindowLongW(target, -20) & 0x10)  # WS_EX_ACCEPTFILES
        files = [self.profile / "参考 2.png", self.profile / "café 1.png"]
        for path, color in zip(files, ("red", "blue"), strict=False):
            Image.new("RGB", (16, 24), color).save(path)

        def drop(paths):
            header = DropFiles(ct.sizeof(DropFiles), wt.POINT(4, 4), False, True)
            payload = bytes(header) + ("\0".join(map(str, paths)) + "\0\0").encode("utf-16-le")
            handle = memory.GlobalAlloc(2, len(payload))  # GMEM_MOVEABLE
            self.assertTrue(handle)
            address = memory.GlobalLock(handle)
            if not address:
                memory.GlobalFree(handle)
                raise ct.WinError(ct.get_last_error())
            ct.memmove(address, payload, len(payload))
            memory.GlobalUnlock(handle)
            if not post(target, 0x0233, handle, 0):  # WM_DROPFILES; receiver owns the handle after posting
                memory.GlobalFree(handle)
                raise ct.WinError(ct.get_last_error())
            bar.root.update()

        drop(files)
        for _ in range(2):  # native handoff, then the background image decoder's completion
            callback, args = bar.calls.get(timeout=5)
            callback(*args)
        self.assertEqual([row["name"] for row in bar.references.rows], [path.name for path in files])
        self.service.add_reference_files.assert_called_once_with(tuple(map(str, files)))

        bar.model_choice.set(self.sdxl_name)
        bar._fill_models()
        drop(files[:1])
        callback, args = bar.calls.get(timeout=5)
        callback(*args)
        self.service.add_reference_files.assert_called_once()
        self.assertIn("Choose a Qwen or FIBO", bar.references.note.cget("text"))

    def test_generate_and_prompt_enter_submit_latest_controls(self):
        bar = self.bar
        view = bar.settings_view.generation
        bar.prompt.set("Make the object blue.")
        view.fibo_edit_mode.pick("Replace selected content")
        view.auto_feather_percent.set(28)
        # Polling can refresh backend status while the settings are still staged.
        bar._update()
        self.service.start.assert_not_called()
        bar.go.invoke()
        self.assertEqual(
            self.service.start.call_args.args[1]["generation_settings"],
            {
                "fibo_edit_mode": "replace",
                "auto_feather_percent": 28,
            },
        )
        self.assertEqual(bar.settings_view.generation_options(), {})
        view.fibo_boundary_strength.set(72)
        self.service.start.side_effect = RuntimeError("Connection check in progress")
        bar.prompt.on_submit()  # PromptBox binds both Return keys to this callback.
        self.assertEqual(bar.settings_view.generation_options(), {"fibo_boundary_strength": 72})
        self.service.start.side_effect = None
        bar.prompt.on_submit()
        self.assertEqual(self.service.start.call_args.args[1]["generation_settings"], {"fibo_boundary_strength": 72})
        self.assertEqual(bar.settings_view.generation_options(), {})

    def test_pending_results_stay_in_view_and_generate_remains_available(self):
        bar, service = self.bar, self.service
        image = Image.new("RGBA", (8, 12), "red")
        layer = SimpleNamespace(origin=SimpleNamespace(width=8, height=12), rgba=image.tobytes())
        service.layers = [layer]
        st = {
            "state": "done",
            "title": "Generative Fill",
            "kind": "fill",
            "model": "Qwen",
            "elapsed": 6,
            "text": "Done",
            "fraction": 1.0,
            "variations": ["First"],
            "shown": 0,
            "variationSet": 1,
        }
        service.state.snapshot.return_value = st
        bar._update()
        self.assertEqual(len(bar.thumb_cells), 1)
        first = bar.thumb_cells[0]
        self.assertEqual(first.itemcget("selected", "state"), "normal")
        self.assertEqual(bar.thumbs.pack_slaves()[-2:], [bar.thumb_row, bar.keep_button.master])
        self.assertEqual(
            bar.keep_button.master.pack_slaves(), [bar.discard_button, bar.keep_all_button, bar.keep_button]
        )
        self.assertEqual(bar.actions.pack_slaves(), [bar.go])
        self.assertFalse(bar.go.disabled)

        def start(_kind, _params):
            service.busy.return_value = True
            st.update(state="running", elapsed=3, text="Sampling", fraction=0.5, previewSeq=1)

        preview = io.BytesIO()
        image.save(preview, format="PNG")
        service.state.lock, service.state.preview = Lock(), preview.getvalue()
        service.start.side_effect = start
        bar.prompt.set("Make it blue.")
        bar.go.invoke()
        service.start.assert_called_once()
        service.keep.assert_not_called()
        self.assertEqual(bar.actions.pack_slaves(), [bar.cancel_button])
        self.assertIs(bar.thumb_cells[0], first)
        self.assertTrue(bar.keep_button.disabled)
        self.assertEqual(bar.progress.value, 0.5)
        self.assertEqual(bar.job.inner.pack_slaves()[-2:], [bar.preview, bar.thumbs])

        service.layers += [layer] * 5
        service.busy.return_value = False
        service.removal_advice.return_value = {
            "status": "ready",
            "method": "photocraft-texture",
            "displayMethod": "photocraft-texture",
            "key": "test-key",
            "source": "local",
            "tip": "Gradual lighting.",
        }
        st.update(state="done", fraction=1.0, variations=["First", *map(str, range(2, 7))])
        bar._update()
        self.assertEqual(len(bar.thumb_cells), 6)
        self.assertEqual(int(bar.thumb_cells[5].grid_info()["row"]), 1)
        self.assertEqual(bar.thumb_cells[0].itemcget("selected", "state"), "normal")
        self.assertEqual(bar.actions.pack_slaves(), [bar.go])
        self.assertFalse(bar.keep_button.disabled)

        last = bar.thumb_cells[5]
        service.layers.append(layer)
        st["variations"].append("7")
        bar._update()
        self.assertEqual(len(bar.thumb_cells), 7)
        self.assertIs(bar.thumb_cells[5], last)
        st.update(state="cancelled", message="Cancelled", shown=5)
        bar._update()
        self.assertIs(bar.thumb_cells[5], last)
        self.assertEqual(last.itemcget("selected", "state"), "normal")
        self.assertEqual(bar.actions.pack_slaves(), [bar.go])


if __name__ == "__main__":
    unittest.main()
