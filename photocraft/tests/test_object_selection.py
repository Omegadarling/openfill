# SPDX-License-Identifier: Apache-2.0
"""Selection correctness when instances overlap, the GPU misses, or the artist switches tabs."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.selection import Selection, key_for  # noqa: E402
from openfill_photocraft.wevisone import WeVisOne, choose_box_matte, choose_matte, encoder  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_grow_remembers_the_exact_radius_after_one_successful_document_edit(self):
        doc = dict(name="Mask", revision=1, width=1600, height=1200, activeLayer=1, hasSelection=True)
        with tempfile.TemporaryDirectory() as directory:
            bridge = Mock(profile=Path(directory))
            bridge.client.call.return_value = {"session": {"active": 0}, "document": doc}
            select = Selection(bridge)
            self.assertEqual(select.grow_pixels, 4)
            params = {"key": key_for(0, doc), "action": "grow"}
            for pixels in (0, 501, -1, 5.5, True, "9", None):
                with self.subTest(pixels=pixels), self.assertRaisesRegex(ValueError, "1 to 500"):
                    select.validate({**params, "pixels": pixels})
            for pixels in (1, 73, 500):
                bridge.client.run.reset_mock()
                select.run(None, {**params, "pixels": pixels}, Mock(), Mock(), Mock())
                bridge.client.run.assert_called_once_with("select.modify.expand", radius=pixels)
                self.assertEqual(Selection(bridge).grow_pixels, pixels)
            bridge.client.run.side_effect = RuntimeError("editor unavailable")
            with self.assertRaisesRegex(RuntimeError, "editor unavailable"):
                select.run(None, {**params, "pixels": 27}, Mock(), Mock(), Mock())
            self.assertEqual(select.grow_pixels, 500)
            self.assertEqual(Selection(bridge).grow_pixels, 500)

    def test_grow_uses_default_for_damaged_or_out_of_range_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = Mock(profile=Path(directory))
            settings = bridge.profile / "selection-settings.json"
            for invalid in ("{", "[]", '{"grow_pixels": 501}', '{"grow_pixels": true}'):
                with self.subTest(invalid=invalid):
                    settings.write_text(invalid, encoding="utf-8")
                    self.assertEqual(Selection(bridge).grow_pixels, 4)

    def test_box_selects_small_overlapping_matte_and_preserves_its_tail_and_hole(self):
        behind = Image.new("L", (80, 60))
        behind.paste(255, (15, 10, 57, 44))
        item = Image.new("L", behind.size)
        item.paste(104, (23, 16, 40, 29))
        item.paste(0, (26, 19, 36, 26))  # hole over the other object
        item.paste(71, (38, 22, 45, 24))  # an end outside the user's box
        result = choose_box_matte([behind, item], (21, 14, 42, 30))
        self.assertEqual(result.tobytes(), item.tobytes())
        self.assertIsNone(choose_box_matte([item], (28, 20, 34, 24)))
        self.assertIsNone(choose_box_matte([behind, item], (1, 1, 5, 9)))

    def test_box_crops_original_detail_and_maps_reversed_edge_drag_to_canvas(self):
        expected = Image.new("L", (4096, 200))
        expected.paste(202, (4078, 91, 4096, 102))
        expected.paste(0, (4082, 95, 4088, 98))
        source = expected.convert("RGB")
        model = WeVisOne(Path(tempfile.gettempdir()))
        model._client = Mock()
        frames = []

        def infer(_client, _name, frame, *_args, **_kwargs):
            frames.append(frame.size)
            return [frame.getchannel("R")]

        model._instances = infer
        with patch("openfill_photocraft.wevisone.encoder", return_value="encoder"):
            result = model.mask(Mock(), source, "pin", None, Mock(), lambda: False, box=[1, 0.54, 4077 / 4096, 0.44])
        self.assertEqual(len(frames), 1)
        self.assertLess(frames[0][0], 100)
        self.assertEqual(result.size, source.size)
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_a_matte_clipped_by_the_temporary_crop_retries_on_the_full_image(self):
        source = Image.new("RGB", (100, 80))
        item = Image.new("L", source.size)
        item.paste(173, (44, 35, 57, 49))
        model = WeVisOne(Path(tempfile.gettempdir()))
        model._client = Mock()
        frames = []

        def infer(_client, _name, frame, *_args, **_kwargs):
            frames.append(frame.size)
            if frame.size == source.size:
                return [item]
            # Even a mostly enclosed object is unreliable if the crop edge cuts off its tail.
            clipped = Image.new("L", frame.size)
            clipped.paste(169, (20, 20, 32, 30))
            clipped.paste(169, (0, 25, 25, 26))
            return [clipped]

        model._instances = infer
        with patch("openfill_photocraft.wevisone.encoder", return_value="encoder"):
            result = model.mask(Mock(), source, "pin", None, Mock(), lambda: False, box=[0.4, 0.375, 0.6, 0.625])
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[-1], source.size)
        self.assertEqual(result.tobytes(), item.tobytes())

    def test_box_combines_model_alpha_with_the_document_selection(self):
        doc = dict(
            name="Drawing",
            revision=3,
            width=20,
            height=10,
            activeLayer=1,
            hasSelection=True,
            selectionBounds=[0, 0, 9, 9],
        )
        mask = Image.new("L", (20, 10))
        mask.paste(116, (12, 3, 18, 6))
        for operation in ("add", "subtract"):
            with self.subTest(operation=operation):
                bridge = Mock(profile=Path(tempfile.gettempdir()))
                bridge.client.call.return_value = {"session": {"active": 0}, "document": doc}
                bridge.client.run.return_value = doc
                bridge.session.return_value = {"active": 1}
                bridge._open.return_value = 1, 90
                selector = Selection(bridge)
                selector.key, selector.source = key_for(0, doc), Image.new("RGB", mask.size)
                selector.model.mask = Mock(return_value=mask)
                box = [0.55, 0.2, 1, 0.7]
                request = dict(key=selector.key, action="box", box=box, operation=operation)
                selector.run(Mock(), request, Mock(), lambda: False, Mock())
                self.assertEqual(selector.model.mask.call_args.kwargs["box"], box)
                self.assertEqual(bridge._open.call_args.args[0].getchannel("A").tobytes(), mask.tobytes())
                bridge.client.run.assert_called_with(
                    "select.loadSelection", channel="transparency", document=1, layer=90, operation=operation
                )
                bridge._close.assert_called_once_with(1)
                for bad in ([0, 0, 1, float("nan")], [0, 0, 1.1, 1], [0, 0.4, 1, 0.4], ["1", 0, 0, 1]):
                    with self.assertRaisesRegex(ValueError, "Drag a box"):
                        selector.validate({**request, "box": bad})

    def test_native_drag_clamps_to_image_ignores_letterbox_and_cancels_with_escape(self):
        try:
            from openfill_photocraft.select_view import SelectView
        except ImportError:
            self.skipTest("Tk unavailable")
        view = object.__new__(SelectView)
        view.drag_start, view.dragged = None, False
        view.rect, view.key = (20, 30, 100, 200), "preview"
        view.canvas, view.bar, view.submit = Mock(), Mock(), Mock()
        view.bar.service.busy.return_value = False
        start = SimpleNamespace(x=45, y=80)
        past_edge = SimpleNamespace(x=4, y=280)
        view._press(past_edge)
        view._release(start)
        view._press(start)
        view._drag(past_edge)
        self.assertEqual(view._cancel_drag(), "break")
        view._release(past_edge)
        view.submit.assert_not_called()
        view._press(start)
        view._release(past_edge)
        view.submit.assert_called_once_with("box", [0, 0.25, 0.25, 1])
        view._press(start)
        view._release(start)
        view.submit.assert_called_with("click", [0.25, 0.25])

    def test_hit_uses_mask_not_box_and_preserves_translucency(self):
        back = Image.new("L", (8, 4))
        back.paste(255, (1, 0, 7, 4))
        glass = Image.new("L", back.size)
        glass.paste(94, (2, 1, 5, 3))
        glass.putpixel((2, 1), 0)  # transparent hole in front of the other object
        self.assertEqual(choose_matte([back, glass], (0.4, 0.4)).tobytes(), glass.tobytes())
        self.assertEqual(choose_matte([back, glass], (0.3, 0.4)).tobytes(), back.tobytes())
        with self.assertRaisesRegex(ValueError, "No object found"):
            choose_matte([back, glass], (0.99, 0.99))
        with self.assertRaisesRegex(ValueError, "No matching"):
            choose_matte([Image.new("L", back.size)], None)

    def test_description_selects_disjoint_matches(self):
        a, b = (Image.new("L", (6, 1)) for _ in range(2))
        a.putdata([254, 110, 0, 0, 0, 0])
        b.putdata([0, 4, 0, 0, 90, 254])
        self.assertEqual(list(choose_matte([a, b], None).tobytes()), [255, 110, 0, 0, 90, 255])

    def test_loader_uses_server_spelling_and_rejects_ordinary_qwen(self):
        client = Mock()
        client.object_info.return_value = {"WeVisOneDetect": {}}
        path = r"Text Encoders\WeVisOne_VL_4B_BF16.safetensors"
        client.combo_options.return_value = ["wevisone_vl_2b_bf16.safetensors", path, "qwen3vl_4b_bf16.safetensors"]
        self.assertEqual(encoder(client), path)
        client.combo_options.return_value = ["qwen3vl_4b_bf16.safetensors"]
        with self.assertRaisesRegex(ValueError, "Install"):
            encoder(client)

    def test_inflight_switch_from_one_unselected_tab_to_another_never_commits(self):
        doc = {"name": "Photo.png", "revision": 3, "width": 30, "height": 20, "activeLayer": 1}
        tab = [0]
        bridge = Mock(profile=Path(tempfile.gettempdir()))
        bridge.client.call.side_effect = lambda _: {"session": {"active": tab[0]}, "document": doc}
        selector = Selection(bridge)
        selector.key, selector.source = key_for(tab[0], doc), Image.new("RGB", (30, 20))

        def while_gpu_runs(*_, **_kwargs):
            tab[0] = 1
            return Image.new("L", (30, 20), 255)

        selector.model.mask = while_gpu_runs
        commit = Mock()
        with self.assertRaisesRegex(ValueError, "document or selection changed"):
            selector.run(Mock(), {"key": selector.key, "action": "click", "point": [0.3, 0.4]}, Mock(), Mock(), commit)
        commit.assert_not_called()
        bridge._open.assert_not_called()
        bridge.client.run.assert_not_called()
