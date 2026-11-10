# SPDX-License-Identifier: Apache-2.0
"""Selection correctness when instances overlap, the GPU misses, or the artist switches tabs."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.selection import Selection, key_for  # noqa: E402
from openfill_photocraft.wevisone import choose_matte, encoder  # noqa: E402


class SelectionTests(unittest.TestCase):
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

        def while_gpu_runs(*_):
            tab[0] = 1
            return Image.new("L", (30, 20), 255)

        selector.model.mask = while_gpu_runs
        commit = Mock()
        with self.assertRaisesRegex(ValueError, "document or selection changed"):
            selector.run(Mock(), {"key": selector.key, "action": "click", "point": [0.3, 0.4]}, Mock(), Mock(), commit)
        commit.assert_not_called()
        bridge._open.assert_not_called()
        bridge.client.run.assert_not_called()
