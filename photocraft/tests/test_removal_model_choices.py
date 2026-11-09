# SPDX-License-Identifier: Apache-2.0
"""A selectable removal method must map to this server's nodes and exact loader names."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.removal_advice import removal_scenes  # noqa: E402

from openfill.catalog import model_spec_for  # noqa: E402
from openfill.remote_models import LAMA_ID, _discover_lama, _discover_rorem  # noqa: E402


class RemovalDiscoveryTests(unittest.TestCase):
    def test_incompatible_lama_node_is_not_offered(self):
        client = Mock()
        fields = {"images": ["IMAGE"], "masks": ["MASK"], "removal_strength": ["INT"], "edge_smoothness": ["INT"]}
        info = {"input": {"required": fields}, "output": ["IMAGE"]}
        client.object_info.return_value = {"AILab_LamaRemover": info}
        self.assertEqual([e.id for e in _discover_lama(client, "Test")], [LAMA_ID])
        fields["weights"] = ["MODEL"]
        self.assertEqual(_discover_lama(client, "Test"), [])

    def test_rorem_requires_all_weights_and_keeps_server_spelling(self):
        raw = r"Models\RORem-unet.safetensors"
        fields = {
            "UNETLoader": {"unet_name": [[raw]]},
            "DualCLIPLoader": {
                "clip_name1": [["clip_l.safetensors"]],
                "clip_name2": [["RORem-clip-g.safetensors"]],
                "type": [["sdxl"]],
            },
            "VAELoader": {"vae_name": [["RORem-sdxl-vae.safetensors"]]},
            "InpaintModelConditioning": {},
        }
        client = Mock()
        client.object_info.side_effect = lambda node: {node: {"input": {"required": fields[node]}}}
        [entry] = _discover_rorem(client, "Test")
        self.assertEqual((model_spec_for(entry).unet, entry.capabilities), (raw, ("remove",)))
        fields["VAELoader"]["vae_name"] = [["a-different-vae.safetensors"]]
        self.assertEqual(_discover_rorem(client, "Test"), [])

    def test_stock_suggestion_points_at_default_menu_alias(self):
        records = {r["scene"]: r["method"] for r in removal_scenes(["lama-big"], False)}
        self.assertEqual(records["Wire or repeating pattern"], "default")
        self.assertIsNone(records["Covered lettering"])
