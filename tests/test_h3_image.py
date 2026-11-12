# SPDX-License-Identifier: Apache-2.0
"""Native H3 discovery and selection-safe graph contract."""

import unittest
from unittest.mock import Mock

from openfill import remote_models, workflow
from openfill.backend import InpaintRequest, ModelSpec, edit_reference_options


class H3ImageTests(unittest.TestCase):
    def client(self, encoder):
        files = {
            "UNETLoader": {"unet_name": ["folder/minimax_h3_image_pruned_int8_convrot.safetensors"]},
            "CLIPLoader": {"clip_name": encoder if isinstance(encoder, list) else [encoder], "type": ["minimax"]},
            "VAELoader": {"vae_name": ["folder/minimax_h3_image_vae_fp16.safetensors"]},
        }

        def info(node):
            if node == "MiniMaxH3ImageConditioning":
                return {node: {"input": {}}}
            return {node: {"input": {"required": {k: [v] for k, v in files.get(node, {}).items()}}}}

        client = Mock()
        client.object_info.side_effect = info
        return client

    def test_discovery_selects_exact_server_path_and_supported_encoder_variants(self):
        for encoder in (
            "h3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "h3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        ):
            with self.subTest(encoder=encoder):
                (entry,) = remote_models._discover_h3_image(self.client(encoder), "host")
                self.assertEqual(entry.id, remote_models.H3_IMAGE_ID)
                self.assertEqual(entry.capabilities, ("fill", "edit", "remove"))
                self.assertEqual(next(f.dest for f in entry.files if f.role == "clip1"), "text_encoders/" + encoder)
        self.assertEqual(remote_models._discover_h3_image(self.client("unrelated.safetensors"), "host"), [])
        both = ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"]
        (entry,) = remote_models._discover_h3_image(self.client(both), "host")
        self.assertEqual(next(f.filename for f in entry.files if f.role == "clip1"), both[1])

    def test_extreme_dimensions_stay_within_node_limits(self):
        self.assertEqual(workflow.h3_image_size(650, 10)[0], 16384)
        with self.assertRaises(ValueError):
            workflow.h3_image_size(5130, 10)

    def test_edit_graph_aligns_and_restores_original_with_exact_selection_support(self):
        spec = ModelSpec(arch="h3_image", template="h3_image_edit", unet="unet", clip1="clip", vae="vae")
        req = InpaintRequest(
            b"img", b"mask", 800, 600, 800, 600, "Change the shirt", "", spec, reference_images=(b"ref",)
        )
        g = workflow.build_fill(
            req, "target.png", "mask.png", 42, workflow.Capabilities(), reference_refs=("reference.png",)
        )
        self.assertEqual(edit_reference_options(spec.template), (4, "<Picture N>"))
        self.assertEqual(
            (g["encode"]["class_type"], g["encode"]["inputs"]["images.image_1"]),
            ("MiniMaxH3ImageConditioning", ["load_image", 0]),
        )
        self.assertEqual(g["encode"]["inputs"]["images.image_2"], ["reference_2", 0])
        self.assertEqual((g["sampler"]["inputs"]["steps"], g["sampler"]["inputs"]["cfg"]), (6, 1.0))
        self.assertEqual(g["selection_composite"]["inputs"]["destination"], ["load_image", 0])
        self.assertEqual(g["selection_composite"]["inputs"]["source"], ["out_scale", 0])
        self.assertEqual(g["selection_composite"]["inputs"]["mask"], ["selection_support", 0])
        self.assertEqual(g["save"]["inputs"]["images"], ["selection_composite", 0])
        self.assertEqual((g["out_scale"]["inputs"]["width"], g["out_scale"]["inputs"]["height"]), (800, 600))


if __name__ == "__main__":
    unittest.main()
