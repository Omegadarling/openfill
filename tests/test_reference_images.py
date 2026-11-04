# SPDX-License-Identifier: Apache-2.0
"""Ordered edit-reference transport and FIBO target geometry regressions."""

import unittest
from dataclasses import replace
from unittest.mock import Mock

from openfill import png, workflow
from openfill.backend import GenerationCancelled, InpaintRequest, ModelSpec, WorkflowRejected, edit_reference_options
from openfill.backends.comfy import ComfyBackend


def rgb(color: bytes) -> bytes:
    return png.encode(2, 2, color * 4, 3)


def request(spec, references=(), **options):
    return InpaintRequest(
        rgb(b"\xff\xff\xff"),
        png.encode(2, 2, b"\xff" * 4, 1),
        2,
        2,
        1024,
        1024,
        "Insert the object from image two.",
        "",
        spec,
        reference_images=references,
        **options,
    )


class ReferenceImageTests(unittest.TestCase):
    def setUp(self):
        self.qwen = ModelSpec(
            arch="qwen_image",
            template="qwen21_edit",
            unet="qwen_image_2.1_lightning_v2_rl_bf16.safetensors",
            clip1="qwen3vl_8b_fp8_scaled.safetensors",
            vae="qwen_image_2.1_vae_bf16.safetensors",
        )
        self.first = rgb(b"\xff\x00\x00")
        self.second = rgb(b"\x00\xff\x00")
        self.client = Mock()
        self.client.upload_image.side_effect = lambda name, data: f"openfill/{name}"
        self.backend = ComfyBackend(client=self.client)
        self.backend.capabilities = Mock(return_value=workflow.Capabilities())
        self.backend._run = Mock(return_value=[b"rendered"])

    def test_comfy_uploads_extra_references_in_order_and_keeps_target_cache_identity(self):
        req = request(self.qwen, (self.first, self.second))
        self.backend.inpaint(req)
        uploads = [item.args for item in self.client.upload_image.call_args_list]
        self.assertEqual(len(uploads), 4)
        self.assertEqual(uploads[2][1], workflow.probe_safe_png(self.first))
        self.assertEqual(uploads[3][1], workflow.probe_safe_png(self.second))
        graph = self.backend._run.call_args.args[0]
        self.assertEqual(graph["encode"]["inputs"]["images.image_1"], ["load_image", 0])
        for index, (filename, _data) in enumerate(uploads[2:], start=2):
            node_id = f"reference_{index}"
            self.assertEqual(graph["encode"]["inputs"][f"images.image_{index}"], [node_id, 0])
            self.assertEqual(graph[node_id]["inputs"]["image"], f"openfill/{filename}")
        self.client.upload_image.reset_mock()
        self.backend.inpaint(replace(req, reference_images=(self.second, self.first)))
        reordered = [item.args for item in self.client.upload_image.call_args_list]
        self.assertEqual(reordered, [uploads[0], uploads[1], uploads[3], uploads[2]])
        next_graph = self.backend._run.call_args.args[0]
        self.assertNotEqual(graph["save"]["inputs"]["filename_prefix"], next_graph["save"]["inputs"]["filename_prefix"])

    def test_backend_rejects_unsupported_and_malformed_references_before_upload(self):
        for spec, ref, message in (
            (ModelSpec(arch="sdxl", template="sdxl_edit"), self.first, "does not support reference images"),
            (self.qwen, b"not-a-png", "Image 2 is not an encoded PNG"),
        ):
            with self.subTest(template=spec.template), self.assertRaisesRegex(WorkflowRejected, message):
                self.backend.inpaint(request(spec, (ref,)))
        self.client.upload_image.assert_not_called()
        self.backend._run.assert_not_called()

    def test_cancelled_job_never_uploads_another_reference_or_queues_a_graph(self):
        cancelled = Mock(side_effect=(False, True))
        with self.assertRaises(GenerationCancelled):
            self.backend.inpaint(request(self.qwen, (self.first, self.second)), cancel=cancelled)
        self.assertEqual(self.client.upload_image.call_count, 3)
        self.backend._run.assert_not_called()

    def test_fibo_both_modes_keep_their_target_slot_when_adding_references(self):
        spec = ModelSpec(
            arch="fibo2",
            template="fibo2_edit",
            unet="fibo2_dev_nvfp4.safetensors",
            clip1="qwen3vl_4b_bf16.safetensors",
            vae="flux2-vae.safetensors",
        )
        self.assertEqual(edit_reference_options(spec.effective_template), (15, "<image_N>"))
        for mode, expected_target in (("preserve", "image_scale"), ("replace", "selection_guide")):
            with self.subTest(mode=mode):
                req = request(spec, (self.first,), fibo_edit_mode=mode)
                graph = workflow.build_fill(
                    req, "target.png", "mask.png", 42, workflow.Capabilities(), reference_refs=("extra.png",)
                )
                self.assertEqual(graph["encode"]["inputs"]["images.image_1"], [expected_target, 0])
                self.assertEqual(graph["encode"]["inputs"]["images.image_2"], ["reference_2", 0])
                self.assertEqual(graph["reference_2"]["inputs"]["image"], "extra.png")
                self.assertEqual(
                    graph["sampler"]["inputs"]["latent_image"],
                    ["latent_mask", 0] if mode == "preserve" else ["encode", 2],
                )
                self.assertEqual(graph["selection_composite"]["inputs"]["mask"], ["selection_support", 0])

    def test_builder_does_not_silently_omit_references(self):
        with self.assertRaisesRegex(ValueError, "uploaded reference list does not match"):
            workflow.build_fill(
                request(self.qwen, (self.first,)), "target.png", "mask.png", 42, workflow.Capabilities()
            )


if __name__ == "__main__":
    unittest.main()
