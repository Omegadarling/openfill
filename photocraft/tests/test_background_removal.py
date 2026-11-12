# SPDX-License-Identifier: Apache-2.0
"""Object removal must use a cleared image and check each reconstruction against it."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.removal_blend import blend_reconstruction  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

from openfill import workflow  # noqa: E402
from openfill.app import App  # noqa: E402
from openfill.backend import (  # noqa: E402
    H3_IMAGE_EDIT_TEMPLATE,
    QWEN21_EDIT_TEMPLATE,
    QWEN21_LIGHTNING_FILE,
    BackendError,
    GenerationResult,
    ModelSpec,
)
from openfill.config import Settings  # noqa: E402
from openfill.geometry import Rect  # noqa: E402


class BackgroundRemovalTests(unittest.TestCase):
    def setUp(self):
        self.app = app = App.__new__(App)
        app.settings = Settings(remote_fibo=True, fibo_edit_mode="preserve", fibo_context_mode="full")
        app.io, app.log, app.server = Mock(), Mock(), Mock()
        app.io.image_rect.return_value = Rect(0, 0, 80, 64)
        app.io.selection_rect.return_value = Rect(22, 20, 16, 18)
        app.io.export_region.return_value = b"original PNG", bytes(80 * 64 * 4)
        self.alpha = bytes([0, 63, 255, 0]) * (80 * 64 // 4)
        app.io.export_selection_mask.return_value = b"selection", self.alpha
        app.io.prepare_removal_mask.side_effect = lambda _size, pixels: pixels
        app.io.removal_guide.return_value = b"clean plate PNG"
        app.io.insert_results.return_value = []
        self.entry = SimpleNamespace(name="FIBO Dev", id="dev", capabilities=("fill", "edit", "remove"))
        model = ModelSpec(arch="fibo2", unet="fibo2_dev_nvfp4.safetensors", clip1="encoder", vae="vae")
        app.resolve_model = Mock(return_value=(self.entry, model))
        app._check_image, app._after_generation, app._debug_dump = Mock(), Mock(), Mock()
        app.paths, app.interactive = Mock(), False
        app.runner_cls = Mock()
        app.runner_cls.return_value.run.side_effect = lambda job: job(Mock(), Mock(), lambda: False)
        app.backend = Mock()
        app.backend.return_value.inpaint.return_value = GenerationResult([], 25)

    def test_rebuild_uses_checked_clean_plate_and_original_soft_mask(self):
        app = self.app
        app.generative_fill(object(), "", "", 1, 1, 25, "dev", remove=True)
        req = app.backend.return_value.inpaint.call_args.args[0]
        self.assertEqual((req.image_png, req.fibo_edit_mode), (b"clean plate PNG", "background"))
        self.assertEqual(app.io.insert_results.call_args.kwargs["mask_pixels"], self.alpha)
        app.io.blend_removal_results.assert_called_once_with(b"clean plate PNG", self.alpha, [])
        self.assertEqual((app.settings.fibo_edit_mode, app.settings.fibo_context_mode), ("preserve", "full"))
        graph = workflow.build_fibo2_edit(req, "plate.png", "mask.png", 25, workflow.Capabilities())
        self.assertEqual(graph["load_image"]["inputs"]["image"], "plate.png")
        self.assertEqual(graph["encode"]["inputs"]["images.image_1"], ["selection_guide", 0])
        self.assertEqual(graph["sampler"]["inputs"]["latent_image"], ["encode", 2])
        self.assertEqual(graph["sampler"]["inputs"]["denoise"], 1.0)
        self.assertNotIn("original_latent", graph)

    def test_rejects_a_large_foreground_and_matches_the_edge_of_a_plausible_repair(self):
        plate = Image.new("RGB", (110, 80), (64, 95, 139))
        box = (38, 20, 73, 62)
        selection = Image.new("L", plate.size)
        selection.paste(255, box)
        proposal = plate.copy()
        proposal.paste((218, 150, 30), box)
        result, fallback = blend_reconstruction(plate, proposal, selection)
        self.assertTrue(fallback)
        self.assertEqual(result.tobytes(), plate.tobytes())
        proposal.paste((83, 107, 147), box)
        result, fallback = blend_reconstruction(plate, proposal, selection)
        self.assertFalse(fallback)
        self.assertEqual(result.getpixel((37, 40)), plate.getpixel((37, 40)))
        self.assertLess(ImageChops.difference(result, plate).getpixel((38, 40))[0], 8)

    def test_neural_removers_use_original_pixels_and_apply_the_prepared_edge_once(self):
        self.app.io.removal_locator.return_value = b"selection highlighted in the original"
        models = [
            ModelSpec(arch="inpaint_model", template="lama_rmbg"),
            ModelSpec(arch="sdxl", template="sdxl_eraser"),
            ModelSpec(arch="qwen_image", template=QWEN21_EDIT_TEMPLATE, unet=QWEN21_LIGHTNING_FILE),
        ]
        for model in models:
            with self.subTest(template=model.template):
                self.app.resolve_model.return_value = (self.entry, model)
                self.app.generative_fill(object(), "", "", 1, 1, 25, "chosen", remove=True, as_picked=True)
                req = self.app.backend.return_value.inpaint.call_args.args[0]
                self.assertEqual(req.image_png, b"original PNG")
                self.assertEqual(
                    req.reference_images, (b"selection highlighted in the original",) if model.qwen21_lightning else ()
                )
                self.assertEqual((req.grow, req.feather), (0, 0))
                inserted = self.app.io.insert_results.call_args
                self.assertEqual(inserted.args[4], 0.0)
                self.assertEqual(inserted.kwargs["mask_pixels"], self.alpha)
                self.app.io.removal_guide.assert_not_called()

    def test_h3_removal_samples_only_from_the_cleared_background(self):
        model = ModelSpec(arch="h3_image", template=H3_IMAGE_EDIT_TEMPLATE, unet="unet", clip1="clip", vae="vae")
        self.app.resolve_model.return_value = (self.entry, model)
        self.app.generative_fill(object(), "", "", 2, 1, 25, "chosen", remove=True, as_picked=True)
        req = self.app.backend.return_value.inpaint.call_args.args[0]
        self.assertEqual(req.image_png, b"original PNG")
        self.assertEqual(req.reference_images, (b"clean plate PNG",))
        self.assertTrue(self.app.io.removal_guide.call_args.kwargs["sample_detail"])
        self.app.io.removal_locator.assert_not_called()
        self.assertIn("<Picture 1>", req.prompt)
        self.assertTrue(req.h3_removal)
        self.app.io.blend_removal_results.assert_called_once_with(b"clean plate PNG", self.alpha, [], h3=True)
        self.assertEqual(self.app.io.insert_results.call_args.kwargs["mask_pixels"], self.alpha)
        graph = workflow.build_fill(
            req, "original.png", "mask.png", 25, workflow.Capabilities(), reference_refs=("background.png",)
        )
        self.assertEqual(graph["encode"]["inputs"]["images.image_1"], ["background", 0])
        self.assertNotIn("images.image_2", graph["encode"]["inputs"])
        self.assertEqual(graph["background"]["inputs"]["image"], "background.png")
        self.assertEqual(graph["background_scale"]["inputs"]["image"], ["background", 0])
        self.assertEqual(graph["background_latent"]["inputs"]["pixels"], ["background_scale", 0])
        self.assertEqual(graph["masked_latent"]["inputs"]["mask"], ["selection_mask", 0])
        self.assertEqual(graph["masked_latent"]["inputs"]["samples"], ["background_latent", 0])
        self.assertEqual(graph["sampler"]["inputs"]["latent_image"], ["masked_latent", 0])
        self.assertEqual(graph["sampler_2"]["inputs"]["latent_image"], ["masked_latent", 0])
        self.assertEqual(graph["sampler_2"]["inputs"]["seed"], 26)
        self.assertEqual(graph["decode"]["inputs"]["samples"], ["latents_2", 0])
        self.assertEqual(graph["sampler"]["inputs"]["denoise"], 0.18)
        self.assertEqual(graph["source_batch"]["inputs"]["image"], ["load_image", 0])
        self.assertEqual(graph["selection_composite"]["inputs"]["mask"], ["selection_support", 0])
        with self.assertRaisesRegex(ValueError, "prepared background"):
            workflow.build_h3_image_edit(req, "original.png", "mask.png", 25, workflow.Capabilities())

    def test_h3_cancels_before_submission_if_the_background_cannot_be_prepared(self):
        model = ModelSpec(arch="h3_image", template=H3_IMAGE_EDIT_TEMPLATE, unet="unet", clip1="clip", vae="vae")
        self.app.resolve_model.return_value = (self.entry, model)
        self.app.io.removal_guide.side_effect = RuntimeError("scratch fill failed")
        with self.assertRaisesRegex(RuntimeError, "scratch fill failed"):
            self.app.generative_fill(object(), "", "", 1, 1, 25, "chosen", remove=True, as_picked=True)
        self.app.backend.assert_not_called()
        self.app.io.insert_results.assert_not_called()

    def test_ineligible_model_or_failed_clean_plate_never_submits_original_image(self):
        self.entry.capabilities = ("fill", "edit")
        with self.assertRaisesRegex(BackendError, "available removal model"):
            self.app.generative_fill(object(), "", "", 1, 1, 25, "turbo", remove=True)
        self.entry.capabilities += ("remove",)
        self.app.io.removal_guide.side_effect = RuntimeError("scratch fill failed")
        with self.assertRaisesRegex(RuntimeError, "scratch fill failed"):
            self.app.generative_fill(object(), "", "", 1, 1, 25, "dev", remove=True)
        self.app.backend.assert_not_called()
        self.app.io.insert_results.assert_not_called()
