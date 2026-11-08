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
from openfill.backend import BackendError, GenerationResult, ModelSpec  # noqa: E402
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
        self.assertEqual(
            (req.image_png, req.mask_png, req.fibo_edit_mode), (b"clean plate PNG", b"selection", "background")
        )
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

    def test_ineligible_model_or_failed_clean_plate_never_submits_original_image(self):
        self.entry.capabilities = ("fill", "edit")
        with self.assertRaisesRegex(BackendError, "FIBO Dev"):
            self.app.generative_fill(object(), "", "", 1, 1, 25, "turbo", remove=True)
        self.entry.capabilities += ("remove",)
        self.app.io.removal_guide.side_effect = RuntimeError("scratch fill failed")
        with self.assertRaisesRegex(RuntimeError, "scratch fill failed"):
            self.app.generative_fill(object(), "", "", 1, 1, 25, "dev", remove=True)
        self.app.backend.assert_not_called()
        self.app.io.insert_results.assert_not_called()
