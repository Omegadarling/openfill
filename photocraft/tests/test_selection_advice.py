# SPDX-License-Identifier: Apache-2.0
"""Selection advice reads, invalidation, availability and managed method aliases."""

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.removal_advice import TEXTURE_REMOVAL, recommend  # noqa: E402
from openfill_photocraft.selection_advice import SelectionSampler, inspect_pixels, live_selection_key  # noqa: E402

from openfill.remote_models import LAMA_ID, QWEN21_LIGHTNING_ID, ROREM_ID  # noqa: E402


class AdviceTests(unittest.TestCase):
    def test_automatic_h3_uses_current_visible_engine_and_preserves_fallback(self):
        from openfill_photocraft.service import Service

        from openfill.remote_models import H3_IMAGE_ID

        service = Mock(layers=[])
        service.busy.return_value = False
        app = service.app_for.return_value
        app.settings.remote_fibo, app.settings.fibo_source, app.settings.hidden_models = True, "studio", []
        service.photocraft.return_value = {"selectionKey": "stroke"}
        advice = Service.removal_advice(service, [LAMA_ID, H3_IMAGE_ID])
        self.assertEqual((advice["method"], advice["source"], advice["key"]), (H3_IMAGE_ID, "studio", "stroke"))
        service.advice_sampler.request.assert_not_called()
        service.photocraft.return_value = {}
        self.assertEqual(Service.removal_advice(service, [H3_IMAGE_ID])["status"], "empty")
        service.advice_sampler.request.return_value = {"status": "sampling"}
        app.settings.hidden_models = [H3_IMAGE_ID]
        self.assertEqual(Service.removal_advice(service, [H3_IMAGE_ID])["status"], "sampling")
        app.settings.hidden_models = []
        self.assertEqual(Service.removal_advice(service, [LAMA_ID])["status"], "sampling")

    def test_gradient_with_an_edge_selection_only_samples_the_canvas(self):
        def gradient(x, y):
            self.assertTrue(0 <= x < 100 and 0 <= y < 80)
            return x / 200, y / 250, 0.5, 1

        f = inspect_pixels((100, 80), (0, 30, 45, 40), gradient)
        self.assertEqual(f["samples"], 21)
        self.assertEqual(recommend(f, [LAMA_ID, ROREM_ID], True)["method"], TEXTURE_REMOVAL)

    def test_low_context_fallback_and_concrete_managed_alias(self):
        f = inspect_pixels((40, 24), (0, 0, 40, 24), lambda x, y: (0, 0, 0, 0))
        self.assertEqual(f["samples"], 0)
        self.assertEqual(recommend(f, [], True)["method"], TEXTURE_REMOVAL)
        advice = recommend(f, ["lama-big", "mat-places512"], False)
        self.assertEqual((advice["method"], advice["displayMethod"]), ("lama-big", "default"))

    def test_removed_model_falls_back_to_a_visible_method(self):
        cues = dict(
            samples=28,
            coverage=0.25,
            aspect=1.1,
            detail=0.14,
            edge=0.23,
            jump=0.35,
            bend=0.10,
            chroma=0.3,
            range=0.4,
            highlights=0,
            cool=0,
        )
        self.assertEqual(recommend(cues, [LAMA_ID, QWEN21_LIGHTNING_ID], True)["method"], QWEN21_LIGHTNING_ID)
        self.assertEqual(recommend(cues, [LAMA_ID], True)["method"], LAMA_ID)

    def test_a_late_pixel_reply_cannot_replace_a_new_selection(self):
        entered, release = threading.Event(), threading.Event()

        class Client:
            revision = 1
            first = True
            reads = 0

            def call(self, name):
                self_name = "OpenFill correction.png"
                assert name == "ui.inspect"
                return {
                    "session": {"active": 1},
                    "document": {
                        "name": self_name,
                        "revision": self.revision,
                        "width": 64,
                        "height": 64,
                        "activeLayer": 7,
                        "hasSelection": True,
                        "selectionBounds": [8, 8, 20, 20],
                    },
                }

            def run(self, name, **kwargs):
                assert name == "document.sampleColor" and kwargs["sampleLayer"] == "all"
                self.reads += 1
                if self.first:
                    self.first = False
                    entered.set()
                    assert release.wait(3)
                return (0.2, 0.4, 0.6, 1)

        c = Client()
        sampler = SelectionSampler(c)
        snap = {
            "document": {"width": 64, "height": 64},
            "selection": [8, 8, 20, 20],
            "selectionKey": live_selection_key(c),
        }
        self.assertEqual(sampler.request(snap)["status"], "sampling")
        self.assertTrue(entered.wait(3))
        c.revision += 1
        updated = dict(snap, selectionKey=live_selection_key(c))
        self.assertNotEqual(updated["selectionKey"], snap["selectionKey"])
        self.assertEqual(sampler.request(updated)["status"], "sampling")
        release.set()
        deadline = time.monotonic() + 4
        while (answer := sampler.request(updated))["status"] == "sampling":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual((answer["status"], answer["key"]), ("ready", updated["selectionKey"]))
        reads = c.reads
        self.assertEqual(sampler.request(updated), answer)
        self.assertEqual(c.reads, reads)
        self.assertEqual(sampler.request({})["status"], "empty")
