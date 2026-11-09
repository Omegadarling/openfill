# SPDX-License-Identifier: Apache-2.0
"""Removal choices must change visible pixels without moving the artist's mask."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.removal_variations import distinct_repairs, render_views  # noqa: E402

from openfill.backend import GenerationCancelled, GenerationResult, ModelSpec  # noqa: E402
from openfill.geometry import Rect, Tile  # noqa: E402
from openfill.workflow import PlannedInpaintRequest  # noqa: E402


def encoded(image, level=1):
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=level)
    return output.getvalue()


class RemovalVariationTests(unittest.TestCase):
    def fixture(self):
        source = Image.frombytes("L", (11, 7), bytes(range(77)))
        mask = Image.new("L", source.size)
        mask.putpixel((3, 4), 255)
        tile = Tile(Rect(2, 1, 5, 5), 40, 48, (Rect(1, 2, 3, 2),))
        request = PlannedInpaintRequest(
            encoded(source), encoded(mask), 11, 7, 88, 56, "", "", ModelSpec(), tiles=(tile,), refine_tiles=(tile,)
        )
        return source, mask, request

    def test_every_view_aligns_image_mask_tiles_and_restored_output(self):
        source, mask, request = self.fixture()
        reports = []

        def inpaint(req, report, preview, cancel):
            with Image.open(io.BytesIO(req.image_png)) as im, Image.open(io.BytesIO(req.mask_png)) as m:
                self.assertEqual(im.size, (req.width, req.height))
                self.assertEqual(im.getpixel(m.getbbox()[:2]), source.getpixel((3, 4)))
                for tile in (*req.tiles, *req.refine_tiles):
                    r = tile.rect
                    crop = im.crop((r.x, r.y, r.right, r.bottom))
                    self.assertEqual(sorted(crop.getdata()), sorted(source.crop((2, 1, 7, 6)).getdata()))
                    d = tile.done[0]
                    self.assertEqual(
                        sorted(crop.crop((d.x, d.y, d.right, d.bottom)).getdata()),
                        sorted(source.crop((3, 3, 6, 5)).getdata()),
                    )
                # Put a mark at the selected pixel. It must land in the same place in all five results.
                output = im.copy()
                output.putpixel(m.getbbox()[:2], 253)
                report(1, "Done")
                return GenerationResult([encoded(output)], 123)

        result = render_views(request, 5, inpaint, lambda frac, _: reports.append(frac), Mock(), lambda: False)
        source.putpixel(mask.getbbox()[:2], 253)
        self.assertEqual(len(result.images), 5)
        for data in result.images:
            with Image.open(io.BytesIO(data)) as restored:
                self.assertEqual((restored.size, restored.tobytes()), (source.size, source.tobytes()))
        self.assertEqual((reports[0], reports[-1]), (0, 1))
        self.assertEqual(reports, sorted(reports))

    def test_cancelling_between_views_stops_the_next_job(self):
        _, _, request = self.fixture()
        inpaint = Mock(return_value=GenerationResult([request.image_png], 4))
        with self.assertRaises(GenerationCancelled):
            render_views(request, 5, inpaint, Mock(), Mock(), lambda: inpaint.called)
        self.assertEqual(inpaint.call_count, 1)

    def test_distinct_compares_the_composite_under_the_mask(self):
        original = Image.new("RGBA", (8, 6), "white")
        mask = Image.new("L", original.size)
        mask.putpixel((3, 2), 128)
        first = original.copy()
        first.putpixel((3, 2), (10, 90, 150, 255))
        second = first.copy()
        second.putpixel((0, 0), (0, 0, 0, 255))
        third = first.copy()
        third.putpixel((3, 2), (200, 20, 30, 255))
        images = [encoded(first), encoded(second, 8), encoded(third)]
        self.assertEqual(distinct_repairs(images, original, mask), [images[0], images[2]])
