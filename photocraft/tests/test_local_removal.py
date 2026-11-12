# SPDX-License-Identifier: Apache-2.0
"""Regressions for large removals on smooth backgrounds and object-selection fringes."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "photocraft"), str(ROOT / "gimp-openfill")]

from openfill_photocraft.local_removal import (  # noqa: E402
    grow_support,
    removal_mask,
    smooth_background,
    texture_donors,
)
from openfill_photocraft.removal_blend import match_h3_lighting  # noqa: E402
from openfill_photocraft.removal_structure import structured_background  # noqa: E402

from openfill.backend import GenerationCancelled  # noqa: E402


class LocalRemovalTests(unittest.TestCase):
    def test_removal_keeps_the_opacity_of_a_wide_feather_and_low_density_selection(self):
        mask = Image.new("L", (240, 140))
        ImageDraw.Draw(mask).rectangle((60, 25, 180, 115), fill=255)
        soft = mask.filter(ImageFilter.GaussianBlur(12))
        prepared = removal_mask(soft)
        self.assertEqual(prepared.getpixel((0, 70)), 0)
        self.assertEqual(prepared.getpixel((120, 70)), 255)
        ramp = [prepared.getpixel((x, 70)) for x in range(15, 90)]
        self.assertGreater(len(set(v for v in ramp if 8 < v < 247)), 24)
        self.assertEqual(ramp, sorted(ramp))
        self.assertLessEqual(removal_mask(soft.point(lambda v: v // 2)).getextrema()[1], 127)

    def test_reconstructs_one_dark_seam_with_matching_ends(self):
        size = (360, 260)
        sky = Image.new("RGBA", size, (190, 192, 197, 255))
        mask = Image.new("L", size)
        ImageDraw.Draw(mask).ellipse((120, 69, 240, 196), fill=255)
        for both_ends in (False, True):
            scene = sky.copy()
            ImageDraw.Draw(scene).line(((0, 130), (359 if both_ends else 175, 130)), fill=(30, 35, 40), width=3)
            ImageDraw.Draw(scene).ellipse((124, 73, 236, 192), fill=(10, 45, 58))
            repair = structured_background(scene, removal_mask(mask), Mock(), lambda: False)
            if not both_ends:
                self.assertIsNone(repair)
                continue
            self.assertIsNotNone(repair)
            self.assertLess(repair.getpixel((180, 130))[0], 65)
            self.assertGreater(repair.getpixel((180, 117))[0], 170)
            self.assertEqual(repair.getpixel((40, 130)), scene.getpixel((40, 130))[:3])

    def test_h3_lighting_retains_rendered_detail_and_exact_exterior(self):
        plate = Image.new("RGB", (160, 120), (170, 176, 182))
        mask = Image.new("L", plate.size)
        ImageDraw.Draw(mask).rectangle((25, 20, 135, 100), fill=255)
        proposal = plate.copy()
        proposal.paste((185, 191, 197), (25, 20, 136, 101))
        ImageDraw.Draw(proposal).line((78, 40, 78, 80), fill=(112, 118, 124), width=2)
        result = match_h3_lighting(plate, proposal, mask)
        self.assertEqual(result.getpixel((10, 60)), plate.getpixel((10, 60)))
        self.assertLessEqual(abs(result.getpixel((50, 60))[0] - 170), 1)
        self.assertGreater(result.getpixel((73, 60))[0] - result.getpixel((78, 60))[0], 45)

    def test_sampling_excludes_unrelated_colors_patch_edges_and_transparency(self):
        sky = (23, 70, 84, 255)
        for color, intruder in ((sky, (240, 243, 244, 255)), ((209, 216, 204, 255), (37, 47, 59, 255))):
            with self.subTest(color=color):
                scene = Image.new("RGBA", (140, 110), color)
                mask = Image.new("L", scene.size)
                ImageDraw.Draw(mask).polygon([(35, 30), (96, 20), (89, 46), (50, 77)], fill=255)
                draw = ImageDraw.Draw(scene)
                draw.rectangle((103, 73, 128, 95), fill=intruder)
                draw.rectangle((4, 80, 30, 97), fill=(*color[:3], 0))
                scene.paste(intruder, mask=mask)
                donors = texture_donors(scene, mask)
                self.assertIsNotNone(donors)
                for point in ((69, 39), (110, 78), (102, 73), (6, 81)):
                    self.assertEqual(donors.getpixel(point), 0)
                self.assertEqual(donors.getpixel((20, 22)), 255)

    def test_multicolored_rim_keeps_both_sides_available(self):
        scene = Image.new("RGBA", (130, 110), (20, 22, 25, 255))
        ImageDraw.Draw(scene).rectangle((65, 0, 130, 110), fill=(225, 177, 145, 255))
        mask = Image.new("L", scene.size)
        ImageDraw.Draw(mask).ellipse((45, 20, 85, 80), fill=255)
        self.assertIsNone(texture_donors(scene, mask))
        self.assertIsNone(texture_donors(scene, Image.new("L", scene.size, 255)))

    def test_expansion_matches_a_maximum_filter_and_keeps_opposite_edges_empty(self):
        mask = Image.new("L", (37, 29))
        ImageDraw.Draw(mask).rectangle((0, 4, 9, 14), fill=255)
        for radius in (1, 2, 5, 8):
            with self.subTest(radius=radius):
                actual = grow_support(mask, radius)
                self.assertEqual(actual.tobytes(), mask.filter(ImageFilter.MaxFilter(2 * radius + 1)).tobytes())
                self.assertEqual(actual.getpixel((36, 4)), 0)
        prepared = removal_mask(mask)
        self.assertEqual(ImageChops.subtract(mask, prepared).getbbox(), None)
        self.assertEqual(prepared.getpixel((36, 4)), 0)
        self.assertTrue(any(0 < value < 255 for value in prepared.getdata()))

    def test_large_foreground_on_a_gradient_has_no_ghost_or_blocky_patch(self):
        size = (128, 128)
        plate = Image.new("RGB", size)
        plate.putdata([(50 + x, 80 + y // 2, 35 + (x + y) // 3) for y in range(128) for x in range(128)])
        source = plate.copy()
        ImageDraw.Draw(source).rounded_rectangle((35, 17, 88, 111), radius=12, fill=(12, 32, 218))
        selection = Image.new("L", size)
        ImageDraw.Draw(selection).rounded_rectangle((35, 17, 88, 111), radius=12, fill=255)
        selected = removal_mask(selection)
        repaired = smooth_background(source.convert("RGBA"), selected, Mock(), lambda: False)
        self.assertIsNotNone(repaired)
        error = ImageStat.Stat(ImageChops.difference(repaired, plate), selection).mean
        self.assertLess(max(error), 2.5)
        output = Image.composite(repaired, source, selected)
        exterior = ImageOps.invert(selected.point(lambda value: 255 if value else 0))
        self.assertEqual(
            Image.composite(ImageChops.difference(output, source), Image.new("RGB", size), exterior).getbbox(), None
        )

    def test_visible_texture_uses_the_texture_path(self):
        source = Image.new("RGBA", (128, 128), "white")
        draw = ImageDraw.Draw(source)
        for x in range(0, 128, 8):
            draw.rectangle((x, 0, x + 3, 127), fill=(50, 80, 130))
        mask = Image.new("L", source.size)
        ImageDraw.Draw(mask).rectangle((40, 35, 85, 95), fill=255)
        self.assertIsNone(smooth_background(source, mask, Mock(), lambda: False))

    def test_cancellation_and_no_background(self):
        source = Image.new("RGBA", (64, 64), (180, 130, 40))
        mask = Image.new("L", source.size)
        ImageDraw.Draw(mask).rectangle((20, 20, 42, 42), fill=255)
        with self.assertRaises(GenerationCancelled):
            smooth_background(source, mask, Mock(), lambda: True)
        self.assertIsNone(smooth_background(source, Image.new("L", source.size, 255), Mock(), lambda: False))
