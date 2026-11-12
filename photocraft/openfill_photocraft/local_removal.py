# SPDX-License-Identifier: Apache-2.0
"""Local background completion for smooth surfaces, using visible pixels as boundary values."""

from __future__ import annotations

import math
from array import array

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from openfill.backend import GenerationCancelled


def grow_support(mask: Image.Image, radius: int) -> Image.Image:
    """Dilate without wrapped edges, using a logarithmic number of pixel shifts."""
    return _dilate(mask.point(lambda value: 255 if value else 0), radius)


def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    result = mask
    for axis in (0, 1):
        covered = 0
        while covered < radius:
            step = min(covered + 1, radius - covered)
            previous = result
            for direction in (-1, 1):
                shifted = Image.new("L", mask.size)
                offset = [0, 0]
                offset[axis] = direction * step
                shifted.paste(previous, tuple(offset))
                result = ImageChops.lighter(result, shifted)
            covered += step
    return result


def removal_mask(selection: Image.Image) -> Image.Image:
    """Clear the object edge and carry the artist's grayscale feather into the result layer."""
    area = sum(index * count for index, count in enumerate(selection.histogram())) / 255
    reach = max(2, min(32, round(math.sqrt(area) * 0.045)))
    # Put the blend on the background side. The original grayscale ramp is
    # dilated at its own opacity, so feathered tails stay partially transparent.
    expanded = _dilate(selection, round(reach * 1.5))
    soft = expanded.filter(ImageFilter.GaussianBlur(max(1.0, reach * 0.48)))
    return ImageChops.lighter(selection, soft)


def texture_donors(source: Image.Image, support: Image.Image) -> Image.Image | None:
    """For a narrow range of background colors, exclude unrelated objects from the sampling channel."""
    size = source.size
    sample = source.copy()
    sample.thumbnail((448, 448), Image.Resampling.BOX)
    hole = support.resize(sample.size, Image.Resampling.BOX).point(lambda value: 255 if value else 0)
    known = ImageChops.subtract(sample.getchannel("A").point(lambda a: 255 if a == 255 else 0), hole)
    radius = max(4, min(18, round(math.sqrt(hole.histogram()[255]) * 0.08)))
    rim = ImageChops.darker(grow_support(hole, radius), known)
    count = rim.histogram()[255]
    if count < 64:
        return None

    # The rim follows the actual lasso. Its robust range tolerates isolated outlines
    # and stars. A multicolored boundary needs the full context for continuing edges.
    tables = []
    for band in sample.convert("RGB").split():
        histogram = band.histogram(mask=rim)
        low = high = 0
        total = 0
        for value, frequency in enumerate(histogram):
            if total < count * 0.05:
                low = value
            total += frequency
            if total >= count * 0.95:
                high = value
                break
        spread = high - low
        if spread > 48:
            return None
        slack = max(12, round(spread * 0.75))
        tables.append([255 if low - slack <= value <= high + slack else 0 for value in range(256)])

    def matching(image):
        red, green, blue = (band.point(table) for band, table in zip(image.convert("RGB").split(), tables, strict=True))
        return ImageChops.darker(ImageChops.darker(red, green), blue)

    neighbors = known.histogram()[255]
    accepted = ImageChops.darker(matching(sample), known).histogram()[255]
    if neighbors - accepted < max(24, neighbors * 0.005) or accepted < max(128, neighbors * 0.18):
        return None
    # Reject patch centers close to a disallowed color too. A center on the sky
    # right next to a person can still copy the person's sleeve into the repair.
    opaque = source.getchannel("A").point(lambda a: 255 if a == 255 else 0)
    allowed = ImageChops.darker(matching(source), ImageChops.subtract(opaque, support))
    guard = max(3, min(12, round(min(size) / 100)))
    donors = ImageOps.invert(grow_support(ImageOps.invert(allowed), guard))
    if donors.histogram()[255] < max(64, support.histogram()[255] * 0.12):
        return None
    return donors


def smooth_background(
    source: Image.Image, selection: Image.Image, progress, cancelled, *, horizontal: float = 0.5, boundary_only=False
) -> Image.Image | None:
    """Use harmonic interpolation when the surrounding image has very little texture."""
    rgb = source.convert("RGB")
    analysis = rgb.copy()
    analysis.thumbnail((256, 256), Image.Resampling.BOX)
    # Exclude every analysis pixel touched by the hole, plus a guard against
    # downsampling foreground colors into adjacent background samples.
    mask = selection.resize(analysis.size, Image.Resampling.BOX)
    support = grow_support(mask, 2)
    known = ImageOps.invert(support)
    opaque = (
        source.getchannel("A")
        .resize(analysis.size, Image.Resampling.BOX)
        .point(lambda value: 255 if value == 255 else 0)
    )
    known = ImageChops.multiply(known, opaque)
    if known.histogram()[255] < max(64, round(analysis.width * analysis.height * 0.12)):
        return None
    detail = ImageChops.difference(analysis, analysis.filter(ImageFilter.GaussianBlur(1.4)))
    red, green, blue = detail.split()
    detail = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    samples = known
    if boundary_only:
        # A verified line crosses the hole. Judge the texture of its two local
        # surfaces without counting other objects farther away in the crop.
        samples = ImageChops.darker(known, grow_support(support, 12))
    counts = detail.histogram(mask=samples)
    total = sum(counts)
    # A strict tail limit keeps backgrounds with visible texture and edges on
    # the texture-synthesis path.
    if not total or sum(counts[3:]) > total * (0.12 if boundary_only else 0.05) or sum(counts[6:]) > total * (0.07 if boundary_only else 0.01):
        return None

    # Classification uses a wider guard. Interpolation follows the actual
    # boundary so the result meets the neighboring background without a seam.
    unknown = ImageChops.lighter(mask.point(lambda value: 255 if value else 0), ImageOps.invert(opaque))
    across, down = 0.5 * horizontal, 0.5 * (1.0 - horizontal)
    current = None
    for longest in (16, 32, 64, 128, 256):
        size = (max(1, round(analysis.width * longest / 256)), max(1, round(analysis.height * longest / 256)))
        if size[0] > analysis.width or size[1] > analysis.height:
            break
        level = analysis.resize(size, Image.Resampling.BOX)
        hole = unknown.resize(size, Image.Resampling.BOX).point(lambda value: 255 if value else 0)
        outside = ImageOps.invert(hole)
        if not outside.getbbox():
            continue
        if cancelled():
            raise GenerationCancelled("Object removal cancelled.")
        progress(0.1 + 0.7 * longest / 256, "Continuing the smooth background...")
        pixels = list(hole.getdata())
        width, height = size
        neighbors = ([], [])
        for index, selected in enumerate(pixels):
            if selected:
                y, x = divmod(index, width)
                neighbors[(x + y) % 2].append(
                    (
                        index,
                        index - 1 if x else index,
                        index + 1 if x + 1 < width else index,
                        index - width if y else index,
                        index + width if y + 1 < height else index,
                    )
                )
        channels = []
        for channel, samples in enumerate(level.split()):
            values = list(samples.getdata())
            if current is None:
                initial = ImageStat.Stat(samples, outside).mean[0]
                field = [initial] * len(values)
            else:
                field = list(current[channel].resize(size, Image.Resampling.BILINEAR).getdata())
            for index, selected in enumerate(pixels):
                if not selected:
                    field[index] = values[index]
            for iteration in range(150):
                largest_change = 0.0
                for parity in neighbors:
                    for index, left, right, top, bottom in parity:
                        delta = (
                            (field[left] + field[right]) * across + (field[top] + field[bottom]) * down - field[index]
                        )
                        field[index] += 1.6 * delta
                        largest_change = max(largest_change, abs(delta))
                if iteration % 8 == 0 and cancelled():
                    raise GenerationCancelled("Object removal cancelled.")
                if largest_change < 0.02:
                    break
            channels.append(Image.frombytes("F", size, array("f", field).tobytes()))
        current = channels
    if current is None:
        return None
    return Image.merge(
        "RGB", [channel.resize(source.size, Image.Resampling.BICUBIC).convert("L") for channel in current]
    )
