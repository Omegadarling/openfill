# SPDX-License-Identifier: Apache-2.0
"""Local background completion for smooth surfaces, using visible pixels as boundary values."""

from __future__ import annotations

import math
from array import array

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from openfill.backend import GenerationCancelled


def grow_support(mask: Image.Image, radius: int) -> Image.Image:
    """Dilate without wrapped edges, using a logarithmic number of pixel shifts."""
    result = mask.point(lambda value: 255 if value else 0)
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
    """Include a narrow, softly blended margin so object-selection fringes disappear."""
    area = sum(index * count for index, count in enumerate(selection.histogram())) / 255
    radius = max(2, min(16, round(math.sqrt(area) * 0.006)))
    support = grow_support(selection, radius)
    soft = support.filter(ImageFilter.GaussianBlur(max(0.5, radius / 3)))
    return ImageChops.lighter(selection, ImageChops.multiply(soft, support))


def smooth_background(source: Image.Image, selection: Image.Image, progress, cancelled) -> Image.Image | None:
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
    counts = detail.histogram(mask=known)
    total = sum(counts)
    # A strict tail limit keeps backgrounds with visible texture and edges on
    # the texture-synthesis path.
    if sum(counts[3:]) > total * 0.05 or sum(counts[6:]) > total * 0.01:
        return None

    # Classification uses a wider guard. Interpolation follows the actual
    # boundary so the result meets the neighboring background without a seam.
    unknown = ImageChops.lighter(mask.point(lambda value: 255 if value else 0), ImageOps.invert(opaque))
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
                        delta = (field[left] + field[right] + field[top] + field[bottom]) * 0.25 - field[index]
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
