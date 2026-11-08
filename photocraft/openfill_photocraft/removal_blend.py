# SPDX-License-Identifier: Apache-2.0
"""Blend a proposed reconstruction against a clean plate using only visible background."""

from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter, ImageOps


def _blur_side(image, support, radius):
    # Premultiplication keeps pixels on the opposite side of the seam out of the average.
    side = image.convert("RGBA")
    side.putalpha(support)
    return side.convert("RGBa").filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")


def blend_reconstruction(plate, proposal, selection):
    """Return an RGB result and whether the clean plate replaced a conflicting proposal."""
    if plate.size != proposal.size or selection.size != plate.size:
        raise ValueError("The background reconstruction has unexpected dimensions.")
    size = plate.size
    clean = plate.convert("RGB")
    clean.thumbnail((512, 512), Image.Resampling.LANCZOS)
    generated = proposal.convert("RGB").resize(clean.size, Image.Resampling.LANCZOS)
    support = selection.point(lambda v: 255 if v else 0)
    mask = support.resize(clean.size, Image.Resampling.BOX).point(lambda v: 255 if v else 0)
    exterior = ImageOps.invert(mask)
    channels = ImageChops.difference(
        clean.filter(ImageFilter.GaussianBlur(2)), generated.filter(ImageFilter.GaussianBlur(2))
    ).split()
    deviation = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    counts = deviation.histogram(mask=mask)
    # A gray-hole edit can paint a new subject. Widespread, large color disagreement
    # with the sampled background is grounds to keep the already-cleared plate.
    if sum(counts[65:]) > 0.12 * sum(counts):
        return plate.copy(), True

    bounds = mask.getbbox()
    if not bounds:
        return plate.copy(), False
    short_side = min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    corrected = generated
    # Correct broad lighting drift first, then the narrow seam. Only the correction
    # field is resized to canvas resolution, preserving the model's fine detail.
    radii = sorted({max(3, min(40, round(short_side * fraction))) for fraction in (0.18, 0.06, 0.02)}, reverse=True)
    for radius in radii:
        inside = _blur_side(corrected, mask, radius).convert("RGB")
        outside = _blur_side(clean, exterior, radius)
        amount = outside.getchannel("A").point(lambda a: max(0, min(255, (a - 10) * 4)))
        donor = outside.convert("RGB")
        brighter = ImageChops.subtract(donor, inside)
        darker = ImageChops.subtract(inside, donor)
        adjusted = ImageChops.subtract(ImageChops.add(corrected, brighter), darker)
        corrected = Image.composite(adjusted, corrected, ImageChops.multiply(amount, mask))

    native = proposal.convert("RGB")
    brighten = ImageChops.subtract(corrected, generated).resize(size, Image.Resampling.BILINEAR)
    darken = ImageChops.subtract(generated, corrected).resize(size, Image.Resampling.BILINEAR)
    matched = ImageChops.subtract(ImageChops.add(native, brighten), darken)
    # The editor applies the original soft selection exactly once on its new layer.
    return Image.composite(matched, plate.convert("RGB"), support), False
