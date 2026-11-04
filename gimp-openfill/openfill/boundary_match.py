"""Conservative color matching at the inside edge of a composited selection.

The graph already protects the exterior, so there is no generated exterior
left to measure. Sample a narrow inner ring instead, reject large/variable
differences as content, and fade the correction before reaching the interior.
This adjusts color drift; it cannot align displaced objects or contours.
"""
from __future__ import annotations

import math

from . import quality


def match_boundary(width: int, height: int, original: bytes, generated: bytes,
                   mask: bytes, strength: float = 0.5) -> bytes:
    """Return RGBA with small edge color offsets corrected; alpha/exterior stay exact."""
    if width < 1 or height < 1 or len(mask) != width * height:
        raise ValueError("Boundary mask dimensions do not match")
    if len(original) != width * height * 4 or len(generated) != len(original):
        raise ValueError("Boundary matching requires matching RGBA buffers")
    amount = float(strength)
    if not math.isfinite(amount):
        raise ValueError("Boundary strength must be finite")
    amount = min(1.0, max(0.0, amount))
    if not amount or not any(mask) or 0 not in mask:
        return generated
    # Inverting binary support makes seam_field's exterior sampling ring lie
    # INSIDE the actual selection. Auto-feather alpha never weights RGB twice.
    inverse = bytes(0 if value else 255 for value in mask)
    selected = sum(value != 0 for value in mask)
    band = max(8, min(48, round(math.sqrt(selected) * 0.04)))
    field = quality.seam_field(width, height, original, generated, inverse, 4, band=band)
    if field is None:
        return generated
    corrected = quality.apply_seam_field(generated, width, height, 4, field)
    # Distance at bounded analysis resolution is enough for a low-frequency
    # color ramp and avoids several full-canvas Python integer arrays.
    step = max(1, math.ceil(max(width, height) / quality.SEAM_SIDE))
    sw, sh = math.ceil(width / step), math.ceil(height / step)
    small = b"".join(inverse[y * width:(y + 1) * width:step] for y in range(0, height, step))
    distance = quality.signed_distance(sw, sh, small)
    output = bytearray(generated)
    reach = 2 * band
    for y in range(height):
        row = (y // step) * sw
        for x in range(width):
            pixel = y * width + x
            if not mask[pixel]:
                continue
            d = max(0, distance[row + x // step] - 1) * step
            if d >= reach:
                continue
            weight = amount * 0.5 * (1 + math.cos(math.pi * d / reach))
            start = 4 * pixel
            for c in range(3):
                value = generated[start + c]
                output[start + c] = round(value + weight * (corrected[start + c] - value))
    return bytes(output)
