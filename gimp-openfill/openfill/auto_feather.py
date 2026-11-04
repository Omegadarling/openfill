"""Area-scaled inward selection feathering, independent of the image editor.

The percentage selects an outer band containing that share of the selection's
alpha-weighted area. A 3/4 chamfer distance approximates Euclidean distance to
unselected pixels, including holes. Ring quantization means coverage is accurate
to one distance bin rather than an exact fractional-pixel area. A cosine ramp
softens that band without growing the selection or multiplying alpha twice.
"""
from __future__ import annotations

from array import array
import math


def feather_selection(
    mask: bytes,
    width: int,
    height: int,
    percent: float,
    *,
    canvas_edges: tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> bytes:
    """Return an 8-bit mask with its outer ``percent`` area smoothly softened.

    ``canvas_edges`` is left, top, right, bottom. True means that crop edge is
    a canvas boundary; no imaginary unselected pixels are introduced there.
    With all four flags true, Select All consequently stays fully opaque.
    Existing soft alpha is an upper bound on the result, and zero stays zero.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ValueError("Selection mask dimensions must be positive integers")
    if len(mask) != width * height:
        raise ValueError("Selection mask byte count does not match its dimensions")
    if len(canvas_edges) != 4:
        raise ValueError("Canvas edges must contain left, top, right and bottom flags")
    amount = float(percent)
    if not math.isfinite(amount):
        raise ValueError("Auto-feather percentage must be finite")
    amount = max(0.0, min(100.0, amount))
    if amount == 0.0 or not any(mask):
        return mask
    left, top, right, bottom = map(bool, canvas_edges)
    # There is no visible edge to soften when the complete canvas is selected.
    if left and top and right and bottom and 0 not in mask:
        return mask

    stride = width + 2
    infinity = 3 * (width + height + 2) + 4
    kind = "H" if infinity <= 65535 else "I"
    distance = array(kind, [infinity]) * (stride * (height + 2))
    if not top:
        distance[:stride] = array(kind, [0]) * stride
    if not bottom:
        distance[-stride:] = array(kind, [0]) * stride
    if not left:
        for y in range(height + 2):
            distance[y * stride] = 0
    if not right:
        for y in range(height + 2):
            distance[y * stride + stride - 1] = 0

    # Forward and reverse passes keep storage compact (normally two bytes per
    # pixel) and avoid repeated morphological erosion of a potentially large band.
    for y in range(height):
        start = (y + 1) * stride + 1
        pixel = y * width
        for index in range(start, start + width):
            if mask[pixel] == 0:
                distance[index] = 0
            else:
                value = distance[index - 1] + 3
                candidate = distance[index - stride] + 3
                if candidate < value:
                    value = candidate
                candidate = distance[index - stride - 1] + 4
                if candidate < value:
                    value = candidate
                candidate = distance[index - stride + 1] + 4
                if candidate < value:
                    value = candidate
                if value < infinity:
                    distance[index] = value
            pixel += 1

    histogram: dict[int, int] = {}
    total = 0
    for y in range(height - 1, -1, -1):
        start = (y + 1) * stride + width
        pixel = (y + 1) * width - 1
        for index in range(start, start - width, -1):
            alpha = mask[pixel]
            if alpha:
                value = distance[index]
                candidate = distance[index + 1] + 3
                if candidate < value:
                    value = candidate
                candidate = distance[index + stride] + 3
                if candidate < value:
                    value = candidate
                candidate = distance[index + stride - 1] + 4
                if candidate < value:
                    value = candidate
                candidate = distance[index + stride + 1] + 4
                if candidate < value:
                    value = candidate
                distance[index] = value
                histogram[value] = histogram.get(value, 0) + alpha
                total += alpha
            pixel -= 1

    # A zero-free mask whose real canvas edges are all protected has no finite
    # edge. The early return above normally catches it, but retain this guard.
    if not total or min(histogram) >= infinity:
        return mask
    target = total * amount / 100.0
    accumulated = 0
    cutoff = 0
    for cutoff in sorted(histogram):
        accumulated += histogram[cutoff]
        if accumulated >= target:
            break
    # Include the entire quantile ring. A one-pixel extension keeps even a
    # one-pixel/one-alpha selection nonempty at 100%, with ordinary rounding.
    radius = cutoff + 3.0
    factors = {value: 0.5 - 0.5 * math.cos(math.pi * value / radius)
               for value in histogram if value <= cutoff}
    output = bytearray(mask)
    for y in range(height):
        start = (y + 1) * stride + 1
        pixel = y * width
        for index in range(start, start + width):
            alpha = mask[pixel]
            if alpha:
                value = distance[index]
                if value <= cutoff:
                    output[pixel] = int(alpha * factors[value] + 0.5)
            pixel += 1
    return bytes(output)
