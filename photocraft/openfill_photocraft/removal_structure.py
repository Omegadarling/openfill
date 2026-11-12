# SPDX-License-Identifier: Apache-2.0
"""Continue an isolated narrow seam when matching segments enter a removal from both sides."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .local_removal import grow_support, smooth_background


@dataclass
class _Segment:
    x: float
    y: float
    dx: float
    dy: float
    length: float
    thickness: float


def _segments(peaks):
    width = peaks.width
    remaining = {i for i, value in enumerate(peaks.tobytes()) if value}
    result = []
    while remaining:
        queue = [remaining.pop()]
        points = []
        while queue:
            i = queue.pop()
            y, x = divmod(i, width)
            points.append((x, y))
            for yy in range(max(0, y - 1), min(peaks.height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = yy * width + xx
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
        count = len(points)
        if count < 10:
            continue
        x = sum(px for px, _ in points) / count
        y = sum(py for _, py in points) / count
        xx = sum((px - x) ** 2 for px, _ in points)
        yy = sum((py - y) ** 2 for _, py in points)
        xy = sum((px - x) * (py - y) for px, py in points)
        angle = 0.5 * math.atan2(2 * xy, xx - yy)
        dx, dy = math.cos(angle), math.sin(angle)
        along = [(px - x) * dx + (py - y) * dy for px, py in points]
        across = [(py - y) * dx - (px - x) * dy for px, py in points]
        length, thickness = max(along) - min(along), max(across) - min(across) + 1
        if length >= 12 and thickness <= 6 and length >= 4 * thickness:
            result.append(_Segment(x, y, dx, dy, length, thickness))
    return result


def _crossing(source, selection):
    # Work at a fixed scale local to the hole. Peaks are sampled outside a guard
    # so a high-contrast outline of the removed object cannot anchor a line.
    size = source.size
    luma = source.convert("L")
    luma.thumbnail((512, 512), Image.Resampling.BOX)
    hole = selection.resize(luma.size, Image.Resampling.BOX).point(lambda v: 255 if v else 0)
    ring = ImageChops.subtract(grow_support(hole, 20), grow_support(hole, 3))
    opaque = source.getchannel("A").resize(luma.size, Image.Resampling.BOX).point(lambda a: 255 if a == 255 else 0)
    ring = ImageChops.darker(ring, opaque)
    dark = ImageChops.subtract(luma.filter(ImageFilter.GaussianBlur(5)), luma)
    peaks = ImageChops.darker(dark.point(lambda v: 255 if v >= 18 else 0), ring)
    segments = _segments(peaks)
    # Other panel edges may also enter the ring. Continue a line only when
    # exactly one pair has aligned tangents and crosses the interior of the hole.
    if len(segments) > 12:
        return None
    pairs = []
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            distance = math.hypot(second.x - first.x, second.y - first.y)
            if distance < max(24, 2 * max(first.length, second.length)):
                continue
            dx, dy = (second.x - first.x) / distance, (second.y - first.y) / distance
            if any(abs(dx * s.dy - dy * s.dx) > 0.055 for s in (first, second)):
                continue
            if max(first.thickness, second.thickness) > 1.8 * min(first.thickness, second.thickness):
                continue
            if all(
                hole.getpixel((round(first.x + t * dx * distance), round(first.y + t * dy * distance)))
                for t in (0.25, 0.5, 0.75)
            ):
                pairs.append((first, second))
    if len(pairs) != 1:
        return None
    segments = pairs[0]
    sx, sy = size[0] / luma.width, size[1] / luma.height
    anchors = [((s.x + 0.5) * sx - 0.5, (s.y + 0.5) * sy - 0.5) for s in segments]
    return anchors, max(4, math.ceil(max(s.thickness for s in segments) * max(sx, sy) * 1.8))


def _continue_profile(surface, source, selection, anchors, radius):
    (x0, y0), (x1, y1) = anchors
    length = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -dy, dx
    pixels = source.convert("RGB").load()
    profiles = []
    for x, y in anchors:
        row = []
        for k in range(-radius, radius + 1):
            values = []
            for t in range(-4, 5):
                px, py = round(x + nx * k + dx * t), round(y + ny * k + dy * t)
                if 0 <= px < source.width and 0 <= py < source.height and not selection.getpixel((px, py)):
                    values.append(pixels[px, py])
            if not values:
                return False
            row.append(tuple(sum(rgb[c] for rgb in values) / len(values) for c in range(3)))
        # Carry the line's measured contrast while preserving the underlying lighting.
        profiles.append(
            [
                tuple(v[c] - ((1 - i / (2 * radius)) * row[0][c] + i / (2 * radius) * row[-1][c]) for c in range(3))
                for i, v in enumerate(row)
            ]
        )
    if any(min(sum(v) / 3 for v in profile) > -25 for profile in profiles):
        return False
    output = surface.load()
    mask = selection.load()
    # Visit a narrow strip along the major axis even for an oblique seam.
    horizontal = abs(dx) >= abs(dy)
    a0, b0, da, db = (x0, y0, dx, dy) if horizontal else (y0, x0, dy, dx)
    a1 = x1 if horizontal else y1
    major, minor = surface.size if horizontal else surface.size[::-1]
    reach = math.ceil(radius / abs(da)) + 1
    for a in range(max(0, math.floor(min(a0, a1)) - reach), min(major, math.ceil(max(a0, a1)) + reach + 1)):
        center = b0 + (a - a0) * db / da
        for b in range(max(0, round(center) - reach), min(minor, round(center) + reach + 1)):
            x, y = (a, b) if horizontal else (b, a)
            k = (x - x0) * nx + (y - y0) * ny + radius
            t = ((x - x0) * dx + (y - y0) * dy) / length
            if not mask[x, y] or not (0 <= k < 2 * radius and 0 <= t <= 1):
                continue
            i, f = math.floor(k), k % 1
            rgb = output[x, y]
            endpoints = [tuple(p[i][c] * (1 - f) + p[i + 1][c] * f for c in range(3)) for p in profiles]
            output[x, y] = tuple(
                max(0, min(255, round(rgb[c] + endpoints[0][c] * (1 - t) + endpoints[1][c] * t))) for c in range(3)
            )
    return True


def structured_background(source, selection, progress, cancelled, *, horizontal=0.5):
    bounds = selection.getbbox()
    if not bounds:
        return None
    x0, y0, x1, y1 = bounds
    pad = max(28, round(min(x1 - x0, y1 - y0) * 0.3))
    box = (max(0, x0 - pad), max(0, y0 - pad), min(source.width, x1 + pad), min(source.height, y1 + pad))
    local, alpha = source.crop(box), selection.crop(box)
    crossing = _crossing(local, alpha)
    if crossing is None:
        return None
    # Remove the thin line from the lighting samples before reconstructing its
    # measured profile. Otherwise its dark endpoints can bleed into the surface.
    anchors, radius = crossing
    smooth_source = local.copy()
    line_mask = Image.new("L", local.size)
    (x, y), (xx, yy) = anchors
    length = math.hypot(xx - x, yy - y)
    extend = max(local.size) / length
    ImageDraw.Draw(line_mask).line(
        [(x - (xx - x) * extend, y - (yy - y) * extend), (xx + (xx - x) * extend, yy + (yy - y) * extend)],
        fill=255,
        width=2 * radius + 1,
    )
    smooth_source.paste(local.filter(ImageFilter.GaussianBlur(radius * 2)), mask=line_mask)
    surface = smooth_background(smooth_source, alpha, progress, cancelled, horizontal=horizontal, boundary_only=True)
    if surface is None or not _continue_profile(surface, local, alpha, anchors, radius):
        return None
    result = source.convert("RGB")
    result.paste(surface, box[:2], alpha.point(lambda v: 255 if v else 0))
    return result
