# SPDX-License-Identifier: Apache-2.0
"""Different views for deterministic erasers, and comparison of visible repairs."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace

from PIL import Image, ImageChops

from openfill.backend import BackendError, GenerationCancelled

# These operations are their own inverses. Keep the normal, single-pass repair first.
VIEWS = (
    None,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.ROTATE_180,
    Image.Transpose.TRANSPOSE,
)


def transform_png(data, view):
    if view is None:
        return data
    with Image.open(io.BytesIO(data)) as original:
        buffer = io.BytesIO()
        original.transpose(view).save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()


def _rect(rect, width, height, view):
    if view == Image.Transpose.TRANSPOSE:
        return replace(rect, x=rect.y, y=rect.x, width=rect.height, height=rect.width)
    x = width - rect.right if view in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.ROTATE_180) else rect.x
    y = height - rect.bottom if view in (Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_180) else rect.y
    return replace(rect, x=x, y=y)


def _request_view(request, view):
    if view is None:
        return request
    swap = view == Image.Transpose.TRANSPOSE

    def orient(tile):
        return replace(
            tile,
            rect=_rect(tile.rect, request.width, request.height, view),
            work_width=tile.work_height if swap else tile.work_width,
            work_height=tile.work_width if swap else tile.work_height,
            done=tuple(_rect(done, tile.rect.width, tile.rect.height, view) for done in tile.done),
        )

    return replace(
        request,
        image_png=transform_png(request.image_png, view),
        mask_png=transform_png(request.mask_png, view),
        width=request.height if swap else request.width,
        height=request.width if swap else request.height,
        work_width=request.work_height if swap else request.work_width,
        work_height=request.work_width if swap else request.work_height,
        tiles=tuple(map(orient, request.tiles)),
        refine_tiles=tuple(map(orient, request.refine_tiles)),
    )


def render_views(request, count, inpaint, progress, preview, cancelled):
    """Run the same model on aligned image/mask views. Stage nothing until all are complete."""
    if not 1 <= count <= len(VIEWS):
        raise ValueError("Choose between one and five removal variations.")
    first = None
    images = []
    for index, view in enumerate(VIEWS[:count]):
        if cancelled():
            raise GenerationCancelled("Object removal cancelled.")

        def report(fraction, message, i=index):
            progress((i + fraction) / count, f"Repair {i + 1}/{count}: {message}")

        report(0, "Removing object...")
        result = inpaint(_request_view(request, view), report, preview if index == 0 else None, cancelled)
        if len(result.images) != 1:
            raise BackendError("The removal model did not return one repair for this view.")
        if first is None:
            first = result
        images.append(transform_png(result.images[0], view))
    return replace(first, images=images)


def distinct_repairs(images, source, mask):
    """Compare the visible, composited pixels. Different encodings or hidden RGB aren't choices."""
    if len(images) < 2:
        return images
    box = mask.getbbox() if mask is not None else None
    box = box or (0, 0, *source.size)
    base = source.crop(box).convert("RGBA")
    alpha = mask.crop(box) if mask is not None else None
    seen, unique = set(), []
    for data in images:
        with Image.open(io.BytesIO(data)) as original:
            candidate = original.convert("RGBA")
        if candidate.size != source.size:
            candidate = candidate.resize(source.size, Image.Resampling.LANCZOS)
        candidate = candidate.crop(box)
        if alpha is not None:
            candidate.putalpha(ImageChops.multiply(candidate.getchannel("A"), alpha))
        key = hashlib.sha256(Image.alpha_composite(base, candidate).tobytes()).digest()
        if key not in seen:
            seen.add(key)
            unique.append(data)
    return unique
