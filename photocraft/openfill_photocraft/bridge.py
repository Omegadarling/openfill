# OpenFill for PhotoCraft
# SPDX-License-Identifier: Apache-2.0
"""PhotoCraft's side of OpenFill: the functions ``openfill.app.App`` calls on ``App.io``.

It mirrors ``openfill.gimp_io`` over PhotoCraft's control channel. Pixels travel as
PNG files through the folder PhotoCraft was given as its automation root.

The user's document is only read until a job's result is ready, and then changed in
one step, so one Undo takes a fill back:

- out: the document is duplicated as one merged layer and saved; the selection is
  saved into that duplicate (not into the document) and saved from there.
- in: results are kept here (``PCImage.pending``) while OpenFill still works on them
  (seam correction, grain, ranking). They are then built in scratch documents (a
  canvas-sized PNG per layer, so each lands pixel-exact, its mask and group) and
  copied into the document with one ``layer.copyToDocument``. The clipboard is never
  used.
"""

from __future__ import annotations

import io as _io
import itertools
import os
import re
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from openfill import png
from openfill.backend import GenerationCancelled
from openfill.geometry import Rect, flatten_over_mean, plan_native_removal

from .control import ControlClient, ControlError
from .local_removal import grow_support, removal_mask, smooth_background, texture_donors
from .removal_blend import blend_reconstruction
from .removal_variations import distinct_repairs, render_views

# How often a mask is tried before a job's result fails (see ``_add_mask``).
MASK_ATTEMPTS = 4


class BridgeError(RuntimeError):
    """PhotoCraft did not do what OpenFill asked."""


@dataclass(eq=False)
class PCLayer:
    """A layer OpenFill made: its pixels, and its id in PhotoCraft once it is in the document."""

    id: int | None
    name: str
    origin: Rect
    rgba: bytes  # origin-sized RGBA
    group: int | None = None
    visible: bool = True
    # How its mask is made: "" none, "selection" (the job's selection grown by ``grow``),
    # "pixels" (``mask``, origin-sized); both feathered by ``feather``.
    mask_kind: str = ""
    grow: int = 0
    feather: float = 0.0
    mask: bytes | None = None
    selection: bytes | None = None  # the job's selection (canvas-sized), for a "selection" mask built later

    def get_name(self) -> str:
        return self.name

    def get_width(self) -> int:
        return self.origin.width

    def get_height(self) -> int:
        return self.origin.height


@dataclass(eq=False)
class PCImage:
    """The PhotoCraft document a job works on, found again by its tab position and name."""

    bridge: PhotoCraftBridge
    name: str
    index: int | None = None
    layers: list[PCLayer] = field(default_factory=list)
    pending: list[PCLayer] = field(default_factory=list)  # made, not yet in the document
    pending_group: str = ""  # the group several pending variations go into
    selection: bytes | None = None  # the job's selection, canvas-sized 8-bit
    _snapshot: tuple[int, Image.Image, Image.Image | None] | None = None  # (revision, visible, selection)
    shift: tuple[int, int] = (0, 0)  # where the job's original (0, 0) is now (an expand grows the canvas)
    excluded_layers: tuple[int, ...] = ()  # existing choices omitted from an appended edit's input
    rgb_profile: bytes | None = None  # RGB space of the exported source pixels

    def _inspect(self) -> dict:
        self.bridge.activate(self)
        return self.bridge.client.run("document.inspect")

    def get_width(self) -> int:
        return int(self._inspect()["width"])

    def get_height(self) -> int:
        return int(self._inspect()["height"])

    def undo_group_start(self) -> None:
        """Results reach the document in one step anyway (see the module docstring)."""

    def undo_group_end(self) -> None:
        pass


class _Progress:
    """Stand-in for GIMP's status-bar progress (``App.check_backend`` uses it)."""

    def __init__(self, title: str, sink: Callable[[float, str], None] | None = None) -> None:
        self.title, self.sink = title, sink

    def restart(self) -> None:
        pass

    def __call__(self, fraction: float, text: str) -> None:
        if self.sink:
            self.sink(fraction, text)

    def done(self) -> None:
        pass


class PhotoCraftBridge:
    """``App.io`` for PhotoCraft (same function names as ``openfill.gimp_io``)."""

    def __init__(
        self,
        client: ControlClient,
        exchange_dir: Path,
        profile: Path,
        notify: Callable[[str], None] | None = None,
        read_root: Path | None = None,
    ) -> None:
        self.client = client
        self.exchange = Path(exchange_dir)  # PhotoCraft's automation write root
        # Its read root: the exchange folder, or a folder that holds it (the home folder, so the
        # launcher can open images there too). Paths for app.open are relative to it.
        self.read_root = Path(read_root) if read_root else self.exchange
        self.profile = Path(profile)
        self.notify = notify or (lambda _text: None)
        self.on_region: Callable[[tuple[int, int, int, int]], None] | None = None
        self._names = itertools.count(1)
        self._known_images: list[PCImage] = []
        self.exchange.mkdir(parents=True, exist_ok=True)

    # session ------------------------------------------------------------------

    def session(self) -> dict:
        return self.client.call("ui.inspect").get("session") or {}

    def documents(self) -> list[dict]:
        return list(self.session().get("documents") or [])

    def active_image(self) -> PCImage | None:
        session = self.session()
        active = session.get("active")
        for d in session.get("documents") or []:
            if d.get("index") == active:
                return PCImage(self, str(d.get("name")), int(d["index"]))
        return None

    def _find(self, image: PCImage, session: dict | None = None) -> dict:
        docs = (session or self.session()).get("documents") or []
        for d in docs:  # where it was, if it is still there (two tabs may share a name)
            if d.get("index") == image.index and d.get("name") == image.name:
                return d
        for d in docs:
            if d.get("name") == image.name:
                image.index = int(d["index"])
                return d
        raise BridgeError(f"the document '{image.name}' is no longer open in PhotoCraft")

    def activate(self, image: PCImage) -> None:
        session = self.session()
        d = self._find(image, session)
        if d.get("index") != session.get("active"):
            self.client.run("document.activate", document=d["index"])

    def has_image(self, image: PCImage) -> bool:
        try:
            self._find(image)
        except BridgeError:
            return False
        return True

    def _revision(self, image: PCImage) -> int:
        return int(self._find(image).get("revision") or 0)

    def _file(self, stem: str) -> str:
        """A fresh name in the exchange folder (relative, as PhotoCraft wants it)."""
        return f"OpenFill {stem} {os.getpid()}-{next(self._names)}.png"

    def _read(self, rel: str) -> Image.Image:
        path = self.exchange / rel
        for _ in range(50):  # the save returns when written; allow a slow disk a moment
            if path.is_file():
                break
            time.sleep(0.05)
        with Image.open(path) as im:
            im.load()
            out = im.copy()
        self._forget(rel)
        return out

    def _write(self, image: Image.Image, stem: str) -> str:
        rel = self._file(stem)
        image.save(self.exchange / rel, compress_level=1)
        return rel

    def _open_path(self, rel: str) -> str:
        """``rel`` (in the exchange folder) as app.open wants it: relative to the read root."""
        return os.path.relpath(self.exchange / rel, self.read_root).replace(os.sep, "/")  # PhotoCraft wants /

    def open_file(self, path: str) -> dict:
        """Open an image the user picked (inside the read root) in PhotoCraft."""
        try:
            rel = os.path.relpath(Path(path).resolve(), self.read_root.resolve())
        except ValueError:  # another drive (Windows)
            rel = ".."
        if rel.startswith(".."):
            raise BridgeError(f"PhotoCraft may only open files in {self.read_root} for OpenFill")
        return self.client.call("app.open", path=rel.replace(os.sep, "/")) or {}

    def _forget(self, rel: str) -> None:
        try:
            (self.exchange / rel).unlink()
        except OSError:
            pass

    def _open(self, image: Image.Image, stem: str) -> tuple[int, int]:
        """Open ``image`` as a scratch document; (its index, its one layer's id)."""
        rel = self._write(image, stem)
        try:
            self.client.call("app.open", path=self._open_path(rel))
        finally:
            self._forget(rel)
        index = int(self.session().get("active"))
        layers = (self.client.run("layer.list") or {}).get("layers") or []
        if not layers:
            raise BridgeError("PhotoCraft opened an empty document")
        return index, int(layers[0]["id"])

    def _close(self, index: int) -> None:
        try:
            self.client.run("file.close", document=index)
        except ControlError:
            pass

    # the functions App calls ----------------------------------------------------

    def profile_dir(self) -> str:
        return str(self.profile)

    def message(self, text: str) -> None:
        self.notify(text)

    def GimpProgress(self, title: str) -> _Progress:  # noqa: N802 - the name App calls
        return _Progress(title)

    def is_indexed(self, image: PCImage) -> bool:
        return False  # PhotoCraft documents are RGB, Gray, CMYK or Lab; the export is RGB(A)

    def image_rect(self, image: PCImage) -> Rect:
        d = image._inspect()
        return Rect(0, 0, int(d["width"]), int(d["height"]))

    def selection_rect(self, image: PCImage) -> Rect | None:
        d = image._inspect()
        bounds = d.get("selectionBounds") if d.get("hasSelection") else None
        if not bounds:
            return None
        x, y, w, h = (int(v) for v in bounds)
        return Rect(x, y, w, h) if w > 0 and h > 0 else None

    def _snapshot(self, image: PCImage) -> tuple[Image.Image, Image.Image | None]:
        """Read the canvas and selection through a duplicate without changing the source."""
        self.commit(image)
        revision = self._revision(image)
        if image._snapshot and image._snapshot[0] == revision:
            return image._snapshot[1], image._snapshot[2]
        self.activate(image)
        has_selection = bool(self.client.run("document.inspect").get("hasSelection"))
        original_layers = self._flat_layers(image) if image.excluded_layers else []
        self.client.run("image.duplicate", name="OpenFill export", mergedOnly=not image.excluded_layers)
        duplicate = int(self.session().get("active"))
        visible_rel, mask_rel = self._file("visible"), self._file("mask")
        mask_doc = None
        try:
            if image.excluded_layers:
                copied_layers = self._layer_rows()

                def structure(rows):
                    return [(row.get("name"), row.get("depth"), row.get("kind")) for row in rows]

                if structure(original_layers) != structure(copied_layers):
                    raise BridgeError("Could not match the source layers in the export copy.")
                copied_ids = {
                    row["id"]: copied["id"] for row, copied in zip(original_layers, copied_layers, strict=True)
                }
                for layer_id in image.excluded_layers:
                    if layer_id not in copied_ids:
                        raise BridgeError("A previous variation is missing from the document.")
                    self.client.run("layer.setProps", layer=copied_ids[layer_id], visible=False)
            self.client.call("app.save", path=visible_rel)
            if has_selection:
                self.activate(image)
                self.client.run("select.saveSelection", name="OpenFill selection", document=duplicate)
                self.client.run("document.activate", document=duplicate)
                self.client.run("channel.duplicate", channel="OpenFill selection", document="new", name="OpenFill mask")
                mask_doc = int(self.session().get("active"))
                self.client.call("app.save", path=mask_rel)
        finally:
            for index in sorted((i for i in (duplicate, mask_doc) if i is not None), reverse=True):
                self._close(index)
            self.activate(image)
        visible = self._read(visible_rel).convert("RGBA")
        profile = visible.info.get("icc_profile") or b""
        image.rgb_profile = profile if profile[16:20] == b"RGB " and profile[36:40] == b"acsp" else None
        selection = self._read(mask_rel).convert("L") if mask_doc is not None else None
        image._snapshot = (revision, visible, selection)
        return visible, selection

    def export_region(self, image: PCImage, rect: Rect, transparent_outside: bool = False) -> tuple[bytes, bytes]:
        """(opaque RGBA PNG, raw RGBA) of ``rect``; outside the canvas is transparent before
        flattening over the mean colour, as GIMP does for an expand."""
        full, _selection = self._snapshot(image)
        if self.on_region:  # where this round's preview frames belong, in the job's first canvas
            self.on_region((rect.x - image.shift[0], rect.y - image.shift[1], rect.width, rect.height))
        crop = full.crop((rect.x, rect.y, rect.x + rect.width, rect.y + rect.height))
        flat = flatten_over_mean(crop.tobytes())
        return png.encode(rect.width, rect.height, flat, 4), flat

    def export_region_png(self, image: PCImage, rect: Rect, transparent_outside: bool = False) -> bytes:
        return self.export_region(image, rect, transparent_outside)[0]

    def export_selection_mask(self, image: PCImage, rect: Rect) -> tuple[bytes, bytes]:
        """(8-bit PNG, raw bytes) of the selection inside ``rect``; 255 = selected. The whole
        selection is kept for the result's mask, so it is exactly this one even if the user
        changes the selection while the job runs."""
        full, selection = self._snapshot(image)
        if selection is None:
            selection = Image.new("L", full.size, 255)
        image.selection = selection.tobytes()
        crop = selection.crop((rect.x, rect.y, rect.x + rect.width, rect.y + rect.height))
        raw = crop.tobytes()
        return png.encode(rect.width, rect.height, raw, 1), raw

    def export_selection_mask_png(self, image: PCImage, rect: Rect) -> bytes:
        return self.export_selection_mask(image, rect)[0]

    def mask_png_from_bytes(self, rect: Rect, pixels: bytes) -> bytes:
        return png.encode(rect.width, rect.height, pixels, 1)

    def remove_with_texture(self, image: PCImage, variations: int, seed: int, progress, cancelled) -> list[PCLayer]:
        """Synthesize from surrounding pixels in scratch documents, then stage the masked results."""
        visible, selection = self._snapshot(image)
        bounds = selection.getbbox() if selection is not None else None
        if bounds is None:
            raise BridgeError("Select the object and its shadow before pressing Remove.")
        if selection.getextrema()[0] > 0:
            raise BridgeError("Leave some background outside the selection so OpenFill can sample it.")
        left, top, right, bottom = bounds
        target = Rect(left, top, right - left, bottom - top)
        context = plan_native_removal(target, Rect(0, 0, *visible.size)).context
        box = (context.x, context.y, context.right, context.bottom)
        source = visible.crop(box)
        alpha = removal_mask(selection.crop(box))
        results = self._sample_texture(image, source, alpha, variations, seed, progress, cancelled)
        results = distinct_repairs(results, source, alpha)
        return self.insert_results(image, results, context, "Object Removal", 0, None, mask_pixels=alpha.tobytes())

    def removal_guide(self, image, size, rgba, mask, seed, progress, cancelled, *, sample_detail=False) -> bytes:
        """Make a clean plate from the exact source crop and mask captured for this AI job."""
        source = Image.frombytes("RGBA", size, rgba)
        alpha = Image.frombytes("L", size, mask)
        data = self._sample_texture(image, source, alpha, 1, seed, progress, cancelled, sample_detail=sample_detail)[0]
        # Texture synthesis also clears its narrow donor guard. The model should
        # see the original scene everywhere outside the actual removal layer.
        with Image.open(_io.BytesIO(data)) as repair:
            guide = Image.composite(
                repair.convert("RGB"), source.convert("RGB"), alpha.point(lambda v: 255 if v else 0)
            )
        buffer = _io.BytesIO()
        guide.save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()

    def removal_locator(self, size, rgba: bytes, mask: bytes) -> bytes:
        """Mark the selected object for an instruction editor while retaining its scene context."""
        source = Image.frombytes("RGBA", size, rgba).convert("RGB")
        tint = Image.frombytes("L", size, mask).point(lambda value: round(value * 0.6))
        marked = Image.composite(Image.new("RGB", size, (255, 32, 64)), source, tint)
        buffer = _io.BytesIO()
        marked.save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()

    render_removal_views = staticmethod(render_views)

    @staticmethod
    def distinct_removal_images(images, size, rgba, mask):
        return distinct_repairs(
            images, Image.frombytes("RGBA", size, rgba), Image.frombytes("L", size, mask) if mask is not None else None
        )

    def prepare_removal_mask(self, size, pixels: bytes) -> bytes:
        return removal_mask(Image.frombytes("L", size, pixels)).tobytes()

    def blend_removal_results(self, clean_plate: bytes, mask: bytes, proposals: list[bytes]) -> list[bytes]:
        with Image.open(_io.BytesIO(clean_plate)) as image:
            plate = image.convert("RGB")
        selection = Image.frombytes("L", plate.size, mask)
        results = []
        for index, data in enumerate(proposals, 1):
            with Image.open(_io.BytesIO(data)) as proposal:
                matched, fallback = blend_reconstruction(plate, proposal, selection)
            if fallback:
                self.notify(
                    f"Variation {index} uses the local repair because the AI reconstruction differed sharply from the sampled background."
                )
            buffer = _io.BytesIO()
            matched.save(buffer, format="PNG", compress_level=1)
            results.append(buffer.getvalue())
        return results

    def _sample_texture(self, image, source, alpha, variations, seed, progress, cancelled, *, sample_detail=False) -> list[bytes]:
        # Every selected pixel is excluded from the donor pool. Apply the artist's
        # soft edge only once, on the result layer in the original document.
        support = grow_support(alpha, 2)
        if not ImageChops.subtract(source.getchannel("A"), support).getbbox():
            raise BridgeError(
                "No visible background surrounds this selection. To erase to transparency, clear the selection on the object's layer."
            )
        smooth_results = []
        for index, horizontal in enumerate((0.5, 0.85, 0.15, 0.70, 0.30)[:variations]):

            def report(fraction, message, i=index):
                progress((i + fraction) / variations, f"Repair {i + 1}/{variations}: {message}")

            smooth = smooth_background(source, alpha, report, cancelled, horizontal=horizontal)
            if smooth is None:
                break  # This scene needs sampled texture.
            if sample_detail:
                # Continue fine surface grain separately from the interpolated lighting.
                # The scratch image contains only detail from the cleared background.
                plate = Image.composite(smooth, source.convert("RGB"), alpha.point(lambda v: 255 if v else 0))
                detail = ImageChops.subtract(plate, plate.filter(ImageFilter.GaussianBlur(8)), offset=128)
                detail.putalpha(source.getchannel("A"))
                sample = self._sample_patches(image, detail, support, 1, seed, progress, cancelled)[0]
                with Image.open(_io.BytesIO(sample)) as grain:
                    textured = ImageChops.add(smooth, grain.convert("RGB"), offset=-128)
                smooth = Image.blend(smooth, textured, 0.5)
            buffer = _io.BytesIO()
            smooth.save(buffer, format="PNG", compress_level=1)
            smooth_results.append(buffer.getvalue())
        if smooth_results:
            return smooth_results
        return self._sample_patches(image, source, support, variations, seed, progress, cancelled)

    def _sample_patches(self, image, source, support, variations, seed, progress, cancelled) -> list[bytes]:
        donors = texture_donors(source, support)
        guide = Image.new("RGBA", source.size, "white")
        guide.putalpha(support)
        first_seed = secrets.randbits(32) if seed < 0 else seed
        results = []
        mask_doc, mask_layer = self._open(guide, "removal mask")
        donor_doc = None
        progress_label = "Sampling nearby texture"
        try:
            if donors is not None:
                guide.putalpha(donors)
                donor_doc, donor_layer = self._open(guide, "texture samples")
                progress_label = "Sampling matching background texture"
            for index in range(variations):
                if cancelled():
                    raise GenerationCancelled("Object removal cancelled.")
                progress(index / variations, f"{progress_label} ({index + 1}/{variations})...")
                scratch, layer = self._open(source, "removal source")
                rel = self._file("removal result")
                try:
                    options = {"sampling": "auto"}
                    if donor_doc is not None:
                        self.client.run(
                            "select.loadSelection", channel="transparency", layer=donor_layer, document=donor_doc
                        )
                        self.client.run("select.saveSelection", name="OpenFill texture samples", document=scratch)
                        options = {"sampling": "custom", "channel": "OpenFill texture samples"}
                    self.client.run("select.loadSelection", channel="transparency", layer=mask_layer, document=mask_doc)
                    self.client.run(
                        "edit.contentAwareFill",
                        layer=layer,
                        output="current",
                        seed=(first_seed + index) % (1 << 64),
                        **options,
                    )
                    self.client.call("app.save", path=rel)
                    results.append((self.exchange / rel).read_bytes())
                finally:
                    self._forget(rel)
                    self._close(scratch)
        finally:
            if donor_doc is not None:
                self._close(donor_doc)
            self._close(mask_doc)
            self.activate(image)
        return results

    # writing ------------------------------------------------------------------------

    def insert_results(
        self,
        image: PCImage,
        images: list[bytes],
        origin: Rect,
        base_name: str,
        feather: float,
        temp_dir: Path | None,
        use_selection_mask: bool = True,
        group_variations: bool = True,
        mask_pixels: bytes | None = None,
        grow: int = 0,
        scale_to_origin: bool = False,
    ) -> list[PCLayer]:
        """Generated PNGs as layers at ``origin`` with a mask (see ``gimp_io.insert_results``).
        Several variations go into a group; only the first is visible. They reach the document
        at the end of the job (``commit``)."""
        self.commit(image)  # an earlier round's results come first
        layers: list[PCLayer] = []
        for index, data in enumerate(images):
            name = base_name if len(images) == 1 else f"{base_name} ({index + 1})"
            with Image.open(_io.BytesIO(data)) as im:
                rgba = im.convert("RGBA")
            if scale_to_origin and rgba.size != (origin.width, origin.height):
                rgba = rgba.resize((origin.width, origin.height), Image.LANCZOS)
            kind = "" if not use_selection_mask else ("pixels" if mask_pixels is not None else "selection")
            layers.append(
                PCLayer(
                    None,
                    name,
                    origin,
                    rgba.tobytes(),
                    None,
                    index == 0,
                    kind,
                    grow,
                    feather,
                    mask_pixels,
                    image.selection,
                )
            )
        image.pending = layers
        image.pending_group = base_name if group_variations and len(images) > 1 else ""
        image.layers.extend(layers)
        return layers

    def layer_rgba(self, layer: PCLayer) -> bytes:
        return layer.rgba

    def set_layer_rgba(self, layer: PCLayer, data: bytes) -> None:
        """Replace a layer's pixels: here while it is pending; in the document, a copy with the
        new pixels, its mask, name and visibility goes above it and the old one is deleted."""
        layer.rgba = data
        if layer.id is None:
            return
        image = self._image_of(layer)
        scratch, _ids, top = self._build(image, [layer], "")
        try:
            self.activate(image)
            self.client.run("layer.select", layer=layer.id)
            copied = self._copy_in(image, scratch, top)
        finally:
            self._close(scratch)
            self.activate(image)
        self.client.run("layer.delete", layer=layer.id)
        layer.id = copied
        self.client.run("layer.select", layer=copied)

    def _mask_image(self, image: PCImage, layer: PCLayer, size: tuple[int, int]) -> Image.Image | None:
        """The canvas-sized 8-bit mask ``layer`` gets before PhotoCraft grows or feathers it."""
        if layer.mask_kind == "pixels" and layer.mask is not None:
            mask = Image.frombytes("L", (layer.origin.width, layer.origin.height), layer.mask)
            if layer.feather > 0:
                # Feathered here, not by PhotoCraft: a Gaussian blur that clamps at the canvas
                # edges keeps the generated area fully opaque (the canvas an expand extended is
                # filled with the background colour, which a soft edge would let through).
                mask = mask.filter(ImageFilter.GaussianBlur(float(layer.feather) / 2))
            full = Image.new("L", size, 0)
            full.paste(mask, (layer.origin.x, layer.origin.y))
            return full
        if layer.mask_kind == "selection":
            selection = layer.selection if layer.selection is not None else image.selection
            if selection is not None and len(selection) == size[0] * size[1]:
                return Image.frombytes("L", size, selection)
            return Image.new("L", size, 255)
        return None

    def _build(self, image: PCImage, layers: list[PCLayer], group_name: str) -> tuple[int, list[int], int]:
        """Build ``layers`` (top first) with masks, names and visibility in a scratch document
        the size of the canvas; (its index, the layer ids there, the id to copy)."""
        d = image._inspect()
        size = (int(d["width"]), int(d["height"]))
        scratch, ids = None, []
        try:
            for layer in reversed(layers):  # each copy lands above the last, so bottom first
                full = Image.new("RGBA", size, (0, 0, 0, 0))
                full.paste(
                    Image.frombytes("RGBA", (layer.origin.width, layer.origin.height), layer.rgba),
                    (layer.origin.x, layer.origin.y),
                )
                # These pixels use the exported document's RGB coordinates. A
                # differently tagged scratch document triggers a second color
                # conversion in PhotoCraft's layer.copyToDocument.
                if image.rgb_profile:
                    full.info["icc_profile"] = image.rgb_profile
                index, layer_id = self._open(full, "layer")
                if scratch is None:
                    scratch = index
                    ids.insert(0, layer_id)
                else:
                    copied = self.client.run("layer.copyToDocument", document=scratch, source=index, layers=[layer_id])
                    self._close(index)
                    ids.insert(0, int((copied or {}).get("layers", [0])[0]))
            assert scratch is not None
            self.client.run("document.activate", document=scratch)
            masks: dict[tuple, list[int]] = {}
            for layer, layer_id in zip(layers, ids, strict=True):
                key = (layer.mask_kind, layer.grow, layer.feather, id(layer.mask))
                if layer.mask_kind:
                    masks.setdefault(key, []).append(layer_id)
            for (kind, grow, feather, _), targets in masks.items():
                sample = next(layer for layer, i in zip(layers, ids, strict=True) if i == targets[0])
                mask = self._mask_image(image, sample, size)
                if mask is None:
                    continue
                rgba = Image.new("RGBA", size, (255, 255, 255, 0))
                rgba.putalpha(mask)
                mask_doc, mask_layer = self._open(rgba, "mask")
                try:
                    self.client.run("document.activate", document=scratch)
                    for layer_id in targets:  # adding a mask uses the selection up, so load it for each
                        self._add_mask(scratch, layer_id, kind, grow, feather, mask_doc, mask_layer)
                finally:
                    self._close(mask_doc)
                    self.client.run("document.activate", document=scratch)
            for layer, layer_id in zip(layers, ids, strict=True):
                self.client.run("layer.setProps", layer=layer_id, name=layer.name, visible=layer.visible)
            top = ids[0]
            if group_name:
                top = int((self.client.run("layer.new.group", name=group_name) or {}).get("layer"))
                for layer_id in reversed(ids):  # "into" puts it at the top of the group
                    self.client.run("layer.moveTo", layer=layer_id, target=top, position="into")
            return scratch, ids, top
        except BaseException:
            if scratch is not None:
                self._close(scratch)
            raise

    def _add_mask(
        self, scratch: int, layer_id: int, kind: str, grow: int, feather: float, mask_doc: int, mask_layer: int
    ) -> None:
        """Give ``layer_id`` (in the scratch document) the mask held by ``mask_layer``'s pixels.

        The mask travels through the scratch document's selection for a moment, because
        PhotoCraft adds a mask from a selection. The mask itself comes from the job's stored
        selection or mask, never the user's. A click in PhotoCraft at that moment (a deselect,
        say) can clear that selection under us ("no selection"), so the whole sequence is
        loaded again and tried a few times before giving up."""
        for attempt in range(MASK_ATTEMPTS):
            try:
                self.client.run("document.activate", document=scratch)
                self.client.run("select.loadSelection", channel="transparency", layer=mask_layer, document=mask_doc)
                if kind == "selection":
                    if grow > 0:
                        self.client.run("select.modify.expand", radius=int(grow), applyAtCanvasBounds=False)
                    if feather > 0:
                        self.client.run("select.modify.feather", radius=float(feather), applyAtCanvasBounds=False)
                self.client.run("layer.layerMask.revealSelection", layer=layer_id)
                return
            except ControlError:
                if attempt == MASK_ATTEMPTS - 1:
                    raise
                time.sleep(0.2)

    def _copy_in(self, image: PCImage, scratch: int, layer_id: int) -> int:
        """Copy ``layer_id`` from the scratch document into ``image``, above its active layer:
        one step in the document's history. The copy's id."""
        target = self._find(image)
        copied = self.client.run(
            "layer.copyToDocument", document=int(target["index"]), source=scratch, layers=[layer_id]
        )
        ids = (copied or {}).get("layers") or []
        if not ids:
            raise BridgeError("PhotoCraft did not report the copied layer")
        return int(ids[0])

    def commit(self, image: PCImage) -> None:
        """Put the pending results into the document in one step, on top of the stack."""
        layers, group_name = image.pending, image.pending_group
        if not layers:
            return
        # The results stay pending until they are in the document, so a failure here can be
        # tried again and never loses them.
        scratch, _ids, top = self._build(image, layers, group_name)
        try:
            copied = self._copy_on_top(image, scratch, top)
        finally:
            self._close(scratch)
            self.activate(image)
        image.pending, image.pending_group = [], ""
        self._take_ids(image, layers, group_name, copied)
        image._snapshot = None

    def _copy_on_top(self, image: PCImage, scratch: int, top: int) -> int:
        self.activate(image)
        top_layer = self._top_layer(image)
        if top_layer is not None:
            self.client.run("layer.select", layer=top_layer)  # not a history step
        return self._copy_in(image, scratch, top)

    def _take_ids(self, image: PCImage, layers: list[PCLayer], group_name: str, copied: int) -> None:
        """Note the ids the copied ``layers`` have in the document; the first is selected."""
        if group_name:
            rows = [row for row in self._flat_layers(image) if row.get("parent") == copied]
            if len(rows) != len(layers):
                raise BridgeError("the copied group does not hold the variations")
            for layer, row in zip(layers, rows, strict=True):
                layer.id, layer.group = int(row["id"]), copied
        else:
            layers[0].id = copied
        visible = next((layer for layer in layers if layer.visible), None)
        if visible is not None:
            self.client.run("layer.select", layer=visible.id)

    def _top_layer(self, image: PCImage) -> int | None:
        """The topmost layer outside any group, so a result lands on top of the stack."""
        for row in self._flat_layers(image):
            if not row.get("depth"):
                return int(row["id"]) if row.get("id") is not None else None
        return None

    def _image_of(self, layer: PCLayer) -> PCImage:
        for image in self._known_images:
            if layer in image.layers:
                return image
        raise BridgeError("layer of an unknown document")

    def track(self, image: PCImage) -> PCImage:
        """Remember ``image`` so its layers can be found again (``set_layer_rgba``)."""
        if image not in self._known_images:
            self._known_images.append(image)
        return image

    def _flat_layers(self, image: PCImage) -> list[dict]:
        self.activate(image)
        return self._layer_rows()

    def _layer_rows(self) -> list[dict]:
        reply = self.client.run("layer.list", flat=True) or {}
        return list(reply.get("layers") or [] if isinstance(reply, dict) else reply)

    def reorder_variations(
        self, image: PCImage, layers: list[PCLayer], order: list[int], base_name: str
    ) -> list[PCLayer]:
        """Best first: the first in ``order`` becomes the top, visible, selected layer."""
        if len(layers) < 2 or order == list(range(len(layers))):
            return layers
        ordered = [layers[i] for i in order]
        for position, layer in enumerate(ordered):
            layer.name, layer.visible = f"{base_name} ({position + 1})", position == 0
        if all(layer.id is None for layer in ordered):
            image.pending = [layer for layer in ordered if layer in image.pending] or image.pending
            return ordered
        self.activate(image)
        for layer in reversed(ordered):
            if layer.group is not None:
                self.client.run("layer.moveTo", layer=layer.id, target=int(layer.group), position="into")
            else:
                self.client.run("layer.arrange.bringToFront", layer=layer.id)
        for layer in ordered:
            self.client.run("layer.setProps", layer=layer.id, name=layer.name, visible=layer.visible)
        self.client.run("layer.select", layer=ordered[0].id)
        return ordered

    def history(self, image: PCImage) -> list[str]:
        """The document's history (its step labels, oldest first)."""
        self.activate(image)
        return list((self.client.run("document.inspect") or {}).get("history") or [])

    def rewind(self, image: PCImage, state: int) -> None:
        """Take the document back to before history step ``state`` (it and every later step go)."""
        self.activate(image)
        self.client.run("history.deleteState", state=int(state))
        image._snapshot = None

    def collapse(self, image: PCImage, since: int, name: str) -> None:
        """Make history steps ``since``.. one step named ``name``: a snapshot of the document as
        it is, the steps taken back, and the snapshot selected (PhotoCraft adds that as one
        step). The snapshot itself is deleted afterwards (the document does not change)."""
        self.activate(image)
        snap = (self.client.run("history.newSnapshot", name=name, **{"from": "fullDocument"}) or {})["snapshot"]
        self.client.run("history.deleteState", state=int(since))
        try:
            self.client.run("history.selectSnapshot", snapshot=snap)
        except ControlError:
            # Once more; if that fails too the snapshot stays in the History panel, so the
            # result can still be brought back by hand.
            self.client.run("history.selectSnapshot", snapshot=snap)
        self.client.run("history.deleteSnapshot", snapshot=snap)
        image._snapshot = None

    def keep_one(self, image: PCImage, since: int, layers: list[PCLayer], chosen: PCLayer, name: str) -> None:
        """Keep: the job's result step (history step ``since``) becomes ``chosen`` alone, still
        one step. The kept layer is built first, from the job's stored pixels and selection (the
        live selection does not matter), and only then is the job's step taken back, so a
        failure while building leaves the variations as they were. Should the copy fail after
        that, the variations are put back (one step again) before the error is raised."""
        kept = PCLayer(
            None,
            name,
            chosen.origin,
            chosen.rgba,
            None,
            True,
            chosen.mask_kind,
            chosen.grow,
            chosen.feather,
            chosen.mask,
            chosen.selection,
        )
        scratch, _ids, top = self._build(image, [kept], "")
        try:
            self.rewind(image, since)
            try:
                copied = self._copy_on_top(image, scratch, top)
            except BaseException:
                self._restore(image, layers)
                raise
        finally:
            self._close(scratch)
            self.activate(image)
        chosen.id, chosen.group, chosen.visible, chosen.name = copied, None, True, name
        self.client.run("layer.select", layer=copied)
        image._snapshot = None

    def _restore(self, image: PCImage, layers: list[PCLayer]) -> None:
        """Put variations that were taken back into the document again (one step)."""
        saved = [(layer.id, layer.group, layer.visible) for layer in layers]
        try:
            for layer in layers:
                layer.id, layer.group = None, None
            group = re.sub(r" \(\d+\)$", "", layers[0].name) if len(layers) > 1 else ""
            image.pending, image.pending_group = list(layers), group
            self.commit(image)
        except BaseException:
            image.pending, image.pending_group = [], ""
            for layer, (layer_id, group_id, visible) in zip(layers, saved, strict=True):
                layer.id, layer.group, layer.visible = layer_id, group_id, visible
            raise

    def show_variation(self, image: PCImage, layers: list[PCLayer], index: int) -> None:
        """The variations picker: only ``layers[index]`` visible (two steps, not one per layer)."""
        self.commit(image)
        self.activate(image)
        for i, layer in enumerate(layers):
            want = i == index
            if layer.visible != want and layer.id is not None:
                self.client.run("layer.setProps", layer=layer.id, visible=want)
                layer.visible = want
        if layers[index].id is not None:
            self.client.run("layer.select", layer=layers[index].id)

    def resize_canvas(self, image: PCImage, new_w: int, new_h: int, off_x: int, off_y: int) -> None:
        """Grow or crop the canvas; ``off_x``/``off_y`` is where the old (0, 0) lands."""
        self.commit(image)
        self.activate(image)
        self.client.run("image.crop", x=-int(off_x), y=-int(off_y), width=int(new_w), height=int(new_h))
        image.shift = (image.shift[0] + int(off_x), image.shift[1] + int(off_y))
        image._snapshot = None
        image.selection = None  # the job's selection was for the old canvas

    def clip_layers_to_canvas(self, image: PCImage) -> None:
        """PhotoCraft's crop already cuts the layers (deleteCroppedPixels)."""

    def new_image_from_pngs(self, images: list[bytes], base_name: str, temp_dir: Path | None) -> PCImage:
        """A new document from the first image; further variations become hidden layers."""
        with Image.open(_io.BytesIO(images[0])) as im:
            first = im.convert("RGBA")
        # The document takes the file's name: make it the prompt's, not a temporary one.
        stem = re.sub(r"[^A-Za-z0-9 ,._-]+", "", base_name).strip()[:60] or "Text to Image"
        rel = f"{stem}.png"
        if (self.exchange / rel).exists():
            rel = f"{stem} {next(self._names)}.png"
        first.save(self.exchange / rel, compress_level=1)
        try:
            opened = self.client.call("app.open", path=self._open_path(rel)) or {}
        finally:
            self._forget(rel)
        index = int(self.session().get("active"))
        image = self.track(PCImage(self, str(opened.get("name") or rel), index))
        base = (self.client.run("document.inspect") or {}).get("activeLayer")
        if base is not None:
            self.client.run("layer.setProps", layer=base, name=base_name if len(images) == 1 else f"{base_name} (1)")
        for number, data in enumerate(images[1:], start=2):
            with Image.open(_io.BytesIO(data)) as im:
                rgba = im.convert("RGBA")
            extra = PCLayer(
                None, f"{base_name} ({number})", Rect(0, 0, rgba.width, rgba.height), rgba.tobytes(), visible=False
            )
            scratch, _ids, top = self._build(image, [extra], "")
            try:
                self.activate(image)
                self._copy_in(image, scratch, top)
            finally:
                self._close(scratch)
                self.activate(image)
        if base is not None:
            self.client.run("layer.select", layer=base)
        return image

    def finish(self, image: PCImage) -> None:
        """End of a job: the results go into the document (a failed job leaves it untouched)."""
        try:
            try:
                self.commit(image)
            except ControlError:
                self.commit(image)  # still pending: once more (a click may have got in the way)
        finally:
            image.selection = None
            image._snapshot = None

    def discard(self, image: PCImage) -> None:
        """A failed or cancelled job: forget results that never reached the document."""
        image.pending, image.pending_group = [], ""
        image.selection = None
        image._snapshot = None


def default_profile() -> Path:
    """Where the companion keeps settings, logs and the exchange folder."""
    override = os.environ.get("OPENFILL_PHOTOCRAFT_PROFILE")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / "OpenFill for PhotoCraft"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenFill for PhotoCraft"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "openfill-photocraft"
