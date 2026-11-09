# OpenFill - generative fill for GIMP 3
# Copyright (C) 2026 Beckett Harriman
# SPDX-License-Identifier: Apache-2.0
"""ComfyUI workflow builders (API "prompt" format).

The graphs are assembled in code so the user never sees ComfyUI. Node ids
are short stable strings; ``/prompt`` errors reference them and are mapped
back to readable messages by the backend.

Graph shapes are documented in ``docs/design.md``. The builders only need
core ComfyUI nodes; when the optional ``comfyui-inpaint-nodes`` package is
present (``Capabilities.inpaint_nodes``) they use its better mask and
pre-fill nodes and can apply the Fooocus SDXL inpaint patch.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache

from .backend import QWEN21_EDIT_TEMPLATE, InpaintRequest, ModelSpec, Txt2ImgRequest, validate_edit_references

OUTPUT_NODE = "save"


@dataclass(frozen=True)
class Capabilities:
    inpaint_nodes: bool = False
    native_lama: bool = False  # OpenFill's own node pack (OpenFillLamaNative) is loaded


class Graph:
    """Tiny helper that numbers nodes and produces the API-format dict."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add(self, node_id: str, class_type: str, **inputs) -> str:
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id}")
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    @staticmethod
    def out(node_id: str, index: int = 0) -> list:
        return [node_id, index]

    def to_dict(self) -> dict:
        return dict(self.nodes)


def _checkpoint(g: Graph, name: str) -> tuple[list, list, list]:
    g.add("ckpt", "CheckpointLoaderSimple", ckpt_name=name)
    return g.out("ckpt", 0), g.out("ckpt", 1), g.out("ckpt", 2)


def _sampler_cfg(req: InpaintRequest) -> float:
    """KSampler guidance: FLUX takes its guidance through FluxGuidance and samples at 1.0."""
    return 1.0 if req.model.arch == "flux" else float(req.sampling.cfg)


def _loaders(g: Graph, spec: ModelSpec) -> tuple[list, list, list]:
    """Model, CLIP and VAE of a fill model: one checkpoint file, or (erasers) separate
    files. RORem is an SDXL inpainting UNet in diffusers layout with SDXL's two text
    encoders and VAE; the FLUX eraser is FLUX.1 Fill with the object-removal LoRA."""
    template = spec.effective_template
    if template == "sdxl_eraser":
        g.add("unet", "UNETLoader", unet_name=spec.unet, weight_dtype="default")
        g.add("clip", "DualCLIPLoader", clip_name1=spec.clip1, clip_name2=spec.clip2, type="sdxl")
        g.add("vae", "VAELoader", vae_name=spec.vae)
        return g.out("unet"), g.out("clip"), g.out("vae")
    if template == "flux_eraser":
        model, clip, vae = _flux_loaders(g, spec)
        if spec.lora:
            g.add("lora", "LoraLoaderModelOnly", model=model, lora_name=spec.lora, strength_model=1.0)
            model = g.out("lora")
        return model, clip, vae
    return _checkpoint(g, spec.checkpoint)


def _prompts(g: Graph, clip: list, prompt: str, negative: str) -> tuple[list, list]:
    g.add("pos", "CLIPTextEncode", text=prompt, clip=clip)
    g.add("neg", "CLIPTextEncode", text=negative, clip=clip)
    return g.out("pos"), g.out("neg")


def _scale_image(g: Graph, node_id: str, image: list, width: int, height: int) -> list:
    g.add(node_id, "ImageScale", image=image, upscale_method="lanczos", width=width, height=height, crop="disabled")
    return g.out(node_id)


def _hard_model_mask(req: InpaintRequest) -> bool:
    """The model mask stays hard (grown, not blurred). The LaMa/MAT node wants
    a hard mask. An SD 1.5 inpainting checkpoint sees the hole as grey in its
    conditioning image wherever the mask is 0.5 or more, while a blurred noise
    mask lets the sampler touch that band only in the last steps; there it
    reproduced the grey of the conditioning image as a pale frame around the
    fill. With a hard mask the whole hole is generated from the first step and
    the GIMP layer mask's feather makes the soft transition. Fills only: the
    expand path keeps its blur."""
    template = req.model.effective_template
    return template == "inpaint_model" or (template == "sd15_inpaint" and req.mode == "fill")


def _prepare_mask(g: Graph, mask_image: list, req: InpaintRequest, caps: Capabilities) -> list:
    """Mask image (gray PNG loaded as IMAGE) -> grown, feathered MASK."""
    if req.needs_scaling:
        mask_image = _scale_image(g, "mask_scale", mask_image, req.work_width, req.work_height)
    g.add("mask", "ImageToMask", image=mask_image, channel="red")
    mask = g.out("mask")
    grow = max(0, int(req.grow))
    # The blur must not exceed the growth, otherwise a small selection never
    # reaches mask value 1.0 and the model only partially repaints it. The
    # LaMa/MAT node wants a hard mask: no blur at all.
    hard = _hard_model_mask(req)
    blur = 0 if hard else max(0, min(int(req.feather), grow))
    if caps.inpaint_nodes:
        if grow or blur:
            g.add("mask_grow", "INPAINT_ExpandMask", mask=mask, grow=grow, blur=blur, blur_type="gaussian")
            mask = g.out("mask_grow")
        g.add("mask_stable", "INPAINT_StabilizeMask", mask=mask, epsilon=0.01)
        return g.out("mask_stable")
    if grow:
        g.add("mask_grow", "GrowMask", mask=mask, expand=grow, tapered_corners=True)
        mask = g.out("mask_grow")
    if blur:
        g.add("mask_img", "MaskToImage", mask=mask)
        radius = max(1, min(31, blur // 2))
        g.add(
            "mask_blur", "ImageBlur", image=g.out("mask_img"), blur_radius=radius, sigma=max(1.0, min(10.0, blur / 3.0))
        )
        g.add("mask_soft", "ImageToMask", image=g.out("mask_blur"), channel="red")
        mask = g.out("mask_soft")
    return mask


def _prefill(g: Graph, image: list, mask: list, req: InpaintRequest, caps: Capabilities) -> list:
    """Hide the old content under the mask so the sampler starts from a neutral
    image instead of the object being removed (fill) or hard transparent
    pixels (expand)."""
    if not (req.prefill and caps.inpaint_nodes):
        return image
    if req.mode == "expand" or req.prefill_style == "border":
        # Fill the hole from its border so nothing of the removed object (its
        # colour, its blurred shape) survives into the sampler's starting image.
        mode = req.prefill_mode if req.prefill_mode in ("navier-stokes", "telea", "neutral") else "navier-stokes"
        g.add("prefill", "INPAINT_MaskedFill", image=image, mask=mask, fill=mode, falloff=0)
        g.add("preblur", "INPAINT_MaskedBlur", image=g.out("prefill"), mask=mask, blur=65, falloff=0)
        return g.out("preblur")
    g.add("preblur", "INPAINT_MaskedBlur", image=image, mask=mask, blur=65, falloff=9)
    return g.out("preblur")


# Templates whose graph supports the tiled main pass and the refine pass.
REFINE_TEMPLATES = ("sd15_inpaint", "sdxl_fooocus")
# Templates whose graph supports the tiled main pass (erasers work at full image
# resolution in tiles of their own working size).
TILED_TEMPLATES = (*REFINE_TEMPLATES, "sdxl_eraser", "flux_eraser")


@dataclass
class PlannedInpaintRequest(InpaintRequest):
    """An inpaint request with an optional tiled main pass and an optional
    full-resolution refine pass (``geometry.FillPlan``).

    ``full_grow`` and ``full_blur`` are the mask growth and blur at context
    resolution (image px), used by the tiled and refine passes, which work on
    the context at its own size.
    """

    tiles: tuple = ()  # geometry.Tile, main pass (empty: one pass at work size)
    refine_tiles: tuple = ()  # geometry.Tile at scale 1 (empty: no refine)
    refine_denoise: float = 0.4
    # Below 1.0 (``FillPlan.layout``): the whole context is generated first in
    # one pass at the work size, then the tiles add detail at this denoise.
    tile_denoise: float = 1.0
    full_grow: int = 0
    full_blur: int = 0
    # Generation model for the refine pass of a prompt-free removal (LaMa/MAT
    # work at 256/512 px; the refine pass restores texture at full resolution).
    refine_model: ModelSpec | None = None
    refine_prompt: str = ""
    refine_negative: str = ""

    @property
    def work_pixels(self) -> int:
        """Pixels sampled over all passes (for the timeout estimate)."""
        main = sum(t.work_width * t.work_height for t in self.tiles) or self.work_width * self.work_height
        return main + sum(t.work_width * t.work_height for t in self.refine_tiles)


def _tiles(req: InpaintRequest) -> tuple:
    return tuple(getattr(req, "tiles", ()) or ())


def _refine_tiles(req: InpaintRequest) -> tuple:
    return tuple(getattr(req, "refine_tiles", ()) or ())


def _single_pass(g: Graph, req: InpaintRequest, caps: Capabilities, model, positive, negative, vae, seed: int) -> list:
    """The classic pass: the whole context at the working size; returns the
    decoded (colour matched) result at context size."""
    image = g.out("load_image", 0)
    if req.needs_scaling:
        image = _scale_image(g, "image_scale", image, req.work_width, req.work_height)
    mask = _prepare_mask(g, g.out("load_mask", 0), req, caps)
    image = _prefill(g, image, mask, req, caps)

    g.add("diff", "DifferentialDiffusion", model=model)
    model = g.out("diff")

    use_fooocus = req.model.uses_fooocus and caps.inpaint_nodes
    if use_fooocus:
        g.add(
            "encode",
            "INPAINT_VAEEncodeInpaintConditioning",
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=image,
            mask=mask,
        )
        g.add("fooocus_patch", "INPAINT_LoadFooocusInpaint", head=req.model.fooocus_head, patch=req.model.fooocus_patch)
        g.add(
            "fooocus",
            "INPAINT_ApplyFooocusInpaint",
            model=model,
            patch=g.out("fooocus_patch"),
            latent=g.out("encode", 2),
        )
        model = g.out("fooocus")
        latent = g.out("encode", 3)
    else:
        g.add(
            "encode",
            "InpaintModelConditioning",
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=image,
            mask=mask,
            noise_mask=True,
        )
        latent = g.out("encode", 2)
    positive, negative = g.out("encode", 0), g.out("encode", 1)

    batch = max(1, int(req.sampling.batch))
    if batch > 1:
        g.add("batch", "RepeatLatentBatch", samples=latent, amount=batch)
        latent = g.out("batch")

    g.add(
        "sampler",
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=_sampler_cfg(req),
        sampler_name=req.sampling.sampler,
        scheduler=req.sampling.scheduler,
        denoise=float(req.sampling.denoise),
    )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=vae)
    out = g.out("decode")
    if caps.inpaint_nodes and req.color_match > 0:
        # Correct the global brightness/color drift of the decoded image using
        # the untouched pixels outside the mask as the reference.
        reference = g.out("image_scale") if req.needs_scaling else g.out("load_image", 0)
        g.add(
            "color_match",
            "INPAINT_ColorMatch",
            target=out,
            reference=reference,
            exclude_mask=mask,
            strength=float(req.color_match),
        )
        out = g.out("color_match")
    if req.needs_scaling:
        out = _scale_image(g, "out_scale", out, req.width, req.height)
    return out


def _full_mask(g: Graph, req: InpaintRequest) -> list:
    """The selection mask at context resolution, grown and blurred like the
    model mask of the single pass (needs comfyui-inpaint-nodes)."""
    g.add("fmask_raw", "ImageToMask", image=g.out("load_mask", 0), channel="red")
    grow = max(0, int(getattr(req, "full_grow", 0)))
    blur = 0 if _hard_model_mask(req) else max(0, min(int(getattr(req, "full_blur", 0)), grow))
    g.add("fmask_grow", "INPAINT_ExpandMask", mask=g.out("fmask_raw"), grow=grow, blur=blur, blur_type="gaussian")
    g.add("fmask", "INPAINT_StabilizeMask", mask=g.out("fmask_grow"), epsilon=0.01)
    return g.out("fmask")


def _tile_mask(g: Graph, p: str, full_mask: list, tile, overlap_blur: int) -> list:
    """The full mask cropped to ``tile``, faded out over the parts earlier
    tiles finished (``tile.done``), so the tile repaints only its own area and
    a blend band."""
    r = tile.rect
    g.add(f"{p}_mcrop", "CropMask", mask=full_mask, x=r.x, y=r.y, width=r.width, height=r.height)
    mask = g.out(f"{p}_mcrop")
    if not tile.done:
        return mask
    g.add(f"{p}_keep", "SolidMask", value=1.0, width=r.width, height=r.height)
    keep = g.out(f"{p}_keep")
    for i, d in enumerate(tile.done):
        g.add(f"{p}_done{i}", "SolidMask", value=1.0, width=d.width, height=d.height)
        g.add(
            f"{p}_keep{i}",
            "MaskComposite",
            destination=keep,
            source=g.out(f"{p}_done{i}"),
            x=d.x,
            y=d.y,
            operation="subtract",
        )
        keep = g.out(f"{p}_keep{i}")
    g.add(f"{p}_ramp", "INPAINT_ExpandMask", mask=keep, grow=0, blur=max(3, overlap_blur), blur_type="gaussian")
    g.add(f"{p}_mask", "MaskComposite", destination=mask, source=g.out(f"{p}_ramp"), x=0, y=0, operation="multiply")
    return g.out(f"{p}_mask")


def _scale_mask(g: Graph, p: str, mask: list, width: int, height: int) -> list:
    g.add(f"{p}_mimg", "MaskToImage", mask=mask)
    g.add(
        f"{p}_mscale",
        "ImageScale",
        image=g.out(f"{p}_mimg"),
        upscale_method="bilinear",
        width=width,
        height=height,
        crop="disabled",
    )
    g.add(f"{p}_mback", "ImageToMask", image=g.out(f"{p}_mscale"), channel="red")
    g.add(f"{p}_mstable", "INPAINT_StabilizeMask", mask=g.out(f"{p}_mback"), epsilon=0.01)
    return g.out(f"{p}_mstable")


def _tile_pass(
    g: Graph,
    p: str,
    req: InpaintRequest,
    caps: Capabilities,
    dest: list,
    full_mask: list,
    tile,
    model,
    positive,
    negative,
    vae,
    seed: int,
    denoise: float,
    refine: bool,
    batch: int,
) -> list:
    """One tile of a tiled pass: crop ``dest``, (scale,) pre-fill (generation
    only), sample, decode, (colour match, scale back) and composite back into
    ``dest`` through the tile mask. Returns the new ``dest``."""
    r = tile.rect
    g.add(f"{p}_crop", "ImageCrop", image=dest, width=r.width, height=r.height, x=r.x, y=r.y)
    crop = g.out(f"{p}_crop")
    tmask = _tile_mask(g, p, full_mask, tile, 65)
    scaled = (tile.work_width, tile.work_height) != (r.width, r.height)
    image, mask = crop, tmask
    if scaled:
        image = _scale_image(g, f"{p}_scale", crop, tile.work_width, tile.work_height)
        mask = _scale_mask(g, p, tmask, tile.work_width, tile.work_height)
    reference = image
    if not refine and req.prefill and caps.inpaint_nodes:
        style = "border" if req.prefill_style == "border" else "blur"
        if style == "border":
            mode = req.prefill_mode if req.prefill_mode in ("navier-stokes", "telea", "neutral") else "navier-stokes"
            g.add(f"{p}_prefill", "INPAINT_MaskedFill", image=image, mask=mask, fill=mode, falloff=0)
            g.add(f"{p}_preblur", "INPAINT_MaskedBlur", image=g.out(f"{p}_prefill"), mask=mask, blur=65, falloff=0)
        else:
            g.add(f"{p}_preblur", "INPAINT_MaskedBlur", image=image, mask=mask, blur=65, falloff=9)
        image = g.out(f"{p}_preblur")
    use_fooocus = not refine and req.model.uses_fooocus and caps.inpaint_nodes
    inpaint_ckpt = req.model.arch == "sd15" or req.model.effective_template in (
        "sd15_inpaint",
        "sdxl_eraser",
        "flux_eraser",
    )
    if use_fooocus:
        first = image
        if batch > 1:
            g.add(f"{p}_first", "ImageFromBatch", image=image, batch_index=0, length=1)
            first = g.out(f"{p}_first")
        g.add(
            f"{p}_penc",
            "INPAINT_VAEEncodeInpaintConditioning",
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=first,
            mask=mask,
        )
        g.add(
            f"{p}_patch",
            "INPAINT_ApplyFooocusInpaint",
            model=model,
            patch=g.out("fooocus_patch"),
            latent=g.out(f"{p}_penc", 2),
        )
        tile_model = g.out(f"{p}_patch")
        pos, neg = g.out(f"{p}_penc", 0), g.out(f"{p}_penc", 1)
        if batch > 1:
            g.add(f"{p}_venc", "VAEEncode", pixels=image, vae=vae)
            g.add(f"{p}_lat", "SetLatentNoiseMask", samples=g.out(f"{p}_venc"), mask=mask)
            latent = g.out(f"{p}_lat")
        else:
            latent = g.out(f"{p}_penc", 3)
    elif inpaint_ckpt and refine:
        # Refine on an inpainting checkpoint: its conditioning image shows the
        # whole tile (an empty conditioning mask), so the model reads the
        # generated pixels instead of a grey hole; the noise mask alone limits
        # the change to the regenerated area. With the hole in the
        # conditioning the refine came back greyer and softer.
        g.add(f"{p}_nomask", "SolidMask", value=0.0, width=tile.work_width, height=tile.work_height)
        g.add(
            f"{p}_enc",
            "InpaintModelConditioning",
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=image,
            mask=g.out(f"{p}_nomask"),
            noise_mask=False,
        )
        g.add(f"{p}_lat", "SetLatentNoiseMask", samples=g.out(f"{p}_enc", 2), mask=mask)
        tile_model, pos, neg, latent = model, g.out(f"{p}_enc", 0), g.out(f"{p}_enc", 1), g.out(f"{p}_lat")
    elif inpaint_ckpt:
        g.add(
            f"{p}_enc",
            "InpaintModelConditioning",
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=image,
            mask=mask,
            noise_mask=True,
        )
        tile_model, pos, neg, latent = model, g.out(f"{p}_enc", 0), g.out(f"{p}_enc", 1), g.out(f"{p}_enc", 2)
    else:
        g.add(f"{p}_venc", "VAEEncode", pixels=image, vae=vae)
        g.add(f"{p}_lat", "SetLatentNoiseMask", samples=g.out(f"{p}_venc"), mask=mask)
        tile_model, pos, neg, latent = model, positive, negative, g.out(f"{p}_lat")
    g.add(
        f"{p}_sampler",
        "KSampler",
        model=tile_model,
        positive=pos,
        negative=neg,
        latent_image=latent,
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=_sampler_cfg(req),
        sampler_name=req.sampling.sampler,
        scheduler=req.sampling.scheduler,
        denoise=float(denoise),
    )
    g.add(f"{p}_decode", "VAEDecode", samples=g.out(f"{p}_sampler"), vae=vae)
    out = g.out(f"{p}_decode")
    if not refine and caps.inpaint_nodes and req.color_match > 0:
        g.add(
            f"{p}_cm",
            "INPAINT_ColorMatch",
            target=out,
            reference=reference,
            exclude_mask=mask,
            strength=float(req.color_match),
        )
        out = g.out(f"{p}_cm")
    if scaled:
        out = _scale_image(g, f"{p}_back", out, r.width, r.height)
    g.add(
        f"{p}_paste",
        "ImageCompositeMasked",
        destination=dest,
        source=out,
        x=r.x,
        y=r.y,
        resize_source=False,
        mask=tmask,
    )
    return g.out(f"{p}_paste")


def build_inpaint(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/fill",
) -> dict:
    """Build the inpaint / outpaint graph.

    ``image_ref`` and ``mask_ref`` are the names returned by ``/upload/image``
    (``subfolder/filename``). Returns the API-format prompt dict.

    A ``PlannedInpaintRequest`` with ``tiles`` runs the main pass as a chain of
    overlapping tiles (each sees the result of the earlier ones); with
    ``refine_tiles`` the result is composited over the original at full
    resolution and repainted there at ``refine_denoise``, tile by tile. Both
    need comfyui-inpaint-nodes; without them the single pass is used.
    """
    g = Graph()
    model, clip, vae = _loaders(g, req.model)
    positive, negative = _prompts(g, clip, req.prompt, req.negative)
    if req.model.effective_template == "flux_eraser":
        g.add("guidance", "FluxGuidance", conditioning=positive, guidance=float(req.sampling.cfg or 30.0))
        positive = g.out("guidance")
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    tiles, refine_tiles = _tiles(req), _refine_tiles(req)
    if not caps.inpaint_nodes:
        tiles, refine_tiles = (), ()
    batch = max(1, int(req.sampling.batch))
    tile_denoise = float(getattr(req, "tile_denoise", 1.0))
    layout = bool(tiles) and tile_denoise < 1.0
    if not tiles or layout:
        out = _single_pass(g, req, caps, model, positive, negative, vae, seed)
        if not tiles and not refine_tiles:
            g.add(OUTPUT_NODE, "SaveImage", images=out, filename_prefix=filename_prefix)
            return g.to_dict()
    full_mask = _full_mask(g, req)
    dest = g.out("load_image", 0)
    if batch > 1:
        g.add("dest_batch", "RepeatImageBatch", image=dest, amount=batch)
        dest = g.out("dest_batch")
    g.add("tdiff", "DifferentialDiffusion", model=model)
    diff_model = g.out("tdiff")
    if tiles and not layout:
        if req.model.uses_fooocus:
            g.add(
                "fooocus_patch",
                "INPAINT_LoadFooocusInpaint",
                head=req.model.fooocus_head,
                patch=req.model.fooocus_patch,
            )
        for i, tile in enumerate(tiles):
            dest = _tile_pass(
                g,
                f"t{i}",
                req,
                caps,
                dest,
                full_mask,
                tile,
                diff_model,
                positive,
                negative,
                vae,
                seed,
                float(req.sampling.denoise),
                False,
                batch,
            )
    else:
        g.add(
            "main_paste",
            "ImageCompositeMasked",
            destination=dest,
            source=out,
            x=0,
            y=0,
            resize_source=False,
            mask=full_mask,
        )
        dest = g.out("main_paste")
    if layout:
        # Detail tiles over the layout pass: no pre-fill and the whole tile in
        # the conditioning (the refine path), so they keep its layout.
        for i, tile in enumerate(tiles):
            dest = _tile_pass(
                g,
                f"t{i}",
                req,
                caps,
                dest,
                full_mask,
                tile,
                diff_model,
                positive,
                negative,
                vae,
                seed,
                tile_denoise,
                True,
                batch,
            )
    for i, tile in enumerate(refine_tiles):
        dest = _tile_pass(
            g,
            f"r{i}",
            req,
            caps,
            dest,
            full_mask,
            tile,
            diff_model,
            positive,
            negative,
            vae,
            seed,
            float(getattr(req, "refine_denoise", 0.4)),
            True,
            batch,
        )
    g.add(OUTPUT_NODE, "SaveImage", images=dest, filename_prefix=filename_prefix)
    return g.to_dict()


# Generative Expand prompts. With an empty prompt the positive stays empty and
# the negative steers away from what the model otherwise invents around a
# finished-looking picture: a frame, a border, a vignette or new subjects.
# The old default (a "seamless continuation of the surrounding background"
# positive and no frame terms) made the SDXL tier paint a matted picture frame
# around the old image (docs/evidence/daytime-2026-10-09/expand/README.md).
EXPAND_NEGATIVE = "new objects, people, text, frame, border, vignette, watermark, blurry"
# Appended to a user's own negative: only the frame terms, so a prompt that asks
# for people or text is not contradicted.
EXPAND_FRAME_NEGATIVE = "frame, border, vignette, picture in picture, watermark"


def expand_prompts(prompt: str, negative: str) -> tuple[str, str]:
    """Positive and negative prompt of a Generative Expand.

    Empty prompt ("continue the scene"): empty positive, ``EXPAND_NEGATIVE``
    plus the user's negative. A user prompt is kept and ``EXPAND_FRAME_NEGATIVE``
    is appended to the user's negative.
    """
    prompt, negative = (prompt or "").strip(), (negative or "").strip()
    extra = EXPAND_FRAME_NEGATIVE if prompt else EXPAND_NEGATIVE
    return prompt, f"{negative}, {extra}" if negative else extra


# Guidance of a Generative Expand on the SDXL tier (Fooocus inpaint patch). With
# the empty "continue the scene" positive the guidance only pushes away from the
# negative; at the general default 7 the new border came back over-saturated
# with hard bands (a bright turf strip, grey backdrop bands on a white product
# shot), at 4 (the Fooocus default) it blends. SD 1.5 keeps the setting.
EXPAND_SDXL_CFG = 4.0


def expand_cfg(arch: str, cfg: float) -> float:
    """Sampler guidance for an expand: at most ``EXPAND_SDXL_CFG`` on the SDXL tier."""
    return min(float(cfg), EXPAND_SDXL_CFG) if arch == "sdxl" else float(cfg)


def _is_gguf(name: str) -> bool:
    return name.lower().endswith(".gguf")


def _flux_loaders(g: Graph, spec) -> tuple[list, list, list]:
    """UNet / dual CLIP / VAE loaders for FLUX, GGUF or safetensors as needed."""
    if _is_gguf(spec.unet):
        g.add("unet", "UnetLoaderGGUF", unet_name=spec.unet)
    else:
        g.add("unet", "UNETLoader", unet_name=spec.unet, weight_dtype="fp8_e4m3fn")
    if _is_gguf(spec.clip1) or _is_gguf(spec.clip2):
        g.add("clip", "DualCLIPLoaderGGUF", clip_name1=spec.clip1, clip_name2=spec.clip2, type="flux")
    else:
        g.add("clip", "DualCLIPLoader", clip_name1=spec.clip1, clip_name2=spec.clip2, type="flux")
    g.add("vae", "VAELoader", vae_name=spec.vae)
    return g.out("unet"), g.out("clip"), g.out("vae")


def _load_inputs(g: Graph, req: InpaintRequest, image_ref: str, mask_ref: str, caps: Capabilities) -> tuple[list, list]:
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    image = g.out("load_image", 0)
    if req.needs_scaling:
        image = _scale_image(g, "image_scale", image, req.work_width, req.work_height)
    mask = _prepare_mask(g, g.out("load_mask", 0), req, caps)
    return image, mask


def _finish(g: Graph, req: InpaintRequest, out: list, mask: list, caps: Capabilities, filename_prefix: str) -> dict:
    if caps.inpaint_nodes and req.color_match > 0:
        reference = g.out("image_scale") if req.needs_scaling else g.out("load_image", 0)
        g.add(
            "color_match",
            "INPAINT_ColorMatch",
            target=out,
            reference=reference,
            exclude_mask=mask,
            strength=float(req.color_match),
        )
        out = g.out("color_match")
    if req.needs_scaling:
        out = _scale_image(g, "out_scale", out, req.width, req.height)
    g.add(OUTPUT_NODE, "SaveImage", images=out, filename_prefix=filename_prefix)
    return g.to_dict()


def build_flux_fill(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/flux",
) -> dict:
    """FLUX.1 Fill: UNet (GGUF or fp8) + T5/CLIP-L + FLUX VAE, InpaintModelConditioning,
    FluxGuidance, cfg 1.0."""
    g = Graph()
    model, clip, vae = _flux_loaders(g, req.model)
    positive, negative = _prompts(g, clip, req.prompt, req.negative)
    g.add("guidance", "FluxGuidance", conditioning=positive, guidance=float(req.sampling.cfg or 30.0))
    positive = g.out("guidance")
    image, mask = _load_inputs(g, req, image_ref, mask_ref, caps)
    image = _prefill(g, image, mask, req, caps)
    g.add("diff", "DifferentialDiffusion", model=model)
    g.add(
        "encode",
        "InpaintModelConditioning",
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=image,
        mask=mask,
        noise_mask=True,
    )
    latent = g.out("encode", 2)
    batch = max(1, int(req.sampling.batch))
    if batch > 1:
        g.add("batch", "RepeatLatentBatch", samples=latent, amount=batch)
        latent = g.out("batch")
    g.add(
        "sampler",
        "KSampler",
        model=g.out("diff"),
        positive=g.out("encode", 0),
        negative=g.out("encode", 1),
        latent_image=latent,
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=float(req.sampling.denoise),
    )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=vae)
    return _finish(g, req, g.out("decode"), mask, caps, filename_prefix)


def build_qwen_edit(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/qwen",
) -> dict:
    """Qwen-Image-Edit (2509/2511 class): instruction-driven edit of the context
    region, limited to the mask with a latent noise mask."""
    g = Graph()
    spec = req.model
    if _is_gguf(spec.unet):
        g.add("unet", "UnetLoaderGGUF", unet_name=spec.unet)
    else:
        g.add("unet", "UNETLoader", unet_name=spec.unet, weight_dtype="default")
    if _is_gguf(spec.clip1):
        g.add("clip", "CLIPLoaderGGUF", clip_name=spec.clip1, type="qwen_image")
    else:
        g.add("clip", "CLIPLoader", clip_name=spec.clip1, type="qwen_image")
    g.add("vae", "VAELoader", vae_name=spec.vae)
    model, clip, vae = g.out("unet"), g.out("clip"), g.out("vae")
    g.add("shift", "ModelSamplingAuraFlow", model=model, shift=3.0)
    g.add("cfgnorm", "CFGNorm", model=g.out("shift"), strength=1.0)
    model = g.out("cfgnorm")
    image, mask = _load_inputs(g, req, image_ref, mask_ref, caps)
    g.add("pos", "TextEncodeQwenImageEditPlus", clip=clip, prompt=req.prompt, vae=vae, image1=image)
    g.add("neg", "TextEncodeQwenImageEditPlus", clip=clip, prompt=req.negative or "", vae=vae, image1=image)
    g.add("encode", "VAEEncode", pixels=image, vae=vae)
    g.add("noise_mask", "SetLatentNoiseMask", samples=g.out("encode"), mask=mask)
    latent = g.out("noise_mask")
    batch = max(1, int(req.sampling.batch))
    if batch > 1:
        g.add("batch", "RepeatLatentBatch", samples=latent, amount=batch)
        latent = g.out("batch")
    g.add(
        "sampler",
        "KSampler",
        model=model,
        positive=g.out("pos"),
        negative=g.out("neg"),
        latent_image=latent,
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=float(req.sampling.cfg if 1.0 < req.sampling.cfg <= 5.0 else 2.5),
        sampler_name="euler",
        scheduler="simple",
        denoise=float(req.sampling.denoise),
    )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=vae)
    return _finish(g, req, g.out("decode"), mask, caps, filename_prefix)


QWEN21_RESOLUTION = 1024
QWEN21_LIGHTNING_SIGMAS = "1.0, 0.95418, 0.89508, 0.704534, 0.0"


def _edit_image_slots(g: Graph, target: list, reference_refs: tuple[str, ...]) -> dict:
    """Build native autogrow inputs with the edit target always in slot 1."""
    slots = {"images.image_1": target}
    for index, image in enumerate(reference_refs, start=2):
        node = g.add(f"reference_{index}", "LoadImage", image=image)
        slots[f"images.image_{index}"] = g.out(node)
    return slots


def qwen21_reference_size(width: int, height: int) -> tuple[int, int]:
    """Match the first image's latent size in native TextEncodeQwenImage21."""
    if min(width, height) < 1:
        raise ValueError("Qwen Image 2.1 needs positive image dimensions")
    ratio = width / height
    area = QWEN21_RESOLUTION * QWEN21_RESOLUTION
    return max(32, round(math.sqrt(area * ratio) / 32) * 32), max(32, round(math.sqrt(area / ratio) / 32) * 32)


def build_qwen21_edit(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/qwen21-edit",
    *,
    reference_refs: tuple[str, ...] = (),
) -> dict:
    """Native Qwen 2.1 instruction editing, clipped to the PhotoCraft selection."""
    if req.mode != "fill" or not req.prompt.strip():
        raise ValueError("Write an edit instruction for Qwen Image 2.1")
    if not all((req.model.unet, req.model.clip1, req.model.vae)):
        raise ValueError("Qwen Image 2.1 needs its diffusion model, Qwen3-VL encoder, and VAE")
    width, height = qwen21_reference_size(req.width, req.height)
    aligned = replace(req, work_width=width, work_height=height, color_match=0.0)
    g = Graph()
    g.add("unet", "UNETLoader", unet_name=req.model.unet, weight_dtype="default")
    g.add("clip", "CLIPLoader", clip_name=req.model.clip1, type="qwen_image", device="default")
    g.add("vae", "VAELoader", vae_name=req.model.vae)
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    g.add("selection_mask", "ImageToMask", image=g.out("load_mask"), channel="red")
    g.add("selection_support", "ThresholdMask", mask=g.out("selection_mask"), value=0.0)
    g.add(
        "encode",
        "TextEncodeQwenImage21",
        clip=g.out("clip"),
        vae=g.out("vae"),
        prompt=req.prompt,
        negative_prompt="",
        resolution=QWEN21_RESOLUTION,
        **_edit_image_slots(g, g.out("load_image"), reference_refs),
    )
    g.add("cache", "QwenImage21Cache", model=g.out("unet"), device="auto", dtype="default")
    latent = g.out("encode", 2)
    batch = max(1, int(req.sampling.batch))
    if batch > 1:
        g.add("latent_batch", "RepeatLatentBatch", samples=latent, amount=batch)
        latent = g.out("latent_batch")
    if req.model.qwen21_lightning:
        # The publisher's sigmas are already shifted. A four-step standard
        # scheduler would use different sigmas and change the edit behavior.
        g.add("guider", "BasicGuider", model=g.out("cache"), conditioning=g.out("encode", 0))
        g.add("sigmas", "ManualSigmas", sigmas=QWEN21_LIGHTNING_SIGMAS)
        g.add("sampler_select", "KSamplerSelect", sampler_name="euler")
        g.add("noise", "RandomNoise", noise_seed=int(seed))
        g.add(
            "sampler",
            "SamplerCustomAdvanced",
            noise=g.out("noise"),
            guider=g.out("guider"),
            sampler=g.out("sampler_select"),
            sigmas=g.out("sigmas"),
            latent_image=latent,
        )
    else:
        g.add(
            "sampler",
            "KSampler",
            model=g.out("cache"),
            positive=g.out("encode", 0),
            negative=g.out("encode", 1),
            latent_image=latent,
            seed=int(seed),
            steps=int(req.sampling.steps),
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=g.out("vae"))
    _finish(g, aligned, g.out("decode"), g.out("selection_support"), caps, filename_prefix)
    generated = g.nodes[OUTPUT_NODE]["inputs"]["images"]
    destination = g.out("load_image")
    if batch > 1:
        g.add("source_batch", "RepeatImageBatch", image=destination, amount=batch)
        destination = g.out("source_batch")
    # The graph protects the exterior with hard support. PhotoCraft applies
    # its captured soft selection once when it adds the result layer.
    g.add(
        "selection_composite",
        "ImageCompositeMasked",
        destination=destination,
        source=generated,
        x=0,
        y=0,
        resize_source=False,
        mask=g.out("selection_support"),
    )
    g.nodes[OUTPUT_NODE]["inputs"]["images"] = g.out("selection_composite")
    return g.to_dict()


SDXL_EDIT_TEMPLATES = ("sdxl_edit", "sdxl_inpaint_edit")
SELECTION_EDIT_TEMPLATES = ("fibo2_edit", QWEN21_EDIT_TEMPLATE, *SDXL_EDIT_TEMPLATES)


def build_sdxl_edit(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/sdxl-edit",
) -> dict:
    """Masked SDXL editing using core nodes and the checkpoint's own CLIP/VAE.

    Header-verified regular (four input channels) checkpoints use ordinary
    image-to-image latents plus a noise mask. Native inpainting checkpoints
    (nine input channels) additionally need the masked-image conditioning from
    InpaintModelConditioning. Neither path installs or applies Fooocus patches.
    A binary support mask controls generation; PhotoCraft applies the original
    soft selection to the returned layer once, preserving its alpha falloff.
    """
    if req.mode != "fill":
        raise ValueError("This SDXL integration supports selected-region editing; Expand is not enabled")
    if req.model.arch != "sdxl" or req.model.effective_template not in SDXL_EDIT_TEMPLATES:
        raise ValueError("SDXL editing requires a verified regular or native inpainting SDXL checkpoint")
    if not req.model.checkpoint:
        raise ValueError("Choose an installed SDXL checkpoint for editing")
    if not req.prompt.strip():
        raise ValueError("Describe the content you want in the selected area for SDXL")
    if min(req.width, req.height, req.work_width, req.work_height) < 1:
        raise ValueError("SDXL editing needs positive image dimensions")
    if not 0.0 <= float(req.sampling.denoise) <= 1.0:
        raise ValueError("SDXL edit strength must be between zero and one")
    width = max(8, ((int(req.work_width) + 7) // 8) * 8)
    height = max(8, ((int(req.work_height) + 7) // 8) * 8)
    aligned = replace(req, work_width=width, work_height=height, color_match=0.0)
    g = Graph()
    model, clip, vae = _checkpoint(g, req.model.checkpoint)
    positive, negative = _prompts(g, clip, req.prompt, req.negative or "")
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    g.add("selection_mask", "ImageToMask", image=g.out("load_mask"), channel="red")
    g.add("selection_support", "ThresholdMask", mask=g.out("selection_mask"), value=0.0)
    image, mask = g.out("load_image"), g.out("selection_support")
    if aligned.needs_scaling:
        image = _scale_image(g, "image_scale", image, width, height)
        # Scale the hard support without inventing a second feathered edge.
        g.add("work_mask_image", "MaskToImage", mask=mask)
        g.add(
            "work_mask_scale",
            "ImageScale",
            image=g.out("work_mask_image"),
            upscale_method="nearest-exact",
            width=width,
            height=height,
            crop="disabled",
        )
        g.add("work_mask", "ImageToMask", image=g.out("work_mask_scale"), channel="red")
        mask = g.out("work_mask")
    if int(req.grow) > 0:
        # Give the sampler context around the seam. The final composite and
        # editor mask still expose only pixels from the original selection.
        g.add("model_mask_grow", "GrowMask", mask=mask, expand=int(req.grow), tapered_corners=True)
        mask = g.out("model_mask_grow")
    if req.model.effective_template == "sdxl_inpaint_edit":
        g.add(
            "encode",
            "InpaintModelConditioning",
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=image,
            mask=mask,
            noise_mask=True,
        )
        positive, negative, latent = g.out("encode", 0), g.out("encode", 1), g.out("encode", 2)
    else:
        g.add("encode", "VAEEncode", pixels=image, vae=vae)
        g.add("noise_mask", "SetLatentNoiseMask", samples=g.out("encode"), mask=mask)
        latent = g.out("noise_mask")
    batch = max(1, int(req.sampling.batch))
    if batch > 1:
        g.add("batch", "RepeatLatentBatch", samples=latent, amount=batch)
        latent = g.out("batch")
    g.add(
        "sampler",
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=float(req.sampling.cfg),
        sampler_name=req.sampling.sampler,
        scheduler=req.sampling.scheduler,
        denoise=float(req.sampling.denoise),
    )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=vae)
    _finish(g, aligned, g.out("decode"), g.out("selection_support"), caps, filename_prefix)
    generated = g.nodes[OUTPUT_NODE]["inputs"]["images"]
    destination = g.out("load_image")
    if batch > 1:
        g.add("source_batch", "RepeatImageBatch", image=destination, amount=batch)
        destination = g.out("source_batch")
    g.add(
        "selection_composite",
        "ImageCompositeMasked",
        destination=destination,
        source=generated,
        x=0,
        y=0,
        resize_source=False,
        mask=g.out("selection_support"),
    )
    g.nodes[OUTPUT_NODE]["inputs"]["images"] = g.out("selection_composite")
    return g.to_dict()


@lru_cache(maxsize=64)
def fibo_trained_size(source_width: int, source_height: int) -> tuple[int, int]:
    """Native FIBO reference size, preserving its multi-reference alignment.

    TextEncodeFibo2 independently sizes multiple references to about 1 MP,
    even when an explicit output size was supplied. Choose that same size
    before resizing without cropping, so original and selection guide cannot
    be center-cropped differently from the image we later paste into PhotoCraft.
    This is the native node's trained_size ordering, including its tie breaks.
    """
    if min(source_width, source_height) < 1:
        raise ValueError("FIBO reference dimensions must be positive")
    aspect_ratio = source_width / source_height
    area = 1024 * 1024
    sizes = (
        (w, h)
        for w in range(64, 2 * area // 64, 16)
        for h in range(
            max(64, 16 * math.ceil(0.9 * area / w / 16)),
            16 * math.floor(1.1 * area / w / 16) + 1,
            16,
        )
    )
    return min(
        sizes,
        key=lambda size: (
            round(1 - min(size[0] / size[1] / aspect_ratio, aspect_ratio * size[1] / size[0]), 9),
            abs(size[0] * size[1] - area),
            -size[0],
        ),
    )


def fibo_edit_prompt(prompt: str, mode: str) -> str:
    """Add mode-specific guidance while retaining the user's complete edit."""
    if mode == "background":
        return (
            "Regenerate the gray selected area in <image_1> with an empty, seamless continuation of the "
            "adjacent background. Extend all background detail, lighting and texture from its edges, "
            "maintaining the natural perspective and depth of the scene. Leave the background clear."
        )
    if mode == "preserve":
        return (
            prompt.strip() + " Keep the existing structure, surfaces, frames, camera angle and perspective unchanged "
            "except for the requested edit. Use existing openings in their original positions. "
            "Size any added subject to fit the relevant opening and surrounding scene."
        )
    if mode == "replace":
        return (
            "Regenerate the gray selected area in <image_1> with this requested content: "
            + prompt.strip()
            + "\nUse the visible surroundings as reference. Match their perspective, scale, lighting, "
            "materials and texture, continuing surfaces and outlines at the boundary. Keep the unmasked "
            "surroundings unchanged. Replace the gray area completely; do not leave a gray patch."
        )
    raise ValueError("Unknown FIBO edit mode")


def build_fibo2_edit(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/fibo2",
    *,
    reference_refs: tuple[str, ...] = (),
) -> dict:
    """Selection-aware FIBO editing with original exterior and one alpha blend.

    Replace uses FIBO's trained grey-region reference preparation and its native
    empty latent. Preserve is the tested source-latent editing preset: retain
    the intact reference, encode the same source as the sampler's initial
    latent, and restrict denoising to selection support at strength 0.75.
    Both retain native FIBO positive/negative conditioning. The intact source
    stays separate for compositing; PhotoCraft applies soft alpha exactly once.
    """
    if req.mode != "fill":
        raise ValueError("FIBO 2.0 Edit supports editing a selection or the whole image; Expand is not supported")
    if not req.prompt.strip():
        raise ValueError("Write an edit instruction for FIBO 2.0 Edit")
    if not all((req.model.unet, req.model.clip1, req.model.vae)):
        raise ValueError("FIBO 2.0 Edit needs its diffusion model, Qwen3-VL text encoder and VAE")
    if _is_gguf(req.model.unet) or _is_gguf(req.model.clip1):
        raise ValueError("This FIBO 2.0 Edit integration requires native safetensors models")
    diffusion_name = req.model.unet.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if diffusion_name not in ("fibo2_edit_turbo_nvfp4.safetensors", "fibo2_dev_nvfp4.safetensors"):
        raise ValueError("PhotoCraft FIBO 2.0 editing requires the NVFP4 Dev or Edit Turbo diffusion model")
    if min(req.width, req.height, req.work_width, req.work_height) < 1:
        raise ValueError("FIBO 2.0 Edit needs positive image dimensions")
    edit_prompt = fibo_edit_prompt(req.prompt, req.fibo_edit_mode)

    # Multi-reference encoding picks its own trained size. Align to it up front
    # with no crop; both modes then have the same size for fair comparisons.
    width, height = fibo_trained_size(req.width, req.height)
    aligned = replace(req, work_width=width, work_height=height, color_match=0.0)
    g = Graph()
    g.add("unet", "UNETLoader", unet_name=req.model.unet, weight_dtype="default")
    g.add("clip", "CLIPLoader", clip_name=req.model.clip1, type="fibo2")
    g.add("vae", "VAELoader", vae_name=req.model.vae)
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    image = g.out("load_image")
    if aligned.needs_scaling:
        image = _scale_image(g, "image_scale", image, width, height)
    g.add("selection_mask", "ImageToMask", image=g.out("load_mask"), channel="red")
    # The user can feather or select a one-pixel line. Keep every selected pixel
    # in the guide; thresholding the already feathered mask at 0.5 would erase
    # thin selections. The resulting model guide support is binary.
    g.add("selection_support", "ThresholdMask", mask=g.out("selection_mask"), value=0.0)
    guide_mask = g.out("selection_support")
    if aligned.needs_scaling:
        g.add("guide_mask_image", "MaskToImage", mask=guide_mask)
        g.add(
            "guide_mask_scale",
            "ImageScale",
            image=g.out("guide_mask_image"),
            upscale_method="nearest-exact",
            width=width,
            height=height,
            crop="disabled",
        )
        g.add("guide_mask", "ImageToMask", image=g.out("guide_mask_scale"), channel="red")
        guide_mask = g.out("guide_mask")
    reference = image
    if req.fibo_edit_mode in ("replace", "background"):
        g.add("guide_gray", "EmptyImage", width=width, height=height, batch_size=1, color=0x808080)
        g.add(
            "selection_guide",
            "ImageCompositeMasked",
            destination=image,
            source=g.out("guide_gray"),
            x=0,
            y=0,
            resize_source=False,
            mask=guide_mask,
        )
        reference = g.out("selection_guide")
    g.add(
        "encode",
        "TextEncodeFibo2",
        clip=g.out("clip"),
        prompt=edit_prompt,
        negative_prompt=req.negative or "",
        vae=g.out("vae"),
        width=width,
        height=height,
        batch_size=max(1, int(req.sampling.batch)),
        **_edit_image_slots(g, reference, reference_refs),
    )
    sampler_latent = g.out("encode", 2)
    denoise = 1.0
    if req.fibo_edit_mode == "preserve":
        # The native FIBO VAE and denoiser share Flux2's 128-channel /16 latent
        # format. Encode the intact source for the structure-preserving preset.
        g.add("original_latent", "VAEEncode", pixels=image, vae=g.out("vae"))
        sampler_latent = g.out("original_latent")
        if int(req.sampling.batch) > 1:
            # TextEncodeFibo2.batch_size only affects its unused empty latent.
            # Repeat source samples here so each variation gets its own noise
            # from KSampler's normal batch generation, with no fixed batch index.
            g.add("latent_batch", "RepeatLatentBatch", samples=sampler_latent, amount=int(req.sampling.batch))
            sampler_latent = g.out("latent_batch")
        g.add("latent_mask", "SetLatentNoiseMask", samples=sampler_latent, mask=guide_mask)
        sampler_latent = g.out("latent_mask")
        denoise = 0.75
    g.add(
        "sampler",
        "KSampler",
        model=g.out("unet"),
        positive=g.out("encode", 0),
        negative=g.out("encode", 1),
        latent_image=sampler_latent,
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=float(req.sampling.cfg),
        sampler_name="euler",
        scheduler="simple",
        denoise=denoise,
    )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=g.out("vae"))
    # Hard support protects the original exterior. The soft alpha is applied
    # later by PhotoCraft; blending it here too would square that alpha.
    _finish(g, aligned, g.out("decode"), g.out("selection_mask"), caps, filename_prefix)
    generated = g.nodes[OUTPUT_NODE]["inputs"]["images"]
    destination = g.out("load_image")
    if int(req.sampling.batch) > 1:
        # ImageCompositeMasked repeats/truncates its source to the destination
        # batch size. Repeat the original or all but the first variation vanish.
        g.add("source_batch", "RepeatImageBatch", image=destination, amount=int(req.sampling.batch))
        destination = g.out("source_batch")
    g.add(
        "selection_composite",
        "ImageCompositeMasked",
        destination=destination,
        source=generated,
        x=0,
        y=0,
        resize_source=False,
        mask=g.out("selection_support"),
    )
    g.nodes[OUTPUT_NODE]["inputs"]["images"] = g.out("selection_composite")
    return g.to_dict()


# The prompts erase-trained models run with (positive, negative). RORem's authors found
# content-free quality words help; the FLUX Fill removal LoRA needs no prompt.
ERASER_PROMPTS = {
    "sdxl_eraser": (
        "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic",
        "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background",
    ),
    "flux_eraser": ("", ""),
}

# OpenFill's own node (comfy_nodes): LaMa at the image's own resolution.
NATIVE_LAMA_NODE = "OpenFillLamaNative"
NATIVE_LAMA_OVERLAP = 256  # px of overlap between windows when one pass does not fit


def build_removal(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/remove",
) -> dict:
    """Prompt-free removal with LaMa / MAT. LaMa runs through OpenFill's own node at the
    crop's own resolution (one pass, or large overlapping windows at 1:1 when the
    graphics memory is short; see comfy_nodes); MAT, or LaMa on a server without that
    node, through comfyui-inpaint-nodes' Inpaint (using Model)."""
    if not caps.inpaint_nodes:
        raise ValueError("prompt-free removal needs comfyui-inpaint-nodes on the server")
    g = Graph()
    if req.model.native_lama and caps.native_lama and not _tiles(req):
        image, mask = _load_inputs(g, req, image_ref, mask_ref, caps)
        g.add("inpaint_model", "INPAINT_LoadInpaintModel", model_name=req.model.inpaint_model)
        g.add(
            "remove",
            NATIVE_LAMA_NODE,
            inpaint_model=g.out("inpaint_model"),
            image=image,
            mask=mask,
            seed=int(seed),
            max_megapixels=0.0,
            overlap=NATIVE_LAMA_OVERLAP,
        )
        out = g.out("remove")
        if req.needs_scaling:
            out = _scale_image(g, "out_scale", out, req.width, req.height)
        g.add(OUTPUT_NODE, "SaveImage", images=out, filename_prefix=filename_prefix)
        return g.to_dict()
    if _tiles(req):
        out = _removal_tiles(g, req, image_ref, mask_ref, seed)
        g.add(OUTPUT_NODE, "SaveImage", images=out, filename_prefix=filename_prefix)
        return g.to_dict()
    image, mask = _load_inputs(g, req, image_ref, mask_ref, caps)
    g.add("inpaint_model", "INPAINT_LoadInpaintModel", model_name=req.model.inpaint_model)
    g.add(
        "remove",
        "INPAINT_InpaintWithModel",
        inpaint_model=g.out("inpaint_model"),
        image=image,
        mask=mask,
        seed=int(seed),
    )
    out = g.out("remove")
    if req.needs_scaling:
        out = _scale_image(g, "out_scale", out, req.width, req.height)
    refine_model = getattr(req, "refine_model", None)
    refine_tiles = _refine_tiles(req)
    if refine_model is not None and refine_tiles:
        out = _removal_refine(g, req, caps, out, refine_model, refine_tiles, seed)
    g.add(OUTPUT_NODE, "SaveImage", images=out, filename_prefix=filename_prefix)
    return g.to_dict()


# Blend band (px) of a removal tile into the parts earlier tiles finished; it
# lies inside the band the tile repaints (half the overlap).
REMOVAL_BLEND = 24


def _removal_tiles(g: Graph, req: InpaintRequest, image_ref: str, mask_ref: str, seed: int) -> list:
    """Tiled removal at 1:1 (``geometry.plan_removal``): each window of the
    context goes through the removal node on its own, with the parts earlier
    windows finished as known pixels, and is blended back over a short ramp."""
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    full_mask = _full_mask(g, req)
    g.add("inpaint_model", "INPAINT_LoadInpaintModel", model_name=req.model.inpaint_model)
    dest = g.out("load_image", 0)
    for i, tile in enumerate(_tiles(req)):
        p = f"rm{i}"
        r = tile.rect
        g.add(f"{p}_crop", "ImageCrop", image=dest, width=r.width, height=r.height, x=r.x, y=r.y)
        g.add(f"{p}_mcrop", "CropMask", mask=full_mask, x=r.x, y=r.y, width=r.width, height=r.height)
        hole, blend = g.out(f"{p}_mcrop"), g.out(f"{p}_mcrop")
        if tile.done:
            g.add(f"{p}_done", "SolidMask", value=0.0, width=r.width, height=r.height)
            done = g.out(f"{p}_done")
            for j, d in enumerate(tile.done):
                g.add(f"{p}_d{j}", "SolidMask", value=1.0, width=d.width, height=d.height)
                g.add(
                    f"{p}_dadd{j}",
                    "MaskComposite",
                    destination=done,
                    source=g.out(f"{p}_d{j}"),
                    x=d.x,
                    y=d.y,
                    operation="add",
                )
                done = g.out(f"{p}_dadd{j}")
            g.add(f"{p}_hole", "MaskComposite", destination=hole, source=done, x=0, y=0, operation="subtract")
            hole = g.out(f"{p}_hole")
            # ramp from 0 at the finished part to 1 inside the repainted band
            g.add(
                f"{p}_dgrow",
                "INPAINT_ExpandMask",
                mask=done,
                grow=REMOVAL_BLEND // 2,
                blur=REMOVAL_BLEND + 1,
                blur_type="gaussian",
            )
            g.add(
                f"{p}_blend",
                "MaskComposite",
                destination=hole,
                source=g.out(f"{p}_dgrow"),
                x=0,
                y=0,
                operation="subtract",
            )
            blend = g.out(f"{p}_blend")
        g.add(
            f"{p}_remove",
            "INPAINT_InpaintWithModel",
            inpaint_model=g.out("inpaint_model"),
            image=g.out(f"{p}_crop"),
            mask=hole,
            seed=int(seed),
        )
        g.add(
            f"{p}_paste",
            "ImageCompositeMasked",
            destination=dest,
            source=g.out(f"{p}_remove"),
            x=r.x,
            y=r.y,
            resize_source=False,
            mask=blend,
        )
        dest = g.out(f"{p}_paste")
    return dest


def _removal_refine(
    g: Graph, req: InpaintRequest, caps: Capabilities, removed: list, spec: ModelSpec, tiles, seed: int
) -> list:
    """Repaint the removal model's (smooth, low-resolution) fill at full
    resolution with a generation checkpoint at ``refine_denoise``."""
    from dataclasses import replace as dc_replace

    model, clip, vae = _checkpoint(g, spec.checkpoint)
    positive, negative = _prompts(g, clip, getattr(req, "refine_prompt", ""), getattr(req, "refine_negative", ""))
    full_mask = _full_mask(g, req)
    g.add("tdiff", "DifferentialDiffusion", model=model)
    dest = removed
    rreq = dc_replace(req, model=spec, mode="fill")
    for i, tile in enumerate(tiles):
        dest = _tile_pass(
            g,
            f"r{i}",
            rreq,
            caps,
            dest,
            full_mask,
            tile,
            g.out("tdiff"),
            positive,
            negative,
            vae,
            seed,
            float(getattr(req, "refine_denoise", 0.4)),
            True,
            1,
        )
    return dest


def build_lama_rmbg(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/remove",
) -> dict:
    """Run the server's Big-LaMa node with the prepared removal mask."""
    g = Graph()
    g.add("load_image", "LoadImage", image=image_ref)
    g.add("load_mask", "LoadImage", image=mask_ref)
    image, mask_image = g.out("load_image"), g.out("load_mask")
    if req.needs_scaling:
        image = _scale_image(g, "image_scale", image, req.work_width, req.work_height)
        mask_image = _scale_image(g, "mask_scale", mask_image, req.work_width, req.work_height)
    g.add("mask", "ImageToMask", image=mask_image, channel="red")
    g.add("support", "ThresholdMask", mask=g.out("mask"), value=0.0)
    g.add("remove", "AILab_LamaRemover", images=image, masks=g.out("support"), removal_strength=128, edge_smoothness=0)
    result = g.out("remove")
    if req.needs_scaling:
        result = _scale_image(g, "restore", result, req.width, req.height)
    g.add(OUTPUT_NODE, "SaveImage", images=result, filename_prefix=filename_prefix)
    return g.to_dict()


BUILDERS = {
    "lama_rmbg": build_lama_rmbg,
    "sd15_inpaint": build_inpaint,
    "sdxl_fooocus": build_inpaint,
    "flux_fill": build_flux_fill,
    "qwen_edit": build_qwen_edit,
    QWEN21_EDIT_TEMPLATE: build_qwen21_edit,
    "sdxl_edit": build_sdxl_edit,
    "sdxl_inpaint_edit": build_sdxl_edit,
    "fibo2_edit": build_fibo2_edit,
    "inpaint_model": build_removal,
    "sdxl_eraser": build_inpaint,
    "flux_eraser": build_inpaint,
}


def select_builder(template: str):
    """Return the graph builder for a catalog template id."""
    try:
        return BUILDERS[template]
    except KeyError:
        raise ValueError(f"unknown workflow template '{template}'") from None


def build_fill(
    req: InpaintRequest,
    image_ref: str,
    mask_ref: str,
    seed: int,
    caps: Capabilities,
    filename_prefix: str = "openfill/fill",
    *,
    reference_refs: tuple[str, ...] = (),
) -> dict:
    """Dispatch on the active model's template."""
    template = req.model.effective_template
    builder = select_builder(template)
    if req.reference_images and len(reference_refs) != len(req.reference_images):
        raise ValueError("The uploaded reference list does not match the edit request.")
    if reference_refs:
        validate_edit_references(template, len(reference_refs))
        return builder(req, image_ref, mask_ref, seed, caps, filename_prefix, reference_refs=reference_refs)
    return builder(req, image_ref, mask_ref, seed, caps, filename_prefix)


def build_txt2img(req: Txt2ImgRequest, seed: int, filename_prefix: str = "openfill/txt2img") -> dict:
    g = Graph()
    if req.model.effective_template == "flux_fill":
        raise ValueError("FLUX.1 Fill is an inpainting model; use it for Generative Fill or Expand")
    if req.model.effective_template in ("qwen_edit", "inpaint_model", *SELECTION_EDIT_TEMPLATES):
        raise ValueError("the active model does not support text to image")
    model, clip, vae = _checkpoint(g, req.model.checkpoint)
    positive, negative = _prompts(g, clip, req.prompt, req.negative)
    g.add(
        "latent",
        "EmptyLatentImage",
        width=int(req.width),
        height=int(req.height),
        batch_size=max(1, int(req.sampling.batch)),
    )
    g.add(
        "sampler",
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=g.out("latent"),
        seed=int(seed),
        steps=int(req.sampling.steps),
        cfg=float(req.sampling.cfg),
        sampler_name=req.sampling.sampler,
        scheduler=req.sampling.scheduler,
        denoise=1.0,
    )
    g.add("decode", "VAEDecode", samples=g.out("sampler"), vae=vae)
    g.add(OUTPUT_NODE, "SaveImage", images=g.out("decode"), filename_prefix=filename_prefix)
    return g.to_dict()


def total_steps(graph: dict) -> int:
    """Sum of sampler steps in a graph (for progress estimation)."""
    steps = 0
    for node in graph.values():
        if node["class_type"] in ("KSampler", "KSamplerAdvanced"):
            steps += int(node["inputs"].get("steps", 0))
        elif node["class_type"] == "SamplerCustomAdvanced":
            link = node["inputs"].get("sigmas")
            sigmas = graph.get(link[0], {}) if link else {}
            if sigmas.get("class_type") == "ManualSigmas":
                steps += len(sigmas["inputs"]["sigmas"].split(",")) - 1
    return steps


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# ComfyUI's LoadImage decodes uploads with PyAV (FFmpeg), which guesses the file
# format from the first 2048 bytes. A mask with identical rows (an outpaint
# strip) deflates into a periodic byte stream that the MPEG-TS probe scores
# higher than PNG, and the load fails with "End of file". A text chunk of
# spaces right after IHDR fills the probe window, so only the PNG signature
# is recognisable there.
PROBE_PADDING = 2048


def probe_safe_png(data: bytes) -> bytes:
    """Return ``data`` with a tEXt chunk of ``PROBE_PADDING`` spaces inserted after IHDR.

    Decoders ignore the chunk; the pixels are unchanged. Input that is not a PNG
    starting with IHDR, or already padded, is returned as is.
    """
    if not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return data
    end_ihdr = 8 + 12 + int.from_bytes(data[8:12], "big")
    if data[end_ihdr + 4 : end_ihdr + 8] == b"tEXt":
        return data
    body = b"Comment\x00" + b" " * PROBE_PADDING
    chunk = len(body).to_bytes(4, "big") + b"tEXt" + body + (zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF).to_bytes(4, "big")
    return data[:end_ihdr] + chunk + data[end_ihdr:]


def model_display_name(name: str) -> str:
    """The user-facing part of a catalog model name.

    Catalog names carry packaging details for the Models tab, e.g. "FLUX.1 Fill
    [dev] (first-party safetensors, fp8 at load time)" or "MAT Places512 (fp16)
    - prompt-free object removal". Messages about a missing or running model
    only need the model itself: everything from the first " (" or " - " on is
    dropped. A name that would become empty is returned unchanged.
    """
    text = (name or "").strip()
    for marker in (" (", " - "):
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    return text.strip() or (name or "").strip()


def describe_node_errors(node_errors: dict, graph: dict) -> str:
    """Turn ComfyUI ``node_errors`` into one readable line per problem."""
    lines = []
    for node_id, info in (node_errors or {}).items():
        class_type = graph.get(node_id, {}).get("class_type", node_id)
        for err in info.get("errors", []):
            msg = err.get("message", "")
            details = err.get("details", "")
            extra = err.get("extra_info", {}) or {}
            name = extra.get("input_name")
            text = f"{class_type}: {msg}"
            if name:
                text += f" ({name})"
            if details:
                text += f" - {details}"
            lines.append(text)
    return "\n".join(lines)
