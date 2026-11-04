"""Verified existing SDXL checkpoints, matched to a host's live loader options.

This is an allowlist of inspected full checkpoints, not filename-based model
classification. Unknown files, refiners and LoRAs are never offered as editors.
New records require a safetensors-header audit and sampling compatibility check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .backend import BackendUnavailable
from .catalog import FileSpec, ModelEntry
from .comfy_api import ComfyClient
from .sources import SOURCES


@dataclass(frozen=True)
class VerifiedSDXL:
    source_id: str
    checkpoint: str  # complete loader path, relative to checkpoints/
    model_id: str  # stable across hosts only for the same verified variant
    name: str
    input_channels: int  # 4 = ordinary SDXL, 9 = native SDXL inpainting
    size: int | None = None
    header_sha256: str = ""


def _load_registry(path: Path | None = None) -> tuple[VerifiedSDXL, ...]:
    data = json.loads((path or Path(__file__).with_name("verified_sdxl.json")).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported SDXL inventory schema")
    records = []
    seen = set()
    for item in data["checkpoints"]:
        key = (item["source_id"], _path(item["checkpoint"]))
        channels = item["input_channels"]
        if (key in seen or item["source_id"] not in {source.id for source in SOURCES}
                or channels not in (4, 9) or not item["classification"].startswith("sdxl_")
                or item.get("sampling_profile") != "generic-sdxl"
                or not all(item.get(field) is True for field in ("embedded_clip_l", "embedded_clip_g", "embedded_vae"))
                or len(item["header_sha256"]) != 64):
            raise ValueError(f"Invalid verified SDXL record: {key}")
        seen.add(key)
        records.append(VerifiedSDXL(
            source_id=item["source_id"], checkpoint=item["checkpoint"], model_id=item["model_id"],
            name=item["name"], input_channels=channels, size=item["size"], header_sha256=item["header_sha256"],
        ))
    return tuple(records)


def _path(name: str) -> str:
    # A Windows separator is not a different model. Case and subfolders matter.
    return name.replace("\\", "/")


VERIFIED_SDXL = _load_registry()


def discover_sdxl(client: ComfyClient, source_label: str) -> list[ModelEntry]:
    source = next((item for item in SOURCES if item.label == source_label), None)
    if source is None:
        raise BackendUnavailable(f"No verified SDXL inventory is registered for {source_label}.")
    verified = [item for item in VERIFIED_SDXL if item.source_id == source.id]
    if not verified:
        return []
    info = client.object_info("CheckpointLoaderSimple")
    node = info.get("CheckpointLoaderSimple", {})
    field = node.get("input", {}).get("required", {}).get("ckpt_name", [])
    choices = field[0] if field and isinstance(field[0], list) else []
    live = {_path(name): name for name in choices if isinstance(name, str)}
    entries = []
    available_nodes: dict[str, bool] = {}
    missing_nodes = set()
    for item in verified:
        filename = live.get(_path(item.checkpoint))
        if filename is None:
            continue
        if item.input_channels == 4:
            template = "sdxl_edit"
            required = ("VAEEncode", "SetLatentNoiseMask")
        elif item.input_channels == 9:
            template = "sdxl_inpaint_edit"
            required = ("InpaintModelConditioning",)
        else:
            continue  # an unsupported architecture cannot become runnable
        for name in required:
            if name not in available_nodes:
                available_nodes[name] = bool(client.object_info(name).get(name))
        absent = [name for name in required if not available_nodes[name]]
        if absent:
            missing_nodes.update(absent)
            continue
        entries.append(ModelEntry(
            id=item.model_id, name=f"{item.name} · {source.label}", tier="edit",
            capabilities=("fill", "edit"), arch="sdxl", template=template,
            page="", license_name="Existing checkpoint's license", license_url="", gated=False,
            source_type="existing-checkpoint", uploader="Existing model on " + source.label,
            min_vram_gb=8, requires_nodes=(), requires_models=(), source="remote", family="sdxl",
            files=(FileSpec("checkpoint", "", "checkpoints/" + filename, size=item.size),),
        ))
    if not entries and missing_nodes:
        raise BackendUnavailable(f"{source.label}'s SDXL editing nodes are unavailable: {', '.join(sorted(missing_nodes))}.")
    return entries
