# OpenFill - generative fill for GIMP 3
# Copyright (C) 2026 Beckett Harriman
# SPDX-License-Identifier: Apache-2.0
"""Settings persisted as JSON in the GIMP profile (``<profile>/openfill``).

No GIMP imports: the caller passes the profile directory.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import ModelSpec

CONFIG_DIRNAME = "openfill"
LEGACY_DIRNAME = "stable-boy"  # folder name before the rename (settings are copied once)
CONFIG_FILENAME = "config.json"
STATE_FILENAME = "server-state.json"
LOG_FILENAME = "openfill.log"
SERVER_LOG_FILENAME = "comfyui.log"
MODELS_CACHE_FILENAME = "models-cache.json"
HUB_CACHE_FILENAME = "hub-cache.json"
HUB_SELECTED_FILENAME = "hub-selected.json"

PANEL_SECTIONS = ("edit", "references", "models", "engine", "generation")
DEFAULT_COLLAPSED = ("models", "engine", "generation")
_SAVE_LOCK = threading.RLock()


def normalize_panel_layout(value: Any) -> dict[str, list[str]]:
    """Keep each known section once and append sections added in later versions."""
    if not isinstance(value, dict):
        value = {}

    def sections(key, default):
        items = value.get(key)
        if not isinstance(items, list):
            return list(default)
        return list(dict.fromkeys(item for item in items if isinstance(item, str) and item in PANEL_SECTIONS))

    order = sections("order", PANEL_SECTIONS)
    order.extend(section for section in PANEL_SECTIONS if section not in order)
    return {"order": order, "collapsed": sections("collapsed", DEFAULT_COLLAPSED)}


def normalize_hidden_models(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and 0 < len(item) <= 256))


FIBO_EDIT_MODES = {"preserve": "Preserve structure", "replace": "Replace selected content"}
FIBO_CONTEXT_MODES = {"local": "Surrounding area", "full": "Full image"}
FIBO_PERCENT_RANGES = {
    "fibo_context_percent": (25.0, 200.0, 100.0, "Surrounding context"),
    "fibo_boundary_strength": (0.0, 100.0, 50.0, "Boundary color matching"),
}
GENERATION_FIELDS = {
    "steps": int,
    "cfg": float,
    "grow_percent": float,
    "feather_percent": float,
    "auto_feather": bool,
    "auto_feather_percent": float,
    "fibo_edit_mode": str,
    "fibo_context_mode": str,
    "fibo_context_percent": float,
    "fibo_boundary_match": bool,
    "fibo_boundary_strength": float,
}


def validate_fibo_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Validate a partial settings form before any settings are changed."""
    result = dict(values)
    for key, choices in (("fibo_edit_mode", FIBO_EDIT_MODES), ("fibo_context_mode", FIBO_CONTEXT_MODES)):
        if key in result and result[key] not in choices:
            raise ValueError(f"{key}: choose one of {', '.join(choices.values())}.")
    for key, (low, high, _default, label) in FIBO_PERCENT_RANGES.items():
        if key in result:
            amount = float(result[key])
            if not math.isfinite(amount):
                raise ValueError(f"{label} must be a finite percentage.")
            result[key] = max(low, min(high, amount))
    if "fibo_boundary_match" in result and not isinstance(result["fibo_boundary_match"], bool):
        raise ValueError("Boundary color matching must be on or off.")
    return result


def validate_generation_settings(values: Any) -> dict[str, Any]:
    """Validate the settings supplied with a PhotoCraft job before saving them."""
    if not isinstance(values, dict) or values.keys() - GENERATION_FIELDS.keys():
        raise ValueError("Provide only Generation settings with a job.")
    result = {}
    for key, value in values.items():
        kind = GENERATION_FIELDS[key]
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be on or off.")
        elif kind in (int, float):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or (kind is int and not isinstance(value, int))
            ):
                raise ValueError(f"{key} must be a finite {'whole number' if kind is int else 'number'}.")
            value = kind(value)
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be a choice from the Generation section.")
        result[key] = value
    if "auto_feather_percent" in result:
        result["auto_feather_percent"] = max(0.0, min(100.0, result["auto_feather_percent"]))
    return validate_fibo_settings(result)


def default_comfy_dir() -> str:
    """The ComfyUI folder assumed when the config does not name one."""
    local = os.environ.get("LOCALAPPDATA")
    if sys.platform == "win32" and local:
        return str(Path(local) / "OpenFill" / "ComfyUI" / "ComfyUI_windows_portable")
    return str(Path.home() / "ComfyUI")


@dataclass
class Settings:
    """User settings stored in ``config.json``.

    ``hf_token`` is a secret: it is left out of ``repr()`` so it cannot end up
    in a log line or a traceback, and :meth:`save` makes the file readable by
    its owner only on POSIX systems.
    """

    backend: str = "comfyui"  # "comfyui" or "a1111"
    remote_fibo: bool = False  # existing remote ComfyUI, including SSH loopback tunnels
    fibo_source: str = "local"  # an existing ComfyUI server ID
    comfy_servers: list[dict] = field(default_factory=list)
    comfy_dir: str = field(default_factory=default_comfy_dir)
    url: str = "http://127.0.0.1:8188"  # ComfyUI (or A1111) base URL
    a1111_url: str = "http://127.0.0.1:7860"
    autostart: bool = True  # start ComfyUI when needed
    idle_timeout_minutes: int = 20  # 0 = keep running until GIMP exits
    startup_timeout: int = 240  # seconds to wait for the server
    request_timeout: int = 0  # 0 = estimate per request
    extra_server_args: str = ""  # appended to the ComfyUI command
    cpu_comfy_dir: str = ""  # the ComfyUI folder Download ComfyUI added --cpu for (no NVIDIA card)
    checkpoint: str = "sd-v1-5-inpainting.ckpt"  # default (SD 1.5) model
    sdxl_checkpoint: str = "sd_xl_base_1.0.safetensors"
    fooocus_head: str = "fooocus_inpaint_head.pth"
    fooocus_patch: str = "inpaint_v26.fooocus.patch"
    quality: str = "fast"  # "fast" (SD 1.5) or "quality" (SDXL)
    steps: int = 25
    cfg: float = 7.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    denoise: float = 1.0
    grow: int = 48  # mask dilation cap (px); actual = grow_percent of selection
    feather: int = 24  # layer mask feather cap (px); actual = feather_percent
    grow_percent: float = 15.0  # dilation as percent of the selection's larger side (covers object rims)
    feather_percent: float = 6.0  # layer mask feather as percent of the larger side
    auto_feather: bool = True  # PhotoCraft edit selection: soften its inner edge
    auto_feather_percent: float = 10.0  # outer percentage of alpha-selected area
    fibo_edit_mode: str = "preserve"  # preserve original structure or replace selected content
    fibo_context_mode: str = "local"  # surrounding crop or full image
    fibo_context_percent: float = 100.0  # per-side padding as % of selection's larger dimension
    fibo_boundary_match: bool = True  # match colors at selection edges; does not realign shapes
    fibo_boundary_strength: float = 50.0  # 0-100% boundary color correction strength
    prefill: bool = True
    # Second pass at full resolution over the regenerated area when the fill
    # was worked below 1:1 (SD 1.5 and SDXL tiers), at this denoise. Off: in
    # the A/B it made fills softer (SD 1.5) or changed their content (SDXL).
    refine: bool = False
    refine_denoise: float = 0.4
    # Generative Expand worked below 1:1 (SD 1.5 and SDXL): a second pass at the image's own
    # resolution over the new area, in windows, at ``expand_detail_denoise``. Off until an
    # A/B on real photos shows it helps (it is untested on hardware); turn on in config.json.
    expand_detail: bool = False
    expand_detail_denoise: float = 0.4
    # Tiled fills: below 1.0 the whole context is generated first in one pass
    # (layout) and the tiles add detail to it at this denoise.
    tile_denoise: float = 0.65
    layout_min_scale: float = 0.45
    # Prompt-free removal (LaMa/MAT) of a context larger than this many px runs
    # in overlapping windows of this size at 1:1 (0 = one downscaled pass).
    removal_tile: int = 512
    removal_tile_overlap: int = 128
    # Add a scene description of the context (light, colour, grain) to prompted fills.
    scene_caption: bool = False
    # Add noise to a fill that is cleaner than the grain around it.
    grain_match: bool = True
    # Remove the faint outline a fill leaves on smooth areas (quality.seam_field).
    seam_correction: bool = True
    # Order several variations best first by cheap quality metrics.
    rank_variations: bool = True
    max_side_sd15: int = 768
    max_side_sdxl: int = 1024
    txt2img_width: int = 768
    txt2img_height: int = 512
    debug: bool = False
    active_model: str = ""  # catalog id set active by hand; "" = the automatic default (catalog.auto_active)
    hidden_models: list[str] = field(default_factory=list)  # chooser only, independently of installed models
    panel_layout: dict[str, list[str]] = field(default_factory=lambda: normalize_panel_layout(None))
    hf_token: str = field(default="", repr=False)  # Hugging Face access token for gated files (local only)
    progress_window: bool = True  # floating progress/preview window
    progress_window_x: int = -1  # last position; -1 = top-right of the image window
    progress_window_y: int = -1
    prompt_history: str = ""  # JSON list of recent prompts, newest first (dialogs)
    hub_search: bool = True  # Models tab: also list current models from Hugging Face and Civitai (read-only API calls)
    hub_civitai: bool = True  # include Civitai in that search (Hugging Face only when False)
    # Not persisted: why the last :meth:`load` fell back to defaults (None when it did not).
    load_problem: str | None = field(default=None, repr=False, compare=False, metadata={"persist": False})
    # Not persisted: a damaged settings file that could not be moved aside; :meth:`save` refuses to overwrite it.
    _protected_path: str | None = field(default=None, repr=False, compare=False, metadata={"persist": False})

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path, backup: bool = True) -> Settings:
        """Read ``path``; a missing file, unknown keys and mistyped values fall back to defaults.

        A file that exists but is not a JSON object (damaged, half written, edited
        by hand) is moved aside to ``config.json.corrupt-<YYYYmmdd-HHMMSS>`` so a
        later :meth:`save` cannot destroy the user's settings and token, and
        :attr:`load_problem` says so in plain words. ``backup=False`` (for
        processes other than the plug-in) only reads and never moves anything.
        """
        settings = cls()
        path = Path(path)
        try:  # utf-8-sig: tolerate the BOM that PowerShell's Set-Content adds
            text = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return settings
        except OSError as err:
            settings.load_problem = (
                f"Your settings file '{path}' could not be opened ({err.strerror or err}); defaults are in use "
                "for now and will not be saved over it. Check the file's permissions, then restart GIMP."
            )
            settings._protected_path = str(path)
            return settings
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            if backup:
                settings._move_aside(path)
            else:
                settings._protected_path = str(path)
            return settings
        known = {f.name for f in fields(cls) if f.metadata.get("persist", True)}
        for key, value in data.items():
            if key in known:
                try:
                    if key == "panel_layout":
                        settings.panel_layout = normalize_panel_layout(value)
                    elif key == "comfy_servers":
                        if isinstance(value, list):
                            settings.comfy_servers = value
                    elif key == "hidden_models":
                        settings.hidden_models = normalize_hidden_models(value)
                    else:
                        setattr(settings, key, _coerce(getattr(settings, key), value))
                except (TypeError, ValueError):
                    pass
        if settings.remote_fibo:
            # Preserve the selected variant when upgrading the PhotoCraft catalog.
            # Loading is still read-only; the caller persists on its next save.
            settings.active_model = {
                "remote-fibo2-edit-turbo-bf16": "remote-fibo2-edit-turbo-nvfp4",
                "remote-fibo2-dev-bf16": "remote-fibo2-dev-nvfp4",
            }.get(settings.active_model, settings.active_model)
        amount = settings.auto_feather_percent
        settings.auto_feather_percent = max(0.0, min(100.0, amount)) if math.isfinite(amount) else 10.0
        if settings.fibo_edit_mode not in FIBO_EDIT_MODES:
            settings.fibo_edit_mode = "preserve"
        if settings.fibo_context_mode not in FIBO_CONTEXT_MODES:
            settings.fibo_context_mode = "local"
        for key, (low, high, default, _label) in FIBO_PERCENT_RANGES.items():
            amount = getattr(settings, key)
            setattr(settings, key, max(low, min(high, amount)) if math.isfinite(amount) else default)
        return settings

    def _move_aside(self, path: Path) -> None:
        """Rename an unreadable settings file to a fresh ``.corrupt-<timestamp>`` name and record why."""
        target = corrupt_backup_path(path)
        try:
            os.rename(path, target)  # never replaces: the name was checked free, and rename fails on Windows if taken
        except OSError as err:
            self._protected_path = str(path)
            self.load_problem = (
                f"Your settings file '{path}' could not be read, and moving it aside failed "
                f"({err.strerror or err}); defaults are in use and the file was left untouched. "
                "Fix or delete the file, then open Filters > OpenFill > Generative Fill Settings again."
            )
            return
        self.load_problem = (
            f"Your settings file could not be read and was moved aside to '{target}'; defaults are in use. "
            "Open Filters > OpenFill > Generative Fill Settings to set the ComfyUI folder and your options again."
        )

    def drop_stale_cpu_flag(self, setup_dir: str | Path | None = None) -> bool:
        """Remove the ``--cpu`` that Download ComfyUI added for its own folder once another
        ComfyUI folder is in use; True when the settings changed.

        Settings from before ``cpu_comfy_dir`` existed name no folder: then the folder
        Download ComfyUI installs to (``setup_dir``) is taken as the one, if it exists. A
        ``--cpu`` typed by the user for the folder in use is kept.
        """
        args = self.extra_server_args.split()
        if "--cpu" not in args:
            return False
        origin = self.cpu_comfy_dir or (str(setup_dir) if setup_dir and Path(setup_dir).is_dir() else "")
        if not origin or _same_dir(origin, self.comfy_dir):
            return False
        self.extra_server_args = " ".join(a for a in args if a != "--cpu")
        self.cpu_comfy_dir = ""
        return True

    def save_server(self, path: Path, source: dict) -> None:
        with _SAVE_LOCK:
            old = self.comfy_servers
            self.comfy_servers = [*old, source]
            try:
                self.save(path)
            except OSError:
                self.comfy_servers = old
                raise

    def save(self, path: Path) -> None:
        """Write atomically (temporary file, then rename); owner-only permissions on POSIX."""
        path = Path(path)
        if self._protected_path and os.path.abspath(path) == os.path.abspath(self._protected_path):
            raise OSError(f"not saving: '{path}' could not be read and was not backed up, so it is kept as it is")
        with _SAVE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {f.name: getattr(self, f.name) for f in fields(self) if f.metadata.get("persist", True)}
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", encoding="utf-8", delete=False
                ) as file:
                    temporary = Path(file.name)
                    file.write(json.dumps(data, indent=2) + "\n")
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    # derived -----------------------------------------------------------

    @property
    def active_url(self) -> str:
        """Base URL of the selected backend."""
        if self.remote_fibo:
            from .sources import get_source

            return get_source(self.fibo_source, self.comfy_servers).url
        return self.a1111_url if self.backend == "a1111" else self.url

    def model_for(self, quality: str | None = None) -> ModelSpec:
        """Return a ``ModelSpec`` for the requested quality tier."""
        from .backend import ModelSpec

        tier = quality or self.quality
        if tier == "quality" and self.sdxl_checkpoint:
            return ModelSpec(self.sdxl_checkpoint, "sdxl", self.fooocus_head or None, self.fooocus_patch or None)
        return ModelSpec(self.checkpoint, "sd15")


def corrupt_backup_path(path: Path, now: float | None = None) -> Path:
    """A name next to ``path`` that does not exist yet: ``<name>.corrupt-<YYYYmmdd-HHMMSS>[-N]``."""
    path = Path(path)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    candidate = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
    return candidate


def _coerce(current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return str(value)


@dataclass
class Paths:
    """All files the plug-in owns inside the GIMP profile."""

    base: Path

    @classmethod
    def for_profile(cls, profile_dir: str | Path) -> Paths:
        """Paths under ``<profile_dir>/openfill``. The first run after the
        rename copies the settings of the old ``stable-boy`` folder over."""
        base = Path(profile_dir) / CONFIG_DIRNAME
        legacy = Path(profile_dir) / LEGACY_DIRNAME
        if not base.exists() and legacy.is_dir():
            try:
                base.mkdir(parents=True)
                for name in (CONFIG_FILENAME, HUB_SELECTED_FILENAME):
                    if (legacy / name).is_file():
                        shutil.copy2(legacy / name, base / name)
            except OSError:
                pass  # start with defaults; the old folder stays untouched
        return cls(base)

    @property
    def config(self) -> Path:
        """Settings file (holds the Hugging Face token)."""
        return self.base / CONFIG_FILENAME

    @property
    def state(self) -> Path:
        """Supervisor state: pids, URL, status."""
        return self.base / STATE_FILENAME

    @property
    def log(self) -> Path:
        """Plug-in and supervisor log."""
        return self.base / LOG_FILENAME

    @property
    def server_log(self) -> Path:
        """ComfyUI stdout and stderr."""
        return self.base / SERVER_LOG_FILENAME

    @property
    def models_cache(self) -> Path:
        """Model files the server listed last time (no secrets)."""
        return self.base / MODELS_CACHE_FILENAME

    @property
    def hub_cache(self) -> Path:
        """Hugging Face / Civitai search results, kept for 24 hours (no secrets)."""
        return self.base / HUB_CACHE_FILENAME

    @property
    def hub_selected(self) -> Path:
        """Models from the live search the user chose to download (re-validated on every read)."""
        return self.base / HUB_SELECTED_FILENAME

    @property
    def temp(self) -> Path:
        """Scratch folder (debug dumps)."""
        return self.base / "tmp"

    def ensure(self) -> Paths:
        """Create the folders; returns ``self``."""
        self.base.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)
        return self


def _same_dir(a: str | Path, b: str | Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    except (TypeError, ValueError):
        return False


def comfy_layout(comfy_dir: str | Path) -> dict[str, Path] | None:
    """Locate python + main.py for a ComfyUI install; None if not found.

    Supports the official Windows portable (``python_embeded/python.exe`` +
    ``ComfyUI/main.py``), a plain git checkout with a ``venv``/``.venv``,
    and a checkout where the interpreter is given explicitly via PATH.
    """
    root = Path(comfy_dir)
    candidates: list[tuple[Path, Path, Path]] = []
    if (root / "python_embeded" / "python.exe").exists() and (root / "ComfyUI" / "main.py").exists():
        candidates.append((root / "python_embeded" / "python.exe", root / "ComfyUI" / "main.py", root))
    main = root / "main.py"
    if main.exists():
        for venv in ("venv", ".venv", "env"):
            for py in ("Scripts/python.exe", "bin/python3", "bin/python"):
                exe = root / venv / py
                if exe.exists():
                    candidates.append((exe, main, root))
        # The interpreter running this code, when it can be started from here: not on macOS
        # when it is GIMP's own Python (GIMP.app has a launch constraint and no PyTorch).
        if not (sys.platform == "darwin" and in_app_bundle(sys.executable)):
            candidates.append((Path(sys.executable), main, root))
    if not candidates:
        return None
    python, main, cwd = candidates[0]
    return {"python": python, "main": main, "cwd": cwd}


def windows_path_problem(python: Path, platform: str | None = None) -> str:
    """Check the path to a deeply nested dependency before Windows startup.

    Keep the user's spelling: resolving a short junction defeats its purpose.
    This is a conservative compatibility check, not a claim that every Windows
    installation enforces MAX_PATH.
    """
    if (platform or sys.platform) != "win32":
        return ""
    site = python.parent
    if site.name.lower() == "scripts":
        site = site.parent
    dependency = (
        site
        / "Lib/site-packages/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py"
    )
    length = len(str(dependency.absolute()).encode("utf-16-le")) // 2
    if length < 260:
        return ""
    return (
        f"The ComfyUI install path is too long for reliable Windows startup ({length} characters "
        "for a Transformers file; the Windows compatibility limit is 259). "
        "Use a shorter folder such as C:\\AI\\ComfyUI, or a short directory junction to the existing "
        "installation, then choose that folder in OpenFill Settings. Keep your models with ComfyUI."
    )


def in_app_bundle(path: str | Path) -> bool:
    """``path`` lies inside a macOS ``.app`` bundle (``GIMP.app/Contents/...``)."""
    return any(part.endswith(".app") for part in Path(path).parts)


def comfy_problem(comfy_dir: str | Path) -> str:
    """Why ``comfy_dir`` is not a ComfyUI OpenFill can start, in plain words ("" when it is)."""
    if not comfy_dir:
        return "No ComfyUI folder is set."
    if comfy_layout(comfy_dir) is not None:
        return ""
    root = Path(comfy_dir)
    if not root.is_dir():
        return f"ComfyUI was not found at '{root}': the folder does not exist."
    if (root / "main.py").is_file():
        return (
            f"ComfyUI was not found at '{root}' in a form OpenFill can start: main.py is there but no venv. "
            "Create one in that folder (python3 -m venv venv, then install ComfyUI's requirements into it)."
        )
    return (
        f"ComfyUI was not found at '{root}': it holds neither main.py with a venv nor the portable build "
        "(python_embeded and ComfyUI)."
    )
