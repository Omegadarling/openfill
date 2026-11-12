"""Connect the selected FIBO source; never start or stop its ComfyUI process."""

from __future__ import annotations

import atexit
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from openfill.config import Paths, Settings
from openfill.sources import get_source

from .server_discovery import probe as comfy_version

_connection_lock = threading.Lock()
_owned = {}


def ensure_connection(url: str, profile: Path, *, port: int = 8188, source_id: str | None = None):
    """Use a direct URL or a loopback forward owned by this companion for this exact SSH target."""
    with _connection_lock:
        return _ensure_connection(url, profile, port=port, source_id=source_id)


def _ensure_connection(url: str, profile: Path, *, port: int, source_id: str | None):
    saved = Settings.load(Paths.for_profile(profile).config, backup=False).comfy_servers
    source = get_source(source_id or "local", saved)
    parsed = urlsplit(url)
    if source_id is not None:
        if url.rstrip("/") != source.url:
            raise RuntimeError(f"{source.label}'s connection must use {source.url}, not {url}.")
        port = parsed.port
    if source.ssh_alias is None:
        if comfy_version(url, timeout=3) is not None:
            return None
        raise RuntimeError(
            f"ComfyUI on {source.label} is not answering at {url}. Start ComfyUI and use Engine > Check."
        )
    target = (source.ssh_alias, source.remote_port)
    existing = _owned.get(port)
    if existing and existing[1].poll() is None and existing[0] == target:
        if comfy_version(url, timeout=3) is not None:
            return existing[1]
        raise RuntimeError(f"The SSH connection to {source.label} is open. Start ComfyUI there and use Engine > Check.")
    if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.port != port:
        raise RuntimeError(f"ComfyUI on {source.label} is not answering at {url}; check its server connection.")
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"Port {port} is already in use. Close the conflicting forward or add its local HTTP address explicitly."
            )
    profile.mkdir(parents=True, exist_ok=True)
    log_name = f"{source.id}-ssh.log"
    with (profile / log_name).open("ab") as log:
        child = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-L",
                f"127.0.0.1:{port}:127.0.0.1:{source.remote_port}",
                source.ssh_alias,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def stop_owned_tunnel():
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()

    atexit.register(stop_owned_tunnel)
    until = time.monotonic() + 15
    while time.monotonic() < until and child.poll() is None:
        if comfy_version(url, timeout=2) is not None and child.poll() is None:
            _owned[port] = (target, child)
            return child
        time.sleep(0.25)
    stop_owned_tunnel()
    raise RuntimeError(
        f"Cannot connect to {source.label}. Check that {source.label} and its ComfyUI service are running; see {log_name}."
    )
