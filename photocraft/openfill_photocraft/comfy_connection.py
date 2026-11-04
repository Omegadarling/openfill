"""Connect the selected FIBO source; never start or stop its ComfyUI process."""

from __future__ import annotations

import atexit
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from openfill.comfy_api import ComfyClient
from openfill.sources import get_source

_connection_lock = threading.Lock()


def ensure_connection(url: str, profile: Path, *, port: int = 8188, source_id: str | None = None):
    """Open the configured loopback forward only if no existing tunnel serves it.

    Uses the existing SSH alias and agent/key setup, in batch mode. An existing
    tunnel belongs to its original owner; only a process created here is closed.
    """
    with _connection_lock:
        return _ensure_connection(url, profile, port=port, source_id=source_id)


def _ensure_connection(url: str, profile: Path, *, port: int, source_id: str | None):
    source = get_source(source_id or "local")
    parsed = urlsplit(url)
    if source_id is not None:
        if url.rstrip("/") != source.url:
            raise RuntimeError(f"{source.label}'s connection must use {source.url}, not {url}.")
        port = parsed.port
    if ComfyClient(url, timeout=3).is_alive():
        return None
    if source.ssh_alias is None:
        raise RuntimeError(
            f"ComfyUI on {source.label} is not answering at {url}. Start its existing ComfyUI, then use Engine > Check."
        )
    if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.port != port:
        raise RuntimeError(f"ComfyUI on {source.label} is not answering at {url}; check its server connection.")
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"Port {port} is occupied but {source.label} ComfyUI is not answering. Check the existing SSH connection."
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
        if ComfyClient(url, timeout=2).is_alive():
            return child
        time.sleep(0.25)
    stop_owned_tunnel()
    raise RuntimeError(
        f"Cannot connect to {source.label}. Check that {source.label} and its ComfyUI service are running; see {log_name}."
    )
