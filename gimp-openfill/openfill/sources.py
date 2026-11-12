"""ComfyUI endpoints. User-added servers belong to the current OpenFill profile."""

from __future__ import annotations

import hashlib
import re
import socket
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class Source:
    id: str
    label: str
    url: str
    ssh_alias: str | None = None
    remote_port: int | None = None

    @property
    def address(self) -> str:
        return f"ssh://{self.ssh_alias}:{self.remote_port}" if self.ssh_alias else self.url


SOURCES = (Source("local", "Local ComfyUI", "http://127.0.0.1:8188"),)
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}\Z")
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\Z")


def server_address(value: str) -> str:
    if not isinstance(value, str) or not (value := value.strip()) or len(value) > 2048:
        raise ValueError("Enter a ComfyUI address.")
    if any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value:
        raise ValueError("The server address contains an invalid character.")
    implicit = "://" not in value
    parts = urlsplit("http://" + value if implicit else value)
    host, port = parts.hostname, parts.port
    if (
        not host
        or parts.scheme not in ("http", "https", "ssh")
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port == 0
        or parts.netloc.endswith(":")
        or host in ("0.0.0.0", "::", "*")
    ):
        raise ValueError("Enter a server address without credentials, a query or a fragment.")
    if parts.scheme == "ssh":
        if not _ALIAS.fullmatch(host) or parts.path not in ("", "/"):
            raise ValueError("Use an SSH host alias and the remote ComfyUI port, for example ssh://studio:8188.")
        return f"ssh://{host}:{port or 8188}"
    if implicit and port is None:
        port = 8188
    authority = f"[{host}]" if ":" in host else host
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        authority += f":{port}"
    return urlunsplit((parts.scheme, authority, parts.path.rstrip("/"), "", ""))


def _saved_source(row) -> Source:
    source = Source(**row)
    if not _ID.fullmatch(source.id) or source.id in ("shared", "selection-helper", "managed", "custom"):
        raise ValueError("Invalid saved server ID.")
    if not isinstance(source.label, str) or not source.label.strip() or len(source.label) > 100:
        raise ValueError("Invalid saved server name.")
    url = server_address(source.url)
    if not url.startswith(("https://", "http://")):
        raise ValueError("Invalid saved server URL.")
    if source.ssh_alias is not None:
        if (
            not _ALIAS.fullmatch(source.ssh_alias)
            or type(source.remote_port) is not int
            or not 1 <= source.remote_port <= 65535
            or not re.fullmatch(r"http://127\.0\.0\.1:[0-9]+", url)
        ):
            raise ValueError("Invalid saved SSH server.")
    elif source.remote_port is not None:
        raise ValueError("A remote port needs an SSH alias.")
    return Source(source.id, source.label.strip(), url, source.ssh_alias, source.remote_port)


def sources(saved=()) -> tuple[Source, ...]:
    result = list(SOURCES)
    for row in saved:
        try:
            source = _saved_source(row)
        except (TypeError, ValueError, AttributeError):
            continue
        if not any(
            item.id == source.id
            or item.label.casefold() == source.label.casefold()
            or item.url == source.url
            or item.address == source.address
            for item in result
        ):
            result.append(source)
    return tuple(result)


def get_source(source_id: str, saved=()) -> Source:
    for source in sources(saved):
        if source.id == source_id:
            return source
    raise ValueError(f"Unknown ComfyUI source: {source_id!r}.")


def list_sources(saved=()) -> list[dict]:
    return [asdict(source) for source in sources(saved)]


def new_source(name: str, address: str, existing: tuple[Source, ...]) -> Source:
    address = server_address(address)
    for source in existing:
        if address in (source.address, source.url):
            return source
    parsed = urlsplit(address)
    if not isinstance(name, str):
        raise ValueError("Enter a server name.")
    label = name.strip() or parsed.netloc
    if len(label) > 100 or any(ord(c) < 32 for c in label):
        raise ValueError("Use a server name up to 100 characters.")
    if label.casefold() in {"managed comfyui", "local wevisone", "other comfyui url…"} or label.casefold().startswith(
        "shared ·"
    ):
        raise ValueError("That name belongs to an engine menu option. Choose a different name.")
    if any(source.label.casefold() == label.casefold() for source in existing):
        raise ValueError("That name already belongs to another server. Choose a different name.")
    key = "server-" + hashlib.sha256(address.encode()).hexdigest()[:20]
    if parsed.scheme != "ssh":
        return Source(key, label, address)
    # Reserve a different loopback port for every saved tunnel, including stopped ones.
    used = {urlsplit(source.url).port for source in existing}
    for _ in range(32):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in used:
            return Source(key, label, f"http://127.0.0.1:{port}", parsed.hostname, parsed.port)
    raise OSError("Could not allocate a local port for this SSH connection.")
