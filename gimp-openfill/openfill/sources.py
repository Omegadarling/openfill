"""Known FIBO backends. Selecting one never starts or stops ComfyUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Source:
    id: str
    label: str
    url: str
    ssh_alias: str | None = None
    remote_port: int | None = None


SOURCES = (
    Source("local", "Local ComfyUI", "http://127.0.0.1:8188"),
)


def get_source(source_id: str) -> Source:
    for source in SOURCES:
        if source.id == source_id:
            return source
    raise ValueError(f"Unknown FIBO source: {source_id!r}.")


def list_sources() -> list[dict]:
    return [asdict(source) for source in SOURCES]
