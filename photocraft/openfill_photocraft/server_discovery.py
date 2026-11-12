# SPDX-License-Identifier: Apache-2.0
"""Find running ComfyUI endpoints on the current LAN when the user requests a search."""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

PRIVATE = tuple(ipaddress.ip_network(s) for s in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
PORTS = (8188, 8000)


def _output(args):
    return subprocess.check_output(
        args, timeout=4, text=True, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )


def local_network():
    """Only the private, directly connected portion of the primary interface, up to 254 hosts."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # Route lookup only. A UDP connect sends no packets.
            sock.connect(("192.0.2.1", 9))
            host = ipaddress.IPv4Address(sock.getsockname()[0])
        if not any(host in subnet for subnet in PRIVATE):
            return None
        if sys.platform == "darwin":
            output = _output(["/sbin/ifconfig"])
            match = re.search(r"\binet " + re.escape(str(host)) + r" netmask (0x[\da-fA-F]+)", output)
            if not match:
                return None
            mask = str(ipaddress.IPv4Address(int(match[1], 16)))
        elif sys.platform == "win32":
            # host comes from IPv4Address, never from UI text.
            mask = _output(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"(Get-NetIPAddress -IPAddress '{host}' -AddressFamily IPv4).PrefixLength",
                ]
            ).strip()
        else:
            links = json.loads(_output(["ip", "-j", "-4", "address", "show"]))
            mask = next(
                item["prefixlen"]
                for link in links
                for item in link.get("addr_info", [])
                if item.get("local") == str(host)
            )
        prefix = ipaddress.ip_network(f"{host}/{mask}", strict=False).prefixlen
        return ipaddress.ip_network(f"{host}/{max(24, prefix)}", strict=False)
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        return None


def probe(url, timeout=0.65):
    """Read only ComfyUI's small status response, without proxies or redirects."""
    parts = urlsplit(url)
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = cls(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.connect()
        connection.sock.settimeout(max(2, timeout))
        connection.request("GET", parts.path.rstrip("/") + "/system_stats", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return None
        raw = response.read(65537)
        if len(raw) > 65536:
            return None
        data = json.loads(raw)
        system = data.get("system")
        if not isinstance(system, dict) or not isinstance(data.get("devices"), list):
            return None
        if not (system.get("comfyui_version") or (system.get("pytorch_version") and system.get("python_version"))):
            return None
        return str(system.get("comfyui_version") or "ComfyUI")[:64]
    except (OSError, ValueError, AttributeError, http.client.HTTPException):
        return None
    finally:
        connection.close()


def discover(registered):
    network = local_network()
    hosts = ["127.0.0.1"]
    if network:
        hosts.extend(str(ip) for ip in network.hosts())
    urls = {f"http://{host}:{port}": None for host in hosts for port in PORTS}
    for source in registered:
        urls[source.url] = source
        parts = urlsplit(source.url)
        # Give a saved hostname priority over numeric addresses for the same endpoint.
        if parts.scheme == "http" and not parts.path and not source.ssh_alias:
            try:
                addresses = socket.getaddrinfo(parts.hostname, parts.port or 80, socket.AF_INET, socket.SOCK_STREAM)
            except OSError:
                continue
            for item in addresses:
                address = f"http://{item[4][0]}:{parts.port or 80}"
                if address != source.url and urls.get(address) is None:
                    urls.pop(address, None)

    def check(item):
        url, source = item
        version = probe(url)
        if version is None:
            return None
        return {
            "address": source.address if source else url,
            "name": source.label if source else urlsplit(url).netloc,
            "saved": source is not None,
            "version": version,
        }

    with ThreadPoolExecutor(max_workers=32, thread_name_prefix="comfy-find") as pool:
        found = [item for item in pool.map(check, urls.items()) if item]
    return sorted(found, key=lambda item: item["name"].casefold())
