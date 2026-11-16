"""In-process MCP servers for demo company systems, addressed as `inproc://<name>`.

They speak real MCP (tool discovery, schemas, structured results) through `mcp.Client`'s
in-process transport, so they need no port, no process and no network.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

LOCAL_SCHEME = "inproc://"


def _agrifarm() -> Any:
    from app.capabilities.adapters.local_mcp.agrifarm import build_server

    return build_server()


FACTORIES: dict[str, Callable[[], Any]] = {"agrifarm": _agrifarm}


def is_local_url(url: str) -> bool:
    return url.startswith(LOCAL_SCHEME)


def local_name(url: str) -> str:
    return url[len(LOCAL_SCHEME) :].strip("/")


@cache
def get_local_server(name: str) -> Any:
    """One server instance per process (its write ledger lives as long as the process)."""
    try:
        factory = FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown in-process MCP server '{name}' (known: {sorted(FACTORIES)})"
        ) from exc
    return factory()
