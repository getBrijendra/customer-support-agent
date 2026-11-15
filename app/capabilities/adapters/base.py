"""Adapter interface + shared types. Adapters are the only code that talks to company systems."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class AdapterError(Exception):
    """The company system returned an error or could not be reached."""


class ConfigError(ValueError):
    """A binding's adapter config is invalid (raised at save time)."""


@dataclass
class BindingSnapshot:
    """Immutable view of a capability binding (plus resolved MCP server info)."""

    id: uuid.UUID
    company_id: uuid.UUID
    capability: str
    adapter_type: str
    config: dict[str, Any]
    policy: dict[str, Any]
    status: str
    server: dict[str, Any] | None = None  # for mcp: {"url", "auth_secret_name"}


SecretGetter = Callable[[str], Awaitable[str]]


@dataclass
class CallContext:
    company_id: uuid.UUID
    get_secret: SecretGetter
    timeout: float
    allow_private_network: bool
    idempotency_key: str | None = None
    used_secrets: list[str] = field(default_factory=list)  # filled for redaction

    async def secret(self, name: str) -> str:
        value = await self.get_secret(name)
        self.used_secrets.append(value)
        return value


class CapabilityAdapter(Protocol):
    adapter_type: str

    def validate_config(self, config: dict[str, Any]) -> None: ...

    async def invoke(
        self, binding: BindingSnapshot, args: dict[str, Any], ctx: CallContext
    ) -> Any: ...


def render_template(value: Any, args: dict[str, Any]) -> Any:
    """Replace `{arg}` placeholders in strings (recursively). Missing args render as ''."""
    if isinstance(value, str):

        class _Safe(dict[str, Any]):
            def __missing__(self, key: str) -> str:
                return ""

        try:
            return value.format_map(_Safe(args))
        except (ValueError, IndexError):
            return value
    if isinstance(value, dict):
        return {k: render_template(v, args) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, args) for v in value]
    return value


def dig(data: Any, path: str | None) -> Any:
    """Follow a dotted path (`data.items.0`) into JSON."""
    if not path:
        return data
    for part in path.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            return None
    return data
