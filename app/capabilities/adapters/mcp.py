"""MCP adapter: calls ONE mapped tool on a company's MCP server (streamable HTTP).

Binding config: {"server_id": "<uuid>", "tool_name": "lookup_order",
                 "arg_map": {"order_id": "id"}}   # capability arg -> tool arg
Only the mapped tool is ever called; other tools on the server are invisible to the model.

`connector` can be overridden (tests connect to an in-process `MCPServer`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mcp import Client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from app.capabilities.adapters.base import AdapterError, BindingSnapshot, CallContext, ConfigError
from app.core.net import UnsafeUrl, check_url

Connector = Callable[[str, dict[str, str]], Any]


def default_connector(url: str, headers: dict[str, str]) -> Any:
    if not headers:
        return url
    return streamable_http_client(url, http_client=create_mcp_http_client(headers=headers))


class McpAdapter:
    adapter_type = "mcp"

    def __init__(self, connector: Connector | None = None) -> None:
        self._connector = connector or default_connector
        self._custom = connector is not None

    def validate_config(self, config: dict[str, Any]) -> None:
        if not config.get("server_id") or not config.get("tool_name"):
            raise ConfigError("mcp config needs server_id and tool_name")
        if not isinstance(config.get("arg_map", {}), dict):
            raise ConfigError("mcp config.arg_map must be an object")

    async def _target(self, server: dict[str, Any], ctx: CallContext) -> Any:
        url = server["url"]
        if not self._custom:
            try:
                check_url(url, allow_private=ctx.allow_private_network)
            except UnsafeUrl as exc:
                raise AdapterError(f"blocked MCP URL: {exc}") from exc
        headers: dict[str, str] = {}
        if name := server.get("auth_secret_name"):
            headers["Authorization"] = f"Bearer {await ctx.secret(name)}"
        return self._connector(url, headers)

    async def invoke(self, binding: BindingSnapshot, args: dict[str, Any], ctx: CallContext) -> Any:
        if not binding.server:
            raise AdapterError("MCP server for this binding no longer exists")
        cfg = binding.config
        arg_map: dict[str, str] = cfg.get("arg_map") or {}
        tool_args = {arg_map.get(k, k): v for k, v in args.items()}
        try:
            async with Client(
                await self._target(binding.server, ctx), read_timeout_seconds=ctx.timeout
            ) as client:
                result = await client.call_tool(cfg["tool_name"], tool_args)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"MCP call failed: {type(exc).__name__}: {exc}") from exc
        if result.is_error:
            raise AdapterError(f"MCP tool error: {_text(result.content)[:300]}")
        if result.structured_content is not None:
            return result.structured_content
        text = _text(result.content)
        try:
            return json.loads(text)
        except ValueError:
            return {"text": text}

    async def discover(self, server: dict[str, Any], ctx: CallContext) -> list[dict[str, Any]]:
        try:
            async with Client(
                await self._target(server, ctx), read_timeout_seconds=ctx.timeout
            ) as client:
                listed = await client.list_tools()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"MCP discovery failed: {type(exc).__name__}: {exc}") from exc
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": getattr(t, "input_schema", None),
            }
            for t in listed.tools
        ]


def _text(content: list[Any]) -> str:
    return "\n".join(getattr(c, "text", "") for c in content if getattr(c, "text", None))
