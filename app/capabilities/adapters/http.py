"""HTTP adapter: one templated request per capability.

Config:
  {"method": "GET", "url": "https://api.example.com/orders/{order_id}",
   "headers": {"Accept": "application/json"}, "query": {"email": "{email}"},
   "body": {...} (POST/PUT/PATCH), "auth_secret": "orders_api_key",
   "auth_header": "Authorization", "auth_scheme": "Bearer", "response_path": "data"}
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from app.capabilities.adapters.base import (
    AdapterError,
    BindingSnapshot,
    CallContext,
    ConfigError,
    dig,
    render_template,
)
from app.core.net import UnsafeUrl, check_url

METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


class HttpAdapter:
    adapter_type = "http"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    def validate_config(self, config: dict[str, Any]) -> None:
        if str(config.get("method", "GET")).upper() not in METHODS:
            raise ConfigError(f"http config.method must be one of {sorted(METHODS)}")
        url = config.get("url")
        if not isinstance(url, str):
            raise ConfigError("http config.url is required")
        try:
            check_url(url.replace("{", "").replace("}", ""), allow_private=True, resolve=False)
        except UnsafeUrl as exc:
            raise ConfigError(str(exc)) from exc

    async def invoke(self, binding: BindingSnapshot, args: dict[str, Any], ctx: CallContext) -> Any:
        cfg = binding.config
        method = str(cfg.get("method", "GET")).upper()
        url = render_template(cfg["url"], {k: quote(str(v), safe="") for k, v in args.items()})
        try:
            check_url(url, allow_private=ctx.allow_private_network)
        except UnsafeUrl as exc:
            raise AdapterError(f"blocked outbound URL: {exc}") from exc
        headers = {k: str(v) for k, v in render_template(cfg.get("headers") or {}, args).items()}
        if secret_name := cfg.get("auth_secret"):
            secret = await ctx.secret(secret_name)
            scheme = cfg.get("auth_scheme", "Bearer")
            headers[cfg.get("auth_header", "Authorization")] = (
                f"{scheme} {secret}".strip() if scheme else secret
            )
        if ctx.idempotency_key and method != "GET":
            headers.setdefault("Idempotency-Key", ctx.idempotency_key)
        params = render_template(cfg.get("query") or {}, args)
        body = render_template(cfg.get("body"), args) if method != "GET" else None
        client = self._client or httpx.AsyncClient()
        try:
            resp = await client.request(
                method,
                url,
                headers=headers,
                params={k: v for k, v in params.items() if v != ""},
                json=body,
                timeout=ctx.timeout,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise AdapterError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if resp.status_code >= 400:
            raise AdapterError(f"HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            data = {"text": resp.text}
        return dig(data, cfg.get("response_path"))
