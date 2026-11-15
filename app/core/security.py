"""API-key authentication. Agent keys: chats + reads. Admin keys: everything."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Query, Request
from fastapi.security import APIKeyHeader

from app.core.config import ApiKey, Settings
from app.core.errors import Forbidden, Unauthorized

Role = Literal["agent", "admin"]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    role: Role
    label: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _match(candidate: str, keys: list[ApiKey]) -> ApiKey | None:
    found = None
    for k in keys:
        # constant-time comparison; keep looping to avoid timing differences
        if hmac.compare_digest(candidate.encode(), k.key.encode()):
            found = k
    return found


def resolve_principal(settings: Settings, api_key: str | None) -> Principal:
    if not api_key:
        raise Unauthorized()
    if admin := _match(api_key, settings.admin_keys):
        return Principal(role="admin", label=admin.label)
    if agent := _match(api_key, settings.agent_keys):
        return Principal(role="agent", label=agent.label)
    raise Unauthorized()


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def require_agent(
    request: Request, api_key: str | None = Depends(api_key_header)
) -> Principal:
    """Any valid key (agent or admin)."""
    return resolve_principal(_settings(request), api_key)


async def require_admin(
    request: Request, api_key: str | None = Depends(api_key_header)
) -> Principal:
    principal = resolve_principal(_settings(request), api_key)
    if not principal.is_admin:
        raise Forbidden()
    return principal


async def require_admin_header_or_query(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    key: str | None = Query(default=None, description="Admin API key (browser convenience)"),
) -> Principal:
    principal = resolve_principal(_settings(request), api_key or key)
    if not principal.is_admin:
        raise Forbidden()
    return principal
