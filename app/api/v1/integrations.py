"""Admin: capability registry, company secrets, MCP servers and capability bindings."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, Path, Response, status

from app.api.deps import integration_service
from app.capabilities.registry import CAPABILITIES
from app.core.security import Principal, require_admin
from app.schemas.api import (
    CAPABILITY_PATTERN,
    SECRET_NAME_PATTERN,
    SLUG_PATTERN,
    BindingIn,
    BindingOut,
    BindingTestIn,
    BindingTestOut,
    ErrorResponse,
    McpServerIn,
    McpServerOut,
    McpServerPatch,
    SecretIn,
    SecretOut,
)
from app.services.integration_service import IntegrationService

router = APIRouter(tags=["integrations"])
SlugPath = Path(pattern=SLUG_PATTERN, examples=["bookshop"])
CapPath = Path(pattern=CAPABILITY_PATTERN, examples=["order.get"])
ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}
Svc = IntegrationService


@router.get("/capabilities", summary="Capability registry (admin)")
async def list_capabilities(_: Principal = Depends(require_admin)) -> list[dict[str, Any]]:
    return [c.summary() for c in CAPABILITIES.values()]


# ------------------------------------------------------------------- secrets
@router.get("/companies/{slug}/secrets", response_model=list[SecretOut], summary="Secret names")
async def list_secrets(
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> list[SecretOut]:
    return [SecretOut.build(s) for s in await service.list_secrets(slug)]


@router.post(
    "/companies/{slug}/secrets",
    response_model=SecretOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create or replace a secret (write-only; encrypted at rest)",
    responses=ERRORS,
)
async def put_secret(
    body: SecretIn,
    slug: str = SlugPath,
    principal: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> SecretOut:
    return SecretOut.build(await service.put_secret(slug, body.name, body.value, principal.label))


@router.delete(
    "/companies/{slug}/secrets/{name}", status_code=status.HTTP_204_NO_CONTENT, responses=ERRORS
)
async def delete_secret(
    slug: str = SlugPath,
    name: str = Path(pattern=SECRET_NAME_PATTERN),
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> Response:
    await service.delete_secret(slug, name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------- mcp servers
@router.get("/companies/{slug}/mcp-servers", response_model=list[McpServerOut])
async def list_mcp_servers(
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> list[McpServerOut]:
    return [McpServerOut.build(s) for s in await service.list_servers(slug)]


@router.post(
    "/companies/{slug}/mcp-servers",
    response_model=McpServerOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a company MCP server (streamable HTTP)",
    responses=ERRORS,
)
async def create_mcp_server(
    body: McpServerIn,
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> McpServerOut:
    server = await service.create_server(
        slug, name=body.name, url=body.url, auth_secret_name=body.auth_secret_name
    )
    return McpServerOut.build(server)


@router.patch(
    "/companies/{slug}/mcp-servers/{server_id}", response_model=McpServerOut, responses=ERRORS
)
async def update_mcp_server(
    body: McpServerPatch,
    server_id: uuid.UUID,
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> McpServerOut:
    changes = body.model_dump(exclude_unset=True)
    return McpServerOut.build(await service.update_server(slug, server_id, changes))


@router.delete(
    "/companies/{slug}/mcp-servers/{server_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ERRORS,
)
async def delete_mcp_server(
    server_id: uuid.UUID,
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> Response:
    await service.delete_server(slug, server_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/companies/{slug}/mcp-servers/{server_id}/discover",
    response_model=McpServerOut,
    summary="List and store the server's tools",
    responses={**ERRORS, 502: {"model": ErrorResponse}},
)
async def discover_mcp_tools(
    server_id: uuid.UUID,
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> McpServerOut:
    return McpServerOut.build(await service.discover(slug, server_id))


# ------------------------------------------------------------------ bindings
@router.get("/companies/{slug}/capability-bindings", response_model=list[BindingOut])
async def list_bindings(
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> list[BindingOut]:
    return [BindingOut.build(b) for b in await service.list_bindings(slug)]


@router.get(
    "/companies/{slug}/capability-bindings/{capability}",
    response_model=BindingOut,
    responses=ERRORS,
)
async def get_binding(
    slug: str = SlugPath,
    capability: str = CapPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> BindingOut:
    return BindingOut.build(await service.get_binding(slug, capability))


@router.put(
    "/companies/{slug}/capability-bindings/{capability}",
    response_model=BindingOut,
    summary="Create / update a binding (status resets to draft)",
    responses=ERRORS,
)
async def put_binding(
    body: BindingIn,
    slug: str = SlugPath,
    capability: str = CapPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> BindingOut:
    binding = await service.put_binding(
        slug, capability, adapter_type=body.adapter_type, config=body.config, policy=body.policy
    )
    return BindingOut.build(binding)


@router.delete(
    "/companies/{slug}/capability-bindings/{capability}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ERRORS,
)
async def delete_binding(
    slug: str = SlugPath,
    capability: str = CapPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> Response:
    await service.delete_binding(slug, capability)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/companies/{slug}/capability-bindings/{capability}/test",
    response_model=BindingTestOut,
    summary="Dry run with sample args (draft -> tested on success)",
    responses=ERRORS,
)
async def test_binding(
    body: BindingTestIn,
    slug: str = SlugPath,
    capability: str = CapPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> BindingTestOut:
    binding, result = await service.test_binding(
        slug, capability, body.sample_args, execute_write=body.execute_write
    )
    return BindingTestOut(binding=BindingOut.build(binding), result=result)


@router.post(
    "/companies/{slug}/capability-bindings/{capability}/go-live",
    response_model=BindingOut,
    summary="Make a tested binding live",
    responses=ERRORS,
)
async def go_live(
    slug: str = SlugPath,
    capability: str = CapPath,
    _: Principal = Depends(require_admin),
    service: Svc = Depends(integration_service),
) -> BindingOut:
    return BindingOut.build(await service.go_live(slug, capability))
