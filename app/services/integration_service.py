"""Company integrations: secrets, MCP servers, capability bindings (draft -> tested -> live),
capability coverage, and the loaders the harness uses (live bindings, secret getter)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.capabilities.adapters.base import (
    BindingSnapshot,
    CallContext,
    ConfigError,
    SecretGetter,
)
from app.capabilities.adapters.local_mcp import FACTORIES as LOCAL_SERVERS
from app.capabilities.adapters.local_mcp import is_local_url, local_name
from app.capabilities.adapters.mcp import McpAdapter
from app.capabilities.gateway import CapabilityGateway, ToolCallRecord
from app.capabilities.policy import check_policy
from app.capabilities.registry import ArgumentError, coerce_args, get_capability
from app.capabilities.secrets import SecretStore
from app.core.config import Settings
from app.core.errors import (
    BindingNotTested,
    CompanyNotFound,
    Conflict,
    NotFound,
    ToolError,
    UnknownCapability,
    ValidationFailed,
    VersionNotFound,
)
from app.core.logging import get_logger
from app.core.net import UnsafeUrl, check_url
from app.db.models import (
    BindingStatus,
    CapabilityBinding,
    Company,
    CompanySecret,
    McpServer,
    utcnow,
)
from app.services.common import AppMetrics, CompanyContextCache, UnitOfWork, UowFactory

logger = get_logger(__name__)
GENERIC_POLICY_KEYS = {"blocked", "allowed_values"}


def snapshot(binding: CapabilityBinding, servers: dict[str, McpServer]) -> BindingSnapshot:
    server = None
    if binding.adapter_type == "mcp":
        s = servers.get(str(binding.config.get("server_id")))
        if s is not None:
            server = {"url": s.url, "auth_secret_name": s.auth_secret_name, "name": s.name}
    return BindingSnapshot(
        id=binding.id,
        company_id=binding.company_id,
        capability=binding.capability,
        adapter_type=binding.adapter_type,
        config=dict(binding.config),
        policy=dict(binding.policy or {}),
        status=binding.status,
        server=server,
    )


class IntegrationService:
    def __init__(
        self,
        uow_factory: UowFactory,
        gateway: CapabilityGateway,
        secrets: SecretStore,
        cache: CompanyContextCache,
        metrics: AppMetrics,
        settings: Settings,
    ) -> None:
        self._uow = uow_factory
        self._gateway = gateway
        self._secrets = secrets
        self._cache = cache
        self._metrics = metrics
        self._settings = settings

    async def _company(self, uow: UnitOfWork, slug: str) -> Company:
        company: Company | None = await uow.companies.get_by_slug(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        return company

    # ---------------------------------------------------------------- loaders
    async def live_bindings(self, company_id: uuid.UUID) -> dict[str, BindingSnapshot]:
        async with self._uow() as uow:
            bindings = await uow.capabilities.list_bindings(company_id, live_only=True)
            servers = await uow.capabilities.servers_by_id(company_id)
        return {b.capability: snapshot(b, servers) for b in bindings}

    def secret_getter(self, company_id: uuid.UUID) -> SecretGetter:
        async def get(name: str) -> str:
            async with self._uow() as uow:
                row: CompanySecret | None = await uow.capabilities.get_secret(company_id, name)
            if row is None:
                raise ToolError(f"Secret '{name}' is not configured")
            return self._secrets.decrypt(row.ciphertext)

        return get

    # ---------------------------------------------------------------- secrets
    async def put_secret(self, slug: str, name: str, value: str, by: str | None) -> CompanySecret:
        ciphertext = self._secrets.encrypt(value)
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            row = await uow.capabilities.get_secret(company.id, name)
            if row is None:
                row = CompanySecret(
                    id=uuid.uuid4(), company_id=company.id, name=name, created_at=utcnow()
                )
                uow.session.add(row)
            row.ciphertext = ciphertext
            row.created_by = by
            await uow.commit()
        logger.info("integration.secret_set", extra={"company": slug, "secret": name})
        return row

    async def list_secrets(self, slug: str) -> list[CompanySecret]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            rows: list[CompanySecret] = await uow.capabilities.list_secrets(company.id)
            return rows

    async def delete_secret(self, slug: str, name: str) -> None:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            if not await uow.capabilities.delete_secret(company.id, name):
                raise NotFound(f"Secret '{name}' not found")
            await uow.commit()

    # ------------------------------------------------------------ mcp servers
    def _check_server_url(self, url: str) -> None:
        if is_local_url(url):  # built-in in-process demo server: no network involved
            if local_name(url) not in LOCAL_SERVERS:
                raise ValidationFailed(
                    f"Unknown in-process MCP server {url!r} (known: {sorted(LOCAL_SERVERS)})"
                )
            return
        try:
            check_url(url, allow_private=self._settings.OUTBOUND_ALLOW_PRIVATE_NETWORK)
        except UnsafeUrl as exc:
            raise ValidationFailed(str(exc)) from exc

    async def create_server(
        self, slug: str, *, name: str, url: str, auth_secret_name: str | None
    ) -> McpServer:
        self._check_server_url(url)
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            server = McpServer(
                id=uuid.uuid4(),
                company_id=company.id,
                name=name,
                url=url,
                transport="in_process" if is_local_url(url) else "streamable_http",
                auth_secret_name=auth_secret_name,
                created_at=utcnow(),
            )
            uow.session.add(server)
            try:
                await uow.commit()
            except IntegrityError as exc:
                raise Conflict(f"MCP server '{name}' already exists") from exc
        self._cache.invalidate(slug)
        return server

    async def list_servers(self, slug: str) -> list[McpServer]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            servers: list[McpServer] = await uow.capabilities.list_servers(company.id)
            return servers

    async def _server(self, uow: UnitOfWork, company: Company, server_id: uuid.UUID) -> McpServer:
        server: McpServer | None = await uow.capabilities.get_server(company.id, server_id)
        if server is None:
            raise NotFound(f"MCP server {server_id} not found")
        return server

    async def update_server(
        self, slug: str, server_id: uuid.UUID, changes: dict[str, Any]
    ) -> McpServer:
        if changes.get("url"):
            self._check_server_url(changes["url"])
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            server = await self._server(uow, company, server_id)
            for key in ("name", "url", "auth_secret_name"):
                if key in changes:
                    setattr(server, key, changes[key])
            if changes.get("url"):
                server.transport = "in_process" if is_local_url(server.url) else "streamable_http"
            await uow.commit()
        self._cache.invalidate(slug)
        return server

    async def delete_server(self, slug: str, server_id: uuid.UUID) -> None:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            server = await self._server(uow, company, server_id)
            await uow.session.delete(server)
            await uow.commit()
        self._cache.invalidate(slug)

    async def discover(self, slug: str, server_id: uuid.UUID) -> McpServer:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            server = await self._server(uow, company, server_id)
        adapter = self._gateway.adapter("mcp")
        assert isinstance(adapter, McpAdapter)
        ctx = CallContext(
            company_id=company.id,
            get_secret=self.secret_getter(company.id),
            timeout=self._settings.TOOL_TIMEOUT_SECONDS,
            allow_private_network=self._settings.OUTBOUND_ALLOW_PRIVATE_NETWORK,
        )
        try:
            tools = await adapter.discover(
                {"url": server.url, "auth_secret_name": server.auth_secret_name}, ctx
            )
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        async with self._uow() as uow:
            server = await self._server(uow, company, server_id)
            server.discovered_tools = tools
            server.last_discovered_at = utcnow()
            await uow.commit()
        return server

    # --------------------------------------------------------------- bindings
    async def list_bindings(self, slug: str) -> list[CapabilityBinding]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            rows: list[CapabilityBinding] = await uow.capabilities.list_bindings(company.id)
            return rows

    async def get_binding(self, slug: str, capability: str) -> CapabilityBinding:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            binding: CapabilityBinding | None = await uow.capabilities.get_binding(
                company.id, capability
            )
        if binding is None:
            raise NotFound(f"No binding for '{capability}'")
        return binding

    async def put_binding(
        self,
        slug: str,
        capability: str,
        *,
        adapter_type: str,
        config: dict[str, Any],
        policy: dict[str, Any],
    ) -> CapabilityBinding:
        cap = get_capability(capability)
        if cap is None:
            raise UnknownCapability(f"Unknown capability '{capability}'")
        try:
            self._gateway.adapter(adapter_type).validate_config(config)
        except (ConfigError, ValueError) as exc:
            raise ValidationFailed(f"Invalid {adapter_type} config: {exc}") from exc
        bad = set(policy) - set(cap.policy_keys) - GENERIC_POLICY_KEYS
        if bad:
            raise ValidationFailed(
                f"Policy keys {sorted(bad)} don't apply to {capability} "
                f"(allowed: {sorted(set(cap.policy_keys) | GENERIC_POLICY_KEYS)})"
            )
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            if adapter_type == "mcp":
                try:
                    server_id = uuid.UUID(str(config["server_id"]))
                except ValueError as exc:
                    raise ValidationFailed("config.server_id must be a UUID") from exc
                await self._server(uow, company, server_id)
            binding = await uow.capabilities.get_binding(company.id, capability)
            if binding is None:
                binding = CapabilityBinding(
                    id=uuid.uuid4(),
                    company_id=company.id,
                    capability=capability,
                    created_at=utcnow(),
                )
                uow.session.add(binding)
            binding.adapter_type = adapter_type
            binding.config = config
            binding.policy = policy
            binding.status = BindingStatus.DRAFT  # any change requires a new test
            binding.last_test_result = None
            binding.updated_at = utcnow()
            await uow.commit()
        self._cache.invalidate(slug)
        logger.info("integration.binding_saved", extra={"company": slug, "capability": capability})
        return binding

    async def delete_binding(self, slug: str, capability: str) -> None:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            binding = await uow.capabilities.get_binding(company.id, capability)
            if binding is None:
                raise NotFound(f"No binding for '{capability}'")
            await uow.session.delete(binding)
            await uow.commit()
        self._cache.invalidate(slug)

    async def test_binding(
        self, slug: str, capability: str, sample_args: dict[str, Any], *, execute_write: bool
    ) -> tuple[CapabilityBinding, dict[str, Any]]:
        """Dry run. Reads are executed with the sample args. Writes are only validated
        (schema + policy) unless `execute_write=true` — a real write would have side effects."""
        cap = get_capability(capability)
        if cap is None:
            raise UnknownCapability(f"Unknown capability '{capability}'")
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            binding = await uow.capabilities.get_binding(company.id, capability)
            if binding is None:
                raise NotFound(f"No binding for '{capability}'")
            servers = await uow.capabilities.servers_by_id(company.id)
        snap = snapshot(binding, servers)
        record: ToolCallRecord | None = None
        if cap.kind == "write" and not execute_write:
            try:
                args = coerce_args(cap, sample_args)
                check = check_policy(cap, args, snap.policy)
                result = {
                    "ok": True,
                    "executed": False,
                    "args": args,
                    "policy": check.to_dict(),
                    "note": "write capability validated without executing",
                }
            except ArgumentError as exc:
                result = {"ok": False, "executed": False, "error": str(exc)}
        else:
            record = await self._gateway.call(
                company_id=company.id,
                capability=capability,
                args=sample_args,
                binding=snap,
                get_secret=self.secret_getter(company.id),
                trigger="test",
                idempotency_key=f"test-{uuid.uuid4().hex}",
                require_live=False,
            )
            result = {
                "ok": record.ok,
                "executed": True,
                "status": record.status,
                "result": record.result,
                "error": record.error,
                "latency_ms": record.latency_ms,
            }
        result["tested_at"] = utcnow().isoformat()
        async with self._uow() as uow:
            binding = await uow.capabilities.get_binding(company.id, capability)
            assert binding is not None
            binding.last_test_result = result
            if result["ok"] and binding.status == BindingStatus.DRAFT:
                binding.status = BindingStatus.TESTED
            if record is not None and (row := record.to_model(company.id)) is not None:
                await uow.capabilities.add_tool_calls([row])
            await uow.commit()
        return binding, result

    async def go_live(self, slug: str, capability: str) -> CapabilityBinding:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            binding = await uow.capabilities.get_binding(company.id, capability)
            if binding is None:
                raise NotFound(f"No binding for '{capability}'")
            if binding.status == BindingStatus.DRAFT:
                raise BindingNotTested(f"Test '{capability}' successfully before going live")
            binding.status = BindingStatus.LIVE
            binding.updated_at = utcnow()
            await uow.commit()
        self._cache.invalidate(slug)
        logger.info("integration.binding_live", extra={"company": slug, "capability": capability})
        return binding

    async def coverage(self, slug: str, number: int) -> dict[str, Any]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            version = await uow.versions.get_by_number(company.id, number)
            if version is None:
                raise VersionNotFound(f"Version {number} not found for '{slug}'")
            intents = await uow.playbooks.list_intents(version.id)
            bindings = {b.capability: b for b in await uow.capabilities.list_bindings(company.id)}
        referenced: dict[str, list[str]] = {}
        for it in intents:
            for st in it.steps:
                if cap := st.get("capability"):
                    referenced.setdefault(cap, []).append(f"{it.key}#{st['step_id']}")
        rows = []
        for cap_name, steps in sorted(referenced.items()):
            cap = get_capability(cap_name)
            b = bindings.get(cap_name)
            rows.append(
                {
                    "capability": cap_name,
                    "kind": cap.kind if cap else None,
                    "steps": steps,
                    "binding_status": b.status if b else None,
                    "adapter_type": b.adapter_type if b else None,
                    "handled_by": "ai" if b and b.status == BindingStatus.LIVE else "agent",
                }
            )
        live = sum(1 for r in rows if r["handled_by"] == "ai")
        return {
            "company": slug,
            "version_number": number,
            "referenced": len(rows),
            "live": live,
            "manual": len(rows) - live,
            "capabilities": rows,
        }
