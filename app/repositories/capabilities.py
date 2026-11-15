"""Integration rows: secrets, MCP servers, capability bindings, tool calls, proposed actions."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    BindingStatus,
    CapabilityBinding,
    CompanySecret,
    McpServer,
    ProposedAction,
    ToolCall,
)


class CapabilityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ---------------------------------------------------------------- secrets
    async def get_secret(self, company_id: uuid.UUID, name: str) -> CompanySecret | None:
        stmt = select(CompanySecret).where(
            CompanySecret.company_id == company_id, CompanySecret.name == name
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_secrets(self, company_id: uuid.UUID) -> list[CompanySecret]:
        stmt = (
            select(CompanySecret)
            .where(CompanySecret.company_id == company_id)
            .order_by(CompanySecret.name)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def delete_secret(self, company_id: uuid.UUID, name: str) -> bool:
        res = await self.session.execute(
            delete(CompanySecret).where(
                CompanySecret.company_id == company_id, CompanySecret.name == name
            )
        )
        return bool(res.rowcount)  # type: ignore[attr-defined]

    # ------------------------------------------------------------ mcp servers
    async def get_server(self, company_id: uuid.UUID, server_id: uuid.UUID) -> McpServer | None:
        stmt = select(McpServer).where(
            McpServer.company_id == company_id, McpServer.id == server_id
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_servers(self, company_id: uuid.UUID) -> list[McpServer]:
        stmt = select(McpServer).where(McpServer.company_id == company_id).order_by(McpServer.name)
        return list((await self.session.execute(stmt)).scalars())

    # --------------------------------------------------------------- bindings
    async def get_binding(self, company_id: uuid.UUID, capability: str) -> CapabilityBinding | None:
        stmt = select(CapabilityBinding).where(
            CapabilityBinding.company_id == company_id, CapabilityBinding.capability == capability
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_bindings(
        self, company_id: uuid.UUID, *, live_only: bool = False
    ) -> list[CapabilityBinding]:
        stmt = select(CapabilityBinding).where(CapabilityBinding.company_id == company_id)
        if live_only:
            stmt = stmt.where(CapabilityBinding.status == BindingStatus.LIVE)
        return list(
            (await self.session.execute(stmt.order_by(CapabilityBinding.capability))).scalars()
        )

    async def servers_by_id(self, company_id: uuid.UUID) -> dict[str, McpServer]:
        return {str(s.id): s for s in await self.list_servers(company_id)}

    # ------------------------------------------------------------- tool calls
    async def add_tool_calls(self, rows: Sequence[ToolCall]) -> None:
        self.session.add_all(list(rows))
        await self.session.flush()

    async def tool_calls_for_chat(self, chat_id: uuid.UUID) -> list[ToolCall]:
        stmt = select(ToolCall).where(ToolCall.chat_id == chat_id).order_by(ToolCall.created_at)
        return list((await self.session.execute(stmt)).scalars())

    # ------------------------------------------------------ proposed actions
    async def add_actions(self, rows: Sequence[ProposedAction]) -> None:
        self.session.add_all(list(rows))
        await self.session.flush()

    async def actions_for_chat(self, chat_id: uuid.UUID) -> list[ProposedAction]:
        stmt = (
            select(ProposedAction)
            .where(ProposedAction.chat_id == chat_id)
            .order_by(ProposedAction.created_at, ProposedAction.id)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def lock_action(self, chat_id: uuid.UUID, action_id: uuid.UUID) -> ProposedAction | None:
        stmt = (
            select(ProposedAction)
            .where(ProposedAction.chat_id == chat_id, ProposedAction.id == action_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
