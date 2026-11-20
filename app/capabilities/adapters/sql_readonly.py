"""Read-only SQL adapter: runs ONE admin-defined, parameterized SELECT against a company DB.

Config: {"dsn_secret": "orders_db_dsn",
         "statement": "SELECT id, status FROM orders WHERE id = :order_id"}

Defence in depth: the statement is validated at save time (single SELECT/WITH, no DML/DDL),
runs in a READ ONLY transaction with statement_timeout, and the result is row-limited.
Model-written SQL is never executed — the model only supplies parameter values.
"""

from __future__ import annotations

import re
from typing import Any

import asyncpg
from sqlalchemy.engine import make_url

from app.capabilities.adapters.base import AdapterError, BindingSnapshot, CallContext, ConfigError

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|grant|revoke|truncate|copy|call|do|lock|"
    r"vacuum|analyze|refresh|comment|security|set|reset|listen|notify|execute|prepare|into|"
    r"pg_sleep|dblink|lo_import|lo_export|pg_read_file)\b",
    re.I,
)
_PARAM = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def validate_select(statement: str) -> str:
    """Return the cleaned statement or raise ConfigError."""
    cleaned = strip_sql_comments(statement).strip().rstrip(";").strip()
    if not cleaned:
        raise ConfigError("SQL statement is empty")
    if ";" in cleaned:
        raise ConfigError("Only a single SQL statement is allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        raise ConfigError("Only SELECT (or WITH ... SELECT) statements are allowed")
    if m := _FORBIDDEN.search(cleaned):
        raise ConfigError(f"Keyword '{m.group(1)}' is not allowed in a read-only statement")
    return cleaned


def to_positional(statement: str) -> tuple[str, list[str]]:
    """`:name` params -> asyncpg `$n` params (casts like `::text` are left alone)."""
    names: list[str] = []

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in names:
            names.append(name)
        return f"${names.index(name) + 1}"

    return _PARAM.sub(repl, statement), names


def plain_dsn(dsn: str) -> str:
    url = make_url(dsn)
    if url.drivername.startswith("postgresql+"):
        url = url.set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


class SqlReadOnlyAdapter:
    adapter_type = "sql_readonly"

    def __init__(self, *, row_limit: int = 50, statement_timeout_s: float = 5) -> None:
        self._limit = row_limit
        self._timeout = statement_timeout_s

    def validate_config(self, config: dict[str, Any]) -> None:
        if not config.get("dsn_secret"):
            raise ConfigError("sql_readonly config.dsn_secret is required")
        validate_select(str(config.get("statement", "")))

    async def invoke(self, binding: BindingSnapshot, args: dict[str, Any], ctx: CallContext) -> Any:
        cfg = binding.config
        statement = validate_select(str(cfg.get("statement", "")))
        sql, names = to_positional(statement)
        values = [args.get(n) for n in names]
        dsn = plain_dsn(await ctx.secret(cfg["dsn_secret"]))
        timeout_ms = int(min(self._timeout, ctx.timeout) * 1000)
        pglite = False
        try:
            conn = await asyncpg.connect(dsn, timeout=ctx.timeout, statement_cache_size=0)
            pglite = "PGlite" in str(await conn.fetchval("SELECT version()"))
        except Exception as exc:
            raise AdapterError(f"cannot connect: {type(exc).__name__}") from exc
        try:
            async with conn.transaction(readonly=True):
                await conn.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                rows = await conn.fetch(
                    f"SELECT * FROM ({sql}) AS q LIMIT {int(self._limit)}", *values
                )
        except asyncpg.PostgresError as exc:
            raise AdapterError(f"query failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if pglite:
                conn.terminate()
            else:
                await conn.close()
        return {"rows": [_jsonable(dict(r)) for r in rows], "row_count": len(rows)}


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[k] = v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v)
    return out
