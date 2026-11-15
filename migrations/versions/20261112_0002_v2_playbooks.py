"""v2: compiled playbooks, pgvector routing index, evals, integrations, chat routing trace

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12

v1 versions have no compiled rows, so they are set to review_status='pending' and every
company's active pointer is cleared: chats return NO_ACTIVE_GUIDELINES until guidelines are
re-uploaded (or re-seeded) and compiled.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIMS = 768
TS_CONFIG = "english"
J = postgresql.JSONB


def _uuid(name: str, nullable: bool = False) -> sa.Column:  # type: ignore[type-arg]
    return sa.Column(name, sa.Uuid(), nullable=nullable)


def _scoped(table: str) -> list[sa.Column | sa.ForeignKeyConstraint]:  # type: ignore[type-arg]
    return [
        _uuid("company_id"),
        _uuid("version_id"),
        sa.ForeignKeyConstraint(
            ["company_id"], ["companies.id"], name=f"fk_{table}_company_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["guideline_versions.id"],
            name=f"fk_{table}_version_id",
            ondelete="CASCADE",
        ),
    ]


def _tsv(table: str, expr: str) -> None:
    op.execute(
        f"ALTER TABLE {table} ADD COLUMN tsv tsvector "
        f"GENERATED ALWAYS AS (to_tsvector('{TS_CONFIG}', {expr})) STORED"
    )
    op.create_index(f"ix_{table}_tsv", table, ["tsv"], postgresql_using="gin")


def _hnsw(table: str) -> None:
    op.execute(
        f"CREATE INDEX ix_{table}_embedding ON {table} USING hnsw (embedding vector_cosine_ops)"
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ---------------------------------------------------------- guideline_versions
    gv = "guideline_versions"
    op.drop_column(gv, "chunk_count")
    op.drop_column(gv, "vector_collection")
    op.drop_column(gv, "chunk_size")
    op.drop_column(gv, "chunk_overlap")
    op.add_column(
        gv, sa.Column("review_status", sa.String(20), nullable=False, server_default="pending")
    )
    op.add_column(
        gv,
        sa.Column("activate_on_approval", sa.Boolean(), nullable=False, server_default="true"),
    )
    op.add_column(gv, sa.Column("reviewed_by", sa.Text(), nullable=True))
    op.add_column(gv, sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(gv, sa.Column("review_note", sa.Text(), nullable=True))
    op.add_column(
        gv, sa.Column("source_format", sa.String(16), nullable=False, server_default="txt")
    )
    op.add_column(
        gv,
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False, server_default=str(DIMS)),
    )
    op.add_column(gv, sa.Column("compiler_model", sa.Text(), nullable=True))
    op.add_column(gv, sa.Column("compile_prompt_version", sa.String(32), nullable=True))
    op.add_column(gv, sa.Column("section_count", sa.Integer(), nullable=True))
    op.add_column(gv, sa.Column("intent_count", sa.Integer(), nullable=True))
    op.add_column(gv, sa.Column("fact_count", sa.Integer(), nullable=True))
    op.add_column(gv, sa.Column("coverage", J(), nullable=True))
    op.add_column(gv, sa.Column("validation_warnings", J(), nullable=True))
    op.add_column(gv, sa.Column("routing_eval_run_id", sa.Uuid(), nullable=True))
    op.add_column(gv, sa.Column("compile_duration_ms", sa.Integer(), nullable=True))
    op.add_column(gv, sa.Column("ingest_duration_ms", sa.Integer(), nullable=True))
    op.create_check_constraint(
        "ck_guideline_versions_review_status",
        gv,
        "review_status IN ('pending', 'approved', 'rejected')",
    )
    # v1 versions can't serve v2 chats: no compiled playbook yet.
    op.execute("UPDATE companies SET active_version_id = NULL")

    # -------------------------------------------------------------- playbook tables
    op.create_table(
        "guideline_sections",
        _uuid("id"),
        *_scoped("guideline_sections"),
        sa.Column("ref", sa.String(16), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(DIMS), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_guideline_sections"),
    )
    op.create_index(
        "ix_guideline_sections_company_version", "guideline_sections", ["company_id", "version_id"]
    )
    _tsv("guideline_sections", "coalesce(heading, '') || ' ' || text")
    _hnsw("guideline_sections")

    op.create_table(
        "intents",
        _uuid("id"),
        *_scoped("intents"),
        sa.Column("key", sa.String(80), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("conditions", J(), nullable=True),
        sa.Column("steps", J(), nullable=False),
        sa.Column("reply_template", sa.Text(), nullable=True),
        sa.Column("placeholders", J(), nullable=True),
        sa.Column("source_section_ids", J(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_intents"),
        sa.UniqueConstraint("version_id", "key", name="uq_intents_version_key"),
    )
    op.create_index("ix_intents_company_version", "intents", ["company_id", "version_id"])

    op.create_table(
        "intent_examples",
        _uuid("id"),
        *_scoped("intent_examples"),
        _uuid("intent_id"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("split", sa.String(8), nullable=False),
        sa.Column("embedding", Vector(DIMS), nullable=True),
        sa.ForeignKeyConstraint(
            ["intent_id"], ["intents.id"], name="fk_intent_examples_intent_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_intent_examples"),
        sa.CheckConstraint("split IN ('index', 'eval')", name="ck_intent_examples_split"),
    )
    op.create_index(
        "ix_intent_examples_company_version", "intent_examples", ["company_id", "version_id"]
    )
    _tsv("intent_examples", "text")
    _hnsw("intent_examples")

    op.create_table(
        "company_rules",
        _uuid("id"),
        *_scoped("company_rules"),
        sa.Column("rules", J(), nullable=False),
        sa.Column("source_section_ids", J(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_company_rules"),
        sa.UniqueConstraint("version_id", name="uq_company_rules_version_id"),
    )

    op.create_table(
        "reference_facts",
        _uuid("id"),
        *_scoped("reference_facts"),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_section_ids", J(), nullable=True),
        sa.Column("embedding", Vector(DIMS), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_reference_facts"),
    )
    op.create_index(
        "ix_reference_facts_company_version", "reference_facts", ["company_id", "version_id"]
    )
    _tsv("reference_facts", "title || ' ' || text")
    _hnsw("reference_facts")

    # ------------------------------------------------------------------- chats
    op.drop_column("chats", "retrieved_chunks")
    op.alter_column("chats", "latency_ms_retrieval", new_column_name="latency_ms_routing")
    op.add_column("chats", sa.Column("primary_intent", sa.String(80), nullable=True))
    op.add_column("chats", sa.Column("routing", J(), nullable=True))
    op.add_column("chats", sa.Column("references", J(), nullable=True))
    op.add_column("chats", sa.Column("feedback", J(), nullable=True))
    op.add_column("chats", sa.Column("router_model", sa.Text(), nullable=True))
    op.add_column("chats", sa.Column("router_prompt_version", sa.String(32), nullable=True))
    op.add_column("chats", sa.Column("latency_ms_tools", sa.Integer(), nullable=True))
    op.add_column("chats", sa.Column("router_input_tokens", sa.Integer(), nullable=True))
    op.add_column("chats", sa.Column("router_output_tokens", sa.Integer(), nullable=True))

    # ------------------------------------------------------------------- evals
    op.create_table(
        "routing_eval_cases",
        _uuid("id"),
        _uuid("company_id"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("expected_intent_keys", J(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        _uuid("created_from_version_id", nullable=True),
        _uuid("chat_id", nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_routing_eval_cases_company_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_from_version_id"],
            ["guideline_versions.id"],
            name="fk_routing_eval_cases_version_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["chat_id"], ["chats.id"], name="fk_routing_eval_cases_chat_id", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_routing_eval_cases"),
        sa.CheckConstraint(
            "source IN ('synthetic', 'agent_feedback', 'manual')",
            name="ck_routing_eval_cases_source",
        ),
    )
    op.create_index("ix_routing_eval_cases_company_id", "routing_eval_cases", ["company_id"])

    op.create_table(
        "routing_eval_runs",
        _uuid("id"),
        *_scoped("routing_eval_runs"),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("metrics", J(), nullable=False),
        sa.Column("results", J(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_routing_eval_runs"),
        sa.CheckConstraint("mode IN ('retrieval', 'full')", name="ck_routing_eval_runs_mode"),
    )
    op.create_index("ix_routing_eval_runs_company_id", "routing_eval_runs", ["company_id"])

    # ------------------------------------------------------------ integrations
    def _company_table(name: str, *cols: sa.Column | sa.Constraint) -> None:  # type: ignore[type-arg]
        op.create_table(
            name,
            _uuid("id"),
            _uuid("company_id"),
            *cols,
            sa.ForeignKeyConstraint(
                ["company_id"], ["companies.id"], name=f"fk_{name}_company_id", ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id", name=f"pk_{name}"),
        )

    _company_table(
        "company_secrets",
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("ciphertext", sa.Text(), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("company_id", "name", name="uq_company_secrets_name"),
    )
    _company_table(
        "mcp_servers",
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("transport", sa.String(32), nullable=False),
        sa.Column("auth_secret_name", sa.String(100), nullable=True),
        sa.Column("discovered_tools", J(), nullable=True),
        sa.Column("last_discovered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("company_id", "name", name="uq_mcp_servers_name"),
    )
    _company_table(
        "capability_bindings",
        sa.Column("capability", sa.String(80), nullable=False),
        sa.Column("adapter_type", sa.String(20), nullable=False),
        sa.Column("config", J(), nullable=False),
        sa.Column("policy", J(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("last_test_result", J(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("company_id", "capability", name="uq_capability_bindings_capability"),
        sa.CheckConstraint(
            "adapter_type IN ('mcp', 'http', 'sql_readonly', 'mock')",
            name="ck_capability_bindings_adapter_type",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'tested', 'live')", name="ck_capability_bindings_status"
        ),
    )
    _company_table(
        "proposed_actions",
        _uuid("chat_id"),
        sa.Column("capability", sa.String(80), nullable=False),
        sa.Column("args", J(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("step_ref", sa.String(120), nullable=True),
        sa.Column("policy_check", J(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("result", J(), nullable=True),
        sa.Column("decided_by", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"], ["chats.id"], name="fk_proposed_actions_chat_id", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("idempotency_key", name="uq_proposed_actions_idempotency_key"),
        sa.CheckConstraint(
            "status IN ('proposed', 'denied', 'approved', 'rejected', 'executing', "
            "'executed', 'failed')",
            name="ck_proposed_actions_status",
        ),
    )
    op.create_index("ix_proposed_actions_chat_id", "proposed_actions", ["chat_id"])
    _company_table(
        "tool_calls",
        _uuid("chat_id", nullable=True),
        _uuid("proposed_action_id", nullable=True),
        sa.Column("capability", sa.String(80), nullable=False),
        sa.Column("adapter_type", sa.String(20), nullable=True),
        _uuid("binding_id", nullable=True),
        sa.Column("trigger", sa.String(16), nullable=False),
        sa.Column("args", J(), nullable=True),
        sa.Column("result", J(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_id"], ["chats.id"], name="fk_tool_calls_chat_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["proposed_action_id"],
            ["proposed_actions.id"],
            name="fk_tool_calls_proposed_action_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "trigger IN ('prefetch', 'agent', 'execution', 'test')", name="ck_tool_calls_trigger"
        ),
    )
    op.create_index("ix_tool_calls_chat_id", "tool_calls", ["chat_id"])
    op.create_index("ix_tool_calls_created_at", "tool_calls", ["created_at"])


def downgrade() -> None:
    for table in (
        "tool_calls",
        "proposed_actions",
        "capability_bindings",
        "mcp_servers",
        "company_secrets",
        "routing_eval_runs",
        "routing_eval_cases",
        "reference_facts",
        "company_rules",
        "intent_examples",
        "intents",
        "guideline_sections",
    ):
        op.drop_table(table)

    for col in (
        "router_output_tokens",
        "router_input_tokens",
        "latency_ms_tools",
        "router_prompt_version",
        "router_model",
        "feedback",
        "references",
        "routing",
        "primary_intent",
    ):
        op.drop_column("chats", col)
    op.alter_column("chats", "latency_ms_routing", new_column_name="latency_ms_retrieval")
    op.add_column("chats", sa.Column("retrieved_chunks", J(), nullable=True))

    gv = "guideline_versions"
    op.drop_constraint("ck_guideline_versions_review_status", gv, type_="check")
    for col in (
        "ingest_duration_ms",
        "compile_duration_ms",
        "routing_eval_run_id",
        "validation_warnings",
        "coverage",
        "fact_count",
        "intent_count",
        "section_count",
        "compile_prompt_version",
        "compiler_model",
        "embedding_dimensions",
        "source_format",
        "review_note",
        "reviewed_at",
        "reviewed_by",
        "activate_on_approval",
        "review_status",
    ):
        op.drop_column(gv, col)
    op.add_column(gv, sa.Column("chunk_count", sa.Integer(), nullable=True))
    op.add_column(gv, sa.Column("vector_collection", sa.Text(), nullable=True))
    op.add_column(gv, sa.Column("chunk_size", sa.Integer(), nullable=False, server_default="800"))
    op.add_column(
        gv, sa.Column("chunk_overlap", sa.Integer(), nullable=False, server_default="120")
    )
    # The extension is left installed (other schemas may use it).
