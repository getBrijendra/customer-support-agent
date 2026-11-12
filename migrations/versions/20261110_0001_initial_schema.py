"""initial schema: companies, guideline_versions, chats

Revision ID: 0001
Revises:
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("active_version_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_companies"),
        sa.UniqueConstraint("slug", name="uq_companies_slug"),
        sa.CheckConstraint("slug ~ '^[a-z0-9-]{2,50}$'", name="ck_companies_slug_format"),
    )
    op.create_table(
        "guideline_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("company_id", sa.Uuid(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("vector_collection", sa.Text(), nullable=True),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("uploaded_by", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_guideline_versions_company_id_companies",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_guideline_versions"),
        sa.UniqueConstraint(
            "company_id", "version_number", name="uq_guideline_versions_company_ver"
        ),
        sa.CheckConstraint(
            "status IN ('processing', 'ready', 'failed')", name="ck_guideline_versions_status"
        ),
    )
    op.create_index("ix_guideline_versions_company_id", "guideline_versions", ["company_id"])
    op.create_index(
        "ix_guideline_versions_company_sha", "guideline_versions", ["company_id", "content_sha256"]
    )
    # Circular FK: companies.active_version_id -> guideline_versions.id
    op.create_foreign_key(
        "fk_companies_active_version_id_guideline_versions",
        "companies",
        "guideline_versions",
        ["active_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "chats",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("company_id", sa.Uuid(), nullable=False),
        sa.Column("guideline_version_id", sa.Uuid(), nullable=True),
        sa.Column("guideline_version_number", sa.Integer(), nullable=True),
        sa.Column("customer_message", sa.Text(), nullable=False),
        sa.Column("agent_notes", sa.Text(), nullable=True),
        sa.Column("channel", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("recommended_response", sa.Text(), nullable=True),
        sa.Column("guidance", postgresql.JSONB(), nullable=True),
        sa.Column("retrieved_chunks", postgresql.JSONB(), nullable=True),
        sa.Column("llm_model", sa.Text(), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("latency_ms_total", sa.Integer(), nullable=True),
        sa.Column("latency_ms_retrieval", sa.Integer(), nullable=True),
        sa.Column("latency_ms_llm", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_chats_company_id_companies",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["guideline_version_id"],
            ["guideline_versions.id"],
            name="fk_chats_guideline_version_id_guideline_versions",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chats"),
        sa.CheckConstraint("status IN ('success', 'error')", name="ck_chats_status"),
    )
    op.create_index("ix_chats_created_at", "chats", ["created_at"])
    op.create_index("ix_chats_company_created", "chats", ["company_id", sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_index("ix_chats_company_created", table_name="chats")
    op.drop_index("ix_chats_created_at", table_name="chats")
    op.drop_table("chats")
    op.drop_constraint(
        "fk_companies_active_version_id_guideline_versions", "companies", type_="foreignkey"
    )
    op.drop_index("ix_guideline_versions_company_sha", table_name="guideline_versions")
    op.drop_index("ix_guideline_versions_company_id", table_name="guideline_versions")
    op.drop_table("guideline_versions")
    op.drop_table("companies")
