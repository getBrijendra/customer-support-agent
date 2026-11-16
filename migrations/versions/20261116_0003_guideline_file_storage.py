"""Original guideline files in object storage: location columns on guideline_versions

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-16

The bytes live in Supabase Storage (bucket created by the app at startup); Postgres only records
where. Existing rows keep NULLs: their originals were never stored (raw_text still has the text).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COLUMNS = ("storage_backend", "storage_bucket", "storage_key", "content_type")


def upgrade() -> None:
    op.add_column("guideline_versions", sa.Column("storage_backend", sa.String(32), nullable=True))
    op.add_column("guideline_versions", sa.Column("storage_bucket", sa.Text(), nullable=True))
    op.add_column("guideline_versions", sa.Column("storage_key", sa.Text(), nullable=True))
    op.add_column("guideline_versions", sa.Column("content_type", sa.Text(), nullable=True))


def downgrade() -> None:
    for name in reversed(COLUMNS):
        op.drop_column("guideline_versions", name)
