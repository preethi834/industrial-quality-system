"""baseline: vector extension, service schemas, platform.events outbox

Revision ID: 0001
Revises:
Create Date: 2026-11-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMAS = ("diagnosis", "vision", "qms", "platform")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for schema in SCHEMAS:
        op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    op.create_table(
        "events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'processed', 'failed')", name=op.f("ck_events_status")
        ),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_events_attempts_non_negative")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_events")),
        schema="platform",
    )
    op.create_index(
        "ix_events_status_created_at",
        "events",
        ["status", "created_at"],
        unique=False,
        schema="platform",
    )


def downgrade() -> None:
    op.drop_index("ix_events_status_created_at", table_name="events", schema="platform")
    op.drop_table("events", schema="platform")
    # RESTRICT (the default): fails loudly if a later migration left objects behind.
    for schema in reversed(SCHEMAS):
        op.execute(f"DROP SCHEMA IF EXISTS {schema}")
    op.execute("DROP EXTENSION IF EXISTS vector")
