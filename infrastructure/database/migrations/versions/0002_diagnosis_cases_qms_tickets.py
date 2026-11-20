"""diagnosis cases, case causes, 8D reports; qms tickets and ticket events (step 3c)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB(astext_type=sa.Text())
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("symptoms", JSONB, server_default="[]", nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("novelty_flag", sa.Boolean(), nullable=True),
        sa.Column("confirmed_cause_code", sa.String(length=64), nullable=True),
        sa.Column("case_json", JSONB, nullable=False),
        sa.Column("created_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('registered', 'awaiting_confirmation', 'confirmed')",
            name=op.f("ck_cases_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cases")),
        schema="diagnosis",
    )
    op.create_table(
        "case_causes",
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("cause_code", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("in_conformal_set", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(
            ["case_id"], ["diagnosis.cases.id"], name=op.f("fk_case_causes_case_id_cases")
        ),
        sa.PrimaryKeyConstraint("case_id", "cause_code", name=op.f("pk_case_causes")),
        schema="diagnosis",
    )
    op.create_table(
        "reports_8d",
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("content", JSONB, nullable=False),
        sa.Column("completeness", sa.Float(), nullable=False),
        sa.Column("created_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"], ["diagnosis.cases.id"], name=op.f("fk_reports_8d_case_id_cases")
        ),
        sa.PrimaryKeyConstraint("case_id", "version", "language", name=op.f("pk_reports_8d")),
        schema="diagnosis",
    )
    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="open", nullable=False),
        sa.Column("summary", sa.Text(), server_default="", nullable=False),
        sa.Column("candidate_causes", JSONB, server_default="[]", nullable=False),
        sa.Column("novel", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("confirmed_cause", sa.String(length=64), nullable=True),
        sa.Column("assignee", sa.String(length=128), nullable=True),
        sa.Column("created_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('open', 'cause_confirmed', 'closed')", name=op.f("ck_tickets_status")
        ),
        sa.CheckConstraint("severity BETWEEN 1 AND 3", name=op.f("ck_tickets_severity")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tickets")),
        schema="qms",
    )
    op.create_index(op.f("ix_qms_tickets_case_id"), "tickets", ["case_id"], schema="qms")
    op.create_table(
        "ticket_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticket_id", sa.Integer(), nullable=False),
        sa.Column("at", TS, server_default=sa.text("now()"), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("changes", JSONB, server_default="{}", nullable=False),
        sa.ForeignKeyConstraint(
            ["ticket_id"], ["qms.tickets.id"], name=op.f("fk_ticket_events_ticket_id_tickets")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ticket_events")),
        schema="qms",
    )
    op.create_index(
        op.f("ix_qms_ticket_events_ticket_id"), "ticket_events", ["ticket_id"], schema="qms"
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_qms_ticket_events_ticket_id"), table_name="ticket_events", schema="qms")
    op.drop_table("ticket_events", schema="qms")
    op.drop_index(op.f("ix_qms_tickets_case_id"), table_name="tickets", schema="qms")
    op.drop_table("tickets", schema="qms")
    op.drop_table("reports_8d", schema="diagnosis")
    op.drop_table("case_causes", schema="diagnosis")
    op.drop_table("cases", schema="diagnosis")
