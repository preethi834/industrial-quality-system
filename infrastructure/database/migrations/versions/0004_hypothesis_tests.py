"""diagnosis.hypothesis_tests (step 5b scientist results)

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "hypothesis_tests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("cause_code", sa.String(length=64), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("sql_template", sa.String(length=64), nullable=False),
        sa.Column("sql", sa.Text(), nullable=False),
        sa.Column("params", JSONB, nullable=False),
        sa.Column("test_name", sa.String(length=64), nullable=False),
        sa.Column("target", sa.String(length=64), nullable=True),
        sa.Column("statistic", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("p_adjusted", sa.Float(), nullable=True),
        sa.Column("q", sa.Float(), nullable=False),
        sa.Column("effect_size", JSONB, nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("evidence", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "verdict IN ('supported', 'rejected', 'inconclusive')",
            name=op.f("ck_hypothesis_tests_verdict"),
        ),
        sa.CheckConstraint("mode IN ('top3', 'all_causes')", name=op.f("ck_hypothesis_tests_mode")),
        sa.ForeignKeyConstraint(
            ["case_id"], ["diagnosis.cases.id"], name=op.f("fk_hypothesis_tests_case_id_cases")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_hypothesis_tests")),
        schema="diagnosis",
    )
    op.create_index(
        op.f("ix_diagnosis_hypothesis_tests_case_id"),
        "hypothesis_tests",
        ["case_id"],
        schema="diagnosis",
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_diagnosis_hypothesis_tests_case_id"),
        table_name="hypothesis_tests",
        schema="diagnosis",
    )
    op.drop_table("hypothesis_tests", schema="diagnosis")
