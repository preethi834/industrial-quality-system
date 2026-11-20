"""diagnosis.calibration_runs; case_causes.calibration_run_id (step 4)

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "calibration_runs",
        sa.Column("id", sa.String(length=96), nullable=False),
        sa.Column("method", sa.String(length=16), nullable=False),
        sa.Column("alpha", sa.Float(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column("qhat", sa.Float(), nullable=False),
        sa.Column("n_calibration", sa.Integer(), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_calibration_runs")),
        schema="diagnosis",
    )
    op.add_column(
        "case_causes",
        sa.Column("calibration_run_id", sa.String(length=96), nullable=True),
        schema="diagnosis",
    )
    op.create_foreign_key(
        op.f("fk_case_causes_calibration_run_id_calibration_runs"),
        "case_causes",
        "calibration_runs",
        ["calibration_run_id"],
        ["id"],
        source_schema="diagnosis",
        referent_schema="diagnosis",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_case_causes_calibration_run_id_calibration_runs"),
        "case_causes",
        schema="diagnosis",
        type_="foreignkey",
    )
    op.drop_column("case_causes", "calibration_run_id", schema="diagnosis")
    op.drop_table("calibration_runs", schema="diagnosis")
