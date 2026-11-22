"""Step 7 quality loop: vision discovery tables, diagnosis inbox + feedback tables

vision.clusters, vision.discovery_observations, vision.processed_events,
diagnosis.cases.source_cluster_id (UNIQUE), diagnosis.processed_events,
diagnosis.cause_counts, diagnosis.calibration_candidates, diagnosis.feedback_applied.

Numbering: step-2 may also add a 0005. Whichever branch merges second
renumbers its file and fixes down_revision (owner decision 2026-11-21).

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB(astext_type=sa.Text())
REALS: postgresql.ARRAY[float] = postgresql.ARRAY(sa.REAL())
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    # --- vision --------------------------------------------------------------------------
    op.create_table(
        "clusters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("centroid", REALS, nullable=False),
        sa.Column("radius", sa.Float(), nullable=False),
        sa.Column("first_seen", TS, nullable=False),
        sa.Column("last_seen", TS, nullable=False),
        sa.Column("representative_id", sa.Uuid(), nullable=False),
        sa.Column("alerted_event_id", sa.Uuid(), nullable=False),
        sa.Column("defect_class_name", sa.String(length=64), nullable=True),
        sa.Column("cause_code", sa.String(length=64), nullable=True),
        sa.Column("case_id", sa.Uuid(), nullable=True),
        sa.Column("named_at", TS, nullable=True),
        sa.Column("created_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('alerted', 'named')", name=op.f("ck_clusters_status")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clusters")),
        sa.UniqueConstraint("alerted_event_id", name=op.f("uq_clusters_alerted_event_id")),
        schema="vision",
    )
    op.create_index(op.f("ix_vision_clusters_category"), "clusters", ["category"], schema="vision")
    op.create_table(
        "discovery_observations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("observed_at", TS, nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("routing", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("embedding", REALS, nullable=True),
        sa.Column("part_id", sa.String(length=64), nullable=True),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.Column("line_id", sa.String(length=64), nullable=True),
        sa.Column("known_cluster_id", sa.Uuid(), nullable=True),
        sa.Column("cluster_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "routing IN ('normal', 'undecided', 'known', 'unknown_buffered')",
            name=op.f("ck_discovery_observations_routing"),
        ),
        sa.ForeignKeyConstraint(
            ["cluster_id"],
            ["vision.clusters.id"],
            name=op.f("fk_discovery_observations_cluster_id_clusters"),
        ),
        sa.ForeignKeyConstraint(
            ["known_cluster_id"],
            ["vision.clusters.id"],
            name=op.f("fk_discovery_observations_known_cluster_id_clusters"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_discovery_observations")),
        schema="vision",
    )
    op.create_index(
        op.f("ix_vision_discovery_observations_category"),
        "discovery_observations",
        ["category"],
        schema="vision",
    )
    op.create_index(
        op.f("ix_vision_discovery_observations_cluster_id"),
        "discovery_observations",
        ["cluster_id"],
        schema="vision",
    )
    op.create_table(
        "processed_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("processed_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_processed_events")),
        schema="vision",
    )

    # --- diagnosis -----------------------------------------------------------------------
    op.add_column(
        "cases",
        sa.Column("source_cluster_id", sa.String(length=64), nullable=True),
        schema="diagnosis",
    )
    op.create_unique_constraint(
        op.f("uq_cases_source_cluster_id"), "cases", ["source_cluster_id"], schema="diagnosis"
    )
    op.create_table(
        "processed_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=True),
        sa.Column("processed_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_processed_events")),
        schema="diagnosis",
    )
    op.create_table(
        "cause_counts",
        sa.Column("context", sa.String(length=160), nullable=False),
        sa.Column("cause_code", sa.String(length=64), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("updated_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("context", "cause_code", name=op.f("pk_cause_counts")),
        schema="diagnosis",
    )
    op.create_table(
        "calibration_candidates",
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("cause_code", sa.String(length=64), nullable=False),
        sa.Column("added_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.Column("used_in_calibration_run", sa.String(length=96), nullable=True),
        sa.PrimaryKeyConstraint("case_id", name=op.f("pk_calibration_candidates")),
        schema="diagnosis",
    )
    op.create_table(
        "feedback_applied",
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("record", JSONB, nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=True),
        sa.Column("applied_at", TS, server_default=sa.text("now()"), nullable=False),
        sa.Column("graph_written_at", TS, nullable=True),
        sa.PrimaryKeyConstraint("case_id", name=op.f("pk_feedback_applied")),
        schema="diagnosis",
    )


def downgrade() -> None:
    op.drop_table("feedback_applied", schema="diagnosis")
    op.drop_table("calibration_candidates", schema="diagnosis")
    op.drop_table("cause_counts", schema="diagnosis")
    op.drop_table("processed_events", schema="diagnosis")
    op.drop_constraint(
        op.f("uq_cases_source_cluster_id"), "cases", schema="diagnosis", type_="unique"
    )
    op.drop_column("cases", "source_cluster_id", schema="diagnosis")
    op.drop_table("processed_events", schema="vision")
    op.drop_index(
        op.f("ix_vision_discovery_observations_cluster_id"),
        table_name="discovery_observations",
        schema="vision",
    )
    op.drop_index(
        op.f("ix_vision_discovery_observations_category"),
        table_name="discovery_observations",
        schema="vision",
    )
    op.drop_table("discovery_observations", schema="vision")
    op.drop_index(op.f("ix_vision_clusters_category"), table_name="clusters", schema="vision")
    op.drop_table("clusters", schema="vision")
