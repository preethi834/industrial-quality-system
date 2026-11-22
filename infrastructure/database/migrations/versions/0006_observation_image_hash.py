"""Step 7 (KNOWN_ISSUES #46): vision.discovery_observations.image_sha256

The discovery trigger counts distinct images, not observations, so repeats of one image
cannot make a cluster on their own. Nullable: rows written before this revision have no hash
and are treated as distinct images.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "discovery_observations",
        sa.Column("image_sha256", sa.String(length=64), nullable=True),
        schema="vision",
    )


def downgrade() -> None:
    op.drop_column("discovery_observations", "image_sha256", schema="vision")
