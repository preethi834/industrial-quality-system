"""Transactional outbox: the platform.events table.

A producer writes its domain rows and one OutboxEvent in the same transaction;
a worker later delivers pending rows (at-least-once, see iqs_common.events).
The row id is the event's event_id, so a duplicate insert of the same event
fails on the primary key instead of being delivered twice.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from iqs_common.db import Base
from iqs_common.events import Event

OUTBOX_SCHEMA = "platform"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


_STATUS_SQL = ", ".join(f"'{s.value}'" for s in OutboxStatus)


class OutboxEvent(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="status"),
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        # The worker's query: oldest pending rows first.
        Index("ix_events_status_created_at", "status", "created_at"),
        {"schema": OUTBOX_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(
        String(16), default=OutboxStatus.PENDING.value, server_default=OutboxStatus.PENDING.value
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @classmethod
    def from_event(cls, event: Event) -> OutboxEvent:
        """Row for one event; payload is the full JSON envelope, readable by parse_event."""
        return cls(
            id=event.event_id,
            type=event.type.value,
            payload=event.model_dump(mode="json"),
            status=OutboxStatus.PENDING.value,
            attempts=0,
        )
