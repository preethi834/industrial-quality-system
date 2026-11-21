"""Mock QMS ticket model (what a real quality-management system would hold)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TicketStatus(StrEnum):
    OPEN = "open"
    CAUSE_CONFIRMED = "cause_confirmed"
    CLOSED = "closed"


class TicketEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    at: datetime
    kind: str  # created / updated
    changes: dict[str, Any] = Field(default_factory=dict)


class Ticket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str
    case_id: str
    title: str = Field(min_length=1, max_length=200)
    severity: int = Field(ge=1, le=3)  # 1 = high
    status: TicketStatus = TicketStatus.OPEN
    summary: str = ""
    candidate_causes: tuple[str, ...] = ()
    novel: bool = False
    confirmed_cause: str | None = None
    assignee: str | None = None
    created_at: datetime
    updated_at: datetime
    history: tuple[TicketEvent, ...] = ()


class TicketUpdate(BaseModel):
    """Fields an update may change (anything omitted stays as it is)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: TicketStatus | None = None
    confirmed_cause: str | None = None
    assignee: str | None = None
    summary: str | None = None
