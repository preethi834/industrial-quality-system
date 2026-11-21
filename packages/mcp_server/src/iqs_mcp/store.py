"""Ticket stores: in memory (tests, demos) and Postgres (``qms`` schema, service).

``create`` is idempotent per case: a second create for a case that already has an
open ticket returns that ticket. A re-run of the agent therefore never opens a
duplicate ticket.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from iqs_common.db import Base, session_scope
from iqs_mcp.models import Ticket, TicketEvent, TicketStatus, TicketUpdate

QMS_SCHEMA = "qms"


class TicketNotFoundError(KeyError):
    pass


class TicketStore(Protocol):
    def create(
        self,
        *,
        case_id: str,
        title: str,
        severity: int,
        summary: str,
        candidate_causes: tuple[str, ...],
        novel: bool,
    ) -> Ticket: ...

    def get(self, ticket_id: str) -> Ticket: ...

    def update(self, ticket_id: str, update: TicketUpdate) -> Ticket: ...


def _now() -> datetime:
    return datetime.now(UTC)


def _changes(update: TicketUpdate) -> dict[str, Any]:
    return {k: v for k, v in update.model_dump(mode="json").items() if v is not None}


class InMemoryTicketStore:
    def __init__(self) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        case_id: str,
        title: str,
        severity: int,
        summary: str,
        candidate_causes: tuple[str, ...],
        novel: bool,
    ) -> Ticket:
        with self._lock:
            for t in self._tickets.values():
                if t.case_id == case_id and t.status is TicketStatus.OPEN:
                    return t
            now = _now()
            ticket = Ticket(
                ticket_id=f"QMS-{len(self._tickets) + 1:06d}",
                case_id=case_id,
                title=title,
                severity=severity,
                summary=summary,
                candidate_causes=candidate_causes,
                novel=novel,
                created_at=now,
                updated_at=now,
                history=(TicketEvent(at=now, kind="created"),),
            )
            self._tickets[ticket.ticket_id] = ticket
            return ticket

    def get(self, ticket_id: str) -> Ticket:
        try:
            return self._tickets[ticket_id]
        except KeyError:
            raise TicketNotFoundError(ticket_id) from None

    def update(self, ticket_id: str, update: TicketUpdate) -> Ticket:
        with self._lock:
            old = self.get(ticket_id)
            changes = _changes(update)
            now = _now()
            new = old.model_copy(
                update={
                    **{k: getattr(update, k) for k in changes},
                    "updated_at": now,
                    "history": (*old.history, TicketEvent(at=now, kind="updated", changes=changes)),
                }
            )
            self._tickets[ticket_id] = new
            return new


# ------------------------------------------------------------------ Postgres (qms schema)

_STATUS_SQL = ", ".join(f"'{s.value}'" for s in TicketStatus)


class TicketRow(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="status"),
        CheckConstraint("severity BETWEEN 1 AND 3", name="severity"),
        {"schema": QMS_SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(200))
    severity: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), server_default=TicketStatus.OPEN.value)
    summary: Mapped[str] = mapped_column(Text, server_default="")
    candidate_causes: Mapped[list[str]] = mapped_column(JSONB, server_default="[]")
    novel: Mapped[bool] = mapped_column(server_default="false")
    confirmed_cause: Mapped[str | None] = mapped_column(String(64))
    assignee: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TicketEventRow(Base):
    __tablename__ = "ticket_events"
    __table_args__ = ({"schema": QMS_SCHEMA},)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey(f"{QMS_SCHEMA}.tickets.id"), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    kind: Mapped[str] = mapped_column(String(32))
    changes: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")


def _ticket_key(row_id: int) -> str:
    return f"QMS-{row_id:06d}"


def _row_id(ticket_id: str) -> int:
    try:
        prefix, number = ticket_id.split("-", 1)
        if prefix != "QMS":
            raise ValueError
        return int(number)
    except ValueError:
        raise TicketNotFoundError(ticket_id) from None


class SqlTicketStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def _load(self, session: Session, row: TicketRow) -> Ticket:
        events = session.scalars(
            select(TicketEventRow)
            .where(TicketEventRow.ticket_id == row.id)
            .order_by(TicketEventRow.id)
        ).all()
        return Ticket(
            ticket_id=_ticket_key(row.id),
            case_id=row.case_id,
            title=row.title,
            severity=row.severity,
            status=TicketStatus(row.status),
            summary=row.summary,
            candidate_causes=tuple(row.candidate_causes),
            novel=row.novel,
            confirmed_cause=row.confirmed_cause,
            assignee=row.assignee,
            created_at=row.created_at,
            updated_at=row.updated_at,
            history=tuple(TicketEvent(at=e.at, kind=e.kind, changes=e.changes) for e in events),
        )

    def create(
        self,
        *,
        case_id: str,
        title: str,
        severity: int,
        summary: str,
        candidate_causes: tuple[str, ...],
        novel: bool,
    ) -> Ticket:
        with session_scope(self._factory) as s:
            existing = s.scalars(
                select(TicketRow)
                .where(TicketRow.case_id == case_id, TicketRow.status == TicketStatus.OPEN.value)
                .with_for_update()
            ).first()
            if existing is not None:
                return self._load(s, existing)
            row = TicketRow(
                case_id=case_id,
                title=title,
                severity=severity,
                status=TicketStatus.OPEN.value,
                summary=summary,
                candidate_causes=list(candidate_causes),
                novel=novel,
            )
            s.add(row)
            s.flush()
            s.add(TicketEventRow(ticket_id=row.id, kind="created", changes={}))
            s.flush()
            s.refresh(row)
            return self._load(s, row)

    def get(self, ticket_id: str) -> Ticket:
        with session_scope(self._factory) as s:
            row = s.get(TicketRow, _row_id(ticket_id))
            if row is None:
                raise TicketNotFoundError(ticket_id)
            return self._load(s, row)

    def update(self, ticket_id: str, update: TicketUpdate) -> Ticket:
        with session_scope(self._factory) as s:
            row = s.get(TicketRow, _row_id(ticket_id), with_for_update=True)
            if row is None:
                raise TicketNotFoundError(ticket_id)
            changes = _changes(update)
            for key, value in changes.items():
                setattr(row, key, value)
            row.updated_at = _now()
            s.add(TicketEventRow(ticket_id=row.id, kind="updated", changes=changes))
            s.flush()
            s.refresh(row)
            return self._load(s, row)
