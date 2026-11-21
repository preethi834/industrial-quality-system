"""Outbox delivery (step 7): claim pending events, hand them to a handler, record the result.

Delivery is **at-least-once**. A handler may see the same event twice (a crash after the
handler committed but before the row was marked processed), so every handler must be
idempotent on ``event_id`` (ADR-015). Each event type has exactly one consumer, so the single
``status`` column of ``platform.events`` is enough; several consumers of one type would need a
per-consumer delivery table.

Two stores implement the same contract:

- :class:`InMemoryOutbox` for unit tests and the in-process quality loop;
- :class:`SqlOutbox` for Postgres. It locks the claimed rows with ``FOR UPDATE SKIP LOCKED``
  for the duration of the batch, so two workers never process the same row at the same time,
  and the row update (processed / attempts / failed) commits with that lock released.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from iqs_common.events import Event, EventType, parse_event
from iqs_common.log import get_logger
from iqs_common.outbox import OutboxEvent, OutboxStatus

Handler = Callable[[Event], None]
DEFAULT_MAX_ATTEMPTS = 5

_log = get_logger("iqs_common.outbox_worker")


@dataclass
class BatchResult:
    processed: list[UUID] = field(default_factory=list)
    retried: list[UUID] = field(default_factory=list)
    failed: list[UUID] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.processed) + len(self.retried) + len(self.failed)


class OutboxStore(Protocol):
    def append(self, event: Event) -> None:
        """Add one pending event (producers that are not inside a SQL transaction)."""
        ...

    def process_pending(
        self,
        types: Collection[EventType],
        handler: Handler,
        *,
        limit: int = 50,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> BatchResult:
        """Deliver up to ``limit`` pending events of ``types``, oldest first."""
        ...


def _deliver(raw: dict[str, object], handler: Handler) -> str | None:
    """Run the handler; ``None`` on success, else the error text (never raises)."""
    try:
        event = parse_event(raw)
    except ValidationError as exc:
        return f"invalid event: {exc.error_count()} validation error(s)"
    try:
        handler(event)
    except Exception as exc:  # noqa: BLE001 - the worker must survive any handler error
        return f"{type(exc).__name__}: {exc}"
    return None


@dataclass
class _Row:
    id: UUID
    type: str
    payload: dict[str, object]
    status: str = OutboxStatus.PENDING.value
    attempts: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None = None
    last_error: str | None = None


class InMemoryOutbox:
    """Process-local outbox with the same semantics as :class:`SqlOutbox`."""

    def __init__(self) -> None:
        self._rows: dict[UUID, _Row] = {}
        self._lock = threading.RLock()

    def append(self, event: Event) -> None:
        with self._lock:
            if event.event_id in self._rows:  # same as the primary-key violation in SQL
                raise ValueError(f"duplicate event_id {event.event_id}")
            self._rows[event.event_id] = _Row(
                id=event.event_id, type=event.type.value, payload=event.model_dump(mode="json")
            )

    def process_pending(
        self,
        types: Collection[EventType],
        handler: Handler,
        *,
        limit: int = 50,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> BatchResult:
        wanted = {t.value for t in types}
        result = BatchResult()
        with self._lock:  # one batch at a time, like the row locks in SQL
            rows = [
                r
                for r in self._rows.values()
                if r.status == OutboxStatus.PENDING.value and r.type in wanted
            ]
            rows.sort(key=lambda r: (r.created_at, r.id))
            for row in rows[:limit]:
                error = _deliver(row.payload, handler)
                _record(row, error, max_attempts, result)
        return result

    # --- inspection helpers (tests, demo) -------------------------------------------------

    def rows(self, type_: EventType | None = None) -> list[_Row]:
        with self._lock:
            return [r for r in self._rows.values() if type_ is None or r.type == type_.value]

    def requeue(self, event_id: UUID) -> None:
        """Set a row back to pending (simulates a redelivery after a lost acknowledgement)."""
        with self._lock:
            row = self._rows[event_id]
            row.status = OutboxStatus.PENDING.value
            row.processed_at = None


def _record(
    row: _Row | OutboxEvent, error: str | None, max_attempts: int, out: BatchResult
) -> None:
    if error is None:
        row.status = OutboxStatus.PROCESSED.value
        row.processed_at = datetime.now(UTC)
        out.processed.append(row.id)
        return
    row.attempts = (row.attempts or 0) + 1
    if row.attempts >= max_attempts:
        row.status = OutboxStatus.FAILED.value
        out.failed.append(row.id)
    else:
        out.retried.append(row.id)
    if isinstance(row, _Row):
        row.last_error = error
    _log.warning(
        "outbox_delivery_failed",
        event_id=str(row.id),
        event_type=row.type,
        attempts=row.attempts,
        status=row.status,
        error=error,
    )


class SqlOutbox:
    """``platform.events`` in Postgres (needs ``FOR UPDATE SKIP LOCKED``)."""

    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def append(self, event: Event) -> None:
        with self._factory.begin() as s:
            s.add(OutboxEvent.from_event(event))

    @staticmethod
    def append_in(session: Session, event: Event) -> None:
        """Add the event inside the caller's transaction (the transactional-outbox write)."""
        session.add(OutboxEvent.from_event(event))

    def process_pending(
        self,
        types: Collection[EventType],
        handler: Handler,
        *,
        limit: int = 50,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> BatchResult:
        result = BatchResult()
        with self._factory.begin() as s:
            rows = (
                s.execute(
                    select(OutboxEvent)
                    .where(
                        OutboxEvent.status == OutboxStatus.PENDING.value,
                        OutboxEvent.type.in_([t.value for t in types]),
                    )
                    .order_by(OutboxEvent.created_at, OutboxEvent.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                .scalars()
                .all()
            )
            for row in rows:
                # The handler uses its own sessions/transactions; it is idempotent.
                error = _deliver(row.payload, handler)
                _record(row, error, max_attempts, result)
        return result


def run_worker(
    store: OutboxStore,
    types: Collection[EventType],
    handler: Handler,
    *,
    poll_s: float = 1.0,
    stop: threading.Event | None = None,
    max_batches: int | None = None,
) -> int:
    """Poll until ``stop`` is set (or ``max_batches`` batches ran); returns events handled."""
    stop = stop or threading.Event()
    handled = 0
    batches = 0
    while not stop.is_set():
        result = store.process_pending(types, handler)
        handled += result.total
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
        if result.total == 0:
            stop.wait(poll_s)
    return handled
