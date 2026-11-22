"""Feedback persistence (step 7): Dirichlet counts, calibration candidate pool, applied log.

One ``apply`` call writes, atomically and at most once per case:

- ``diagnosis.feedback_applied``  (case_id PK: the idempotency key),
- ``diagnosis.cause_counts``      (+1 for (context, cause); prior alpha_0 = 1 is implicit),
- ``diagnosis.calibration_candidates`` (the case, for the *next* versioned calibration set),
- ``platform.events``             (``case_confirmed``, vision cases only).

The active calibration run is never touched here (blueprint §2.8 A5, owner decision D3).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import DateTime, Integer, String, Uuid, func, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from iqs_common.db import Base, session_scope
from iqs_common.events import CaseConfirmedEvent
from iqs_common.outbox_worker import InMemoryOutbox, SqlOutbox
from iqs_diagnosis.persistence.repository import DIAGNOSIS_SCHEMA


@dataclass(frozen=True)
class FeedbackRecord:
    case_id: str
    source: str
    cause_code: str | None
    engineer: str
    confirmed_at: datetime
    count_context: str | None = None
    count_before: int | None = None
    count_after: int | None = None
    category: str | None = None
    defect_class_name: str | None = None
    event_id: uuid.UUID | None = None
    candidate_added: bool = False
    graph_written_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "cause_code": self.cause_code,
            "engineer": self.engineer,
            "confirmed_at": self.confirmed_at.isoformat(),
            "count_context": self.count_context,
            "count_before": self.count_before,
            "count_after": self.count_after,
            "category": self.category,
            "defect_class_name": self.defect_class_name,
            "event_id": None if self.event_id is None else str(self.event_id),
            "candidate_added": self.candidate_added,
            "graph_written_at": (
                None if self.graph_written_at is None else self.graph_written_at.isoformat()
            ),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FeedbackRecord:
        def ts(v: str | None) -> datetime | None:
            return None if v is None else datetime.fromisoformat(v)

        confirmed = ts(d["confirmed_at"])
        assert confirmed is not None
        return cls(
            case_id=d["case_id"],
            source=d["source"],
            cause_code=d["cause_code"],
            engineer=d["engineer"],
            confirmed_at=confirmed,
            count_context=d["count_context"],
            count_before=d["count_before"],
            count_after=d["count_after"],
            category=d["category"],
            defect_class_name=d["defect_class_name"],
            event_id=None if d["event_id"] is None else uuid.UUID(d["event_id"]),
            candidate_added=d["candidate_added"],
            graph_written_at=ts(d["graph_written_at"]),
        )


class FeedbackStore(Protocol):
    def get(self, case_id: str) -> FeedbackRecord | None: ...
    def apply(
        self, record: FeedbackRecord, event: CaseConfirmedEvent | None
    ) -> tuple[FeedbackRecord, bool]:
        """Write the feedback once; ``(stored record, True)`` the first time, else the
        record stored earlier and ``False``."""
        ...

    def count(self, context: str, cause_code: str) -> int: ...
    def counts(self) -> list[tuple[str, str, int]]: ...
    def candidates(self) -> list[dict[str, Any]]: ...
    def mark_graph_written(self, case_id: str, at: datetime) -> None: ...
    def pending_graph(self) -> list[FeedbackRecord]: ...


def _counted(record: FeedbackRecord, before: int) -> FeedbackRecord:
    return replace(record, count_before=before, count_after=before + 1)


class InMemoryFeedbackStore:
    def __init__(self, outbox: InMemoryOutbox | None = None) -> None:
        self.outbox = outbox or InMemoryOutbox()
        self._applied: dict[str, FeedbackRecord] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._candidates: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, case_id: str) -> FeedbackRecord | None:
        with self._lock:
            return self._applied.get(case_id)

    def apply(
        self, record: FeedbackRecord, event: CaseConfirmedEvent | None
    ) -> tuple[FeedbackRecord, bool]:
        with self._lock:
            done = self._applied.get(record.case_id)
            if done is not None:
                return done, False
            if event is not None:
                self.outbox.append(event)  # first, so a duplicate leaves no partial state
            if record.cause_code is not None and record.count_context is not None:
                key = (record.count_context, record.cause_code)
                record = _counted(record, self._counts.get(key, 0))
                self._counts[key] = record.count_after or 0
            if record.candidate_added and record.cause_code is not None:
                self._candidates[record.case_id] = {
                    "case_id": record.case_id,
                    "source": record.source,
                    "cause_code": record.cause_code,
                    "used_in_calibration_run": None,
                }
            self._applied[record.case_id] = record
            return record, True

    def count(self, context: str, cause_code: str) -> int:
        with self._lock:
            return self._counts.get((context, cause_code), 0)

    def counts(self) -> list[tuple[str, str, int]]:
        with self._lock:
            return sorted((c, k, n) for (c, k), n in self._counts.items())

    def candidates(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(v) for _, v in sorted(self._candidates.items())]

    def mark_graph_written(self, case_id: str, at: datetime) -> None:
        with self._lock:
            self._applied[case_id] = replace(self._applied[case_id], graph_written_at=at)

    def pending_graph(self) -> list[FeedbackRecord]:
        with self._lock:
            return [
                r
                for r in self._applied.values()
                if r.cause_code is not None and r.graph_written_at is None
            ]


# --- Postgres (tables from migration 0005; tested in 7g) --------------------------------


class CauseCountRow(Base):
    __tablename__ = "cause_counts"
    __table_args__ = ({"schema": DIAGNOSIS_SCHEMA},)

    context: Mapped[str] = mapped_column(String(160), primary_key=True)
    cause_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    count: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class CalibrationCandidateRow(Base):
    __tablename__ = "calibration_candidates"
    __table_args__ = ({"schema": DIAGNOSIS_SCHEMA},)

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    cause_code: Mapped[str] = mapped_column(String(64))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    used_in_calibration_run: Mapped[str | None] = mapped_column(String(96))


class FeedbackAppliedRow(Base):
    __tablename__ = "feedback_applied"
    __table_args__ = ({"schema": DIAGNOSIS_SCHEMA},)

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    record: Mapped[dict[str, Any]] = mapped_column(JSONB)
    event_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    graph_written_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SqlFeedbackStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def get(self, case_id: str) -> FeedbackRecord | None:
        with session_scope(self._factory) as s:
            row = s.get(FeedbackAppliedRow, case_id)
            return None if row is None else self._record(row)

    @staticmethod
    def _record(row: FeedbackAppliedRow) -> FeedbackRecord:
        rec = FeedbackRecord.from_json(row.record)
        return replace(rec, graph_written_at=row.graph_written_at)

    def apply(
        self, record: FeedbackRecord, event: CaseConfirmedEvent | None
    ) -> tuple[FeedbackRecord, bool]:
        with session_scope(self._factory) as s:
            claimed = s.execute(
                insert(FeedbackAppliedRow)
                .values(case_id=record.case_id, record=record.to_json())
                .on_conflict_do_nothing(index_elements=["case_id"])
                .returning(FeedbackAppliedRow.case_id)
            ).first()
            if claimed is None:
                row = s.get(FeedbackAppliedRow, record.case_id)
                assert row is not None
                return self._record(row), False
            if record.cause_code is not None and record.count_context is not None:
                after = s.execute(
                    insert(CauseCountRow)
                    .values(context=record.count_context, cause_code=record.cause_code, count=1)
                    .on_conflict_do_update(
                        index_elements=["context", "cause_code"],
                        set_={"count": CauseCountRow.count + 1, "updated_at": func.now()},
                    )
                    .returning(CauseCountRow.count)
                ).scalar_one()
                record = _counted(record, int(after) - 1)
            if record.candidate_added and record.cause_code is not None:
                s.add(
                    CalibrationCandidateRow(
                        case_id=record.case_id, source=record.source, cause_code=record.cause_code
                    )
                )
            if event is not None:
                SqlOutbox.append_in(s, event)
            s.execute(
                update(FeedbackAppliedRow)
                .where(FeedbackAppliedRow.case_id == record.case_id)
                .values(record=record.to_json(), event_id=record.event_id)
            )
            return record, True

    def count(self, context: str, cause_code: str) -> int:
        with session_scope(self._factory) as s:
            row = s.get(CauseCountRow, (context, cause_code))
            return 0 if row is None else row.count

    def counts(self) -> list[tuple[str, str, int]]:
        with session_scope(self._factory) as s:
            rows = s.scalars(
                select(CauseCountRow).order_by(CauseCountRow.context, CauseCountRow.cause_code)
            ).all()
            return [(r.context, r.cause_code, r.count) for r in rows]

    def candidates(self) -> list[dict[str, Any]]:
        with session_scope(self._factory) as s:
            rows = s.scalars(
                select(CalibrationCandidateRow).order_by(CalibrationCandidateRow.case_id)
            ).all()
            return [
                {
                    "case_id": r.case_id,
                    "source": r.source,
                    "cause_code": r.cause_code,
                    "used_in_calibration_run": r.used_in_calibration_run,
                }
                for r in rows
            ]

    def mark_graph_written(self, case_id: str, at: datetime) -> None:
        with session_scope(self._factory) as s:
            s.execute(
                update(FeedbackAppliedRow)
                .where(FeedbackAppliedRow.case_id == case_id)
                .values(graph_written_at=at)
            )

    def pending_graph(self) -> list[FeedbackRecord]:
        with session_scope(self._factory) as s:
            rows = s.scalars(
                select(FeedbackAppliedRow).where(FeedbackAppliedRow.graph_written_at.is_(None))
            ).all()
            return [r for r in map(self._record, rows) if r.cause_code is not None]


def utcnow() -> datetime:
    return datetime.now(UTC)
