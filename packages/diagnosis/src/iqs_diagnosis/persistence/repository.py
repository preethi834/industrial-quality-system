"""Case repository: in memory (tests, evaluation) and Postgres (``diagnosis`` schema).

Tables follow blueprint §2.4 with two simplifications for step 3c: causes are stored by
taxonomy code (the ``cause_taxonomy`` table is not created yet), and
``in_conformal_set`` / ``calibration_run_id`` stay empty until step 4.
"""

from __future__ import annotations

import threading
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from iqs_common.db import Base, session_scope
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.reporting.eight_d import EightDReport

DIAGNOSIS_SCHEMA = "diagnosis"


class CaseStatus(StrEnum):
    REGISTERED = "registered"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"


class CaseRepository(Protocol):
    def add(self, case: DiagnosisCase) -> None: ...
    def get(self, case_id: str) -> DiagnosisCase | None: ...
    def status(self, case_id: str) -> CaseStatus | None: ...
    def save_ranking(self, case_id: str, top3: list[tuple[str, float]], novel: bool) -> None: ...
    def save_report(self, report: EightDReport, completeness: float) -> None: ...
    def save_confirmation(self, case_id: str, cause_code: str | None) -> None: ...


class InMemoryCaseRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cases: dict[str, DiagnosisCase] = {}
        self.statuses: dict[str, CaseStatus] = {}
        self.rankings: dict[str, dict[str, Any]] = {}
        self.reports: dict[str, list[EightDReport]] = {}
        self.confirmations: dict[str, str | None] = {}

    def add(self, case: DiagnosisCase) -> None:
        with self._lock:
            if case.case_id not in self.cases:
                self.cases[case.case_id] = case
                self.statuses[case.case_id] = CaseStatus.REGISTERED

    def get(self, case_id: str) -> DiagnosisCase | None:
        return self.cases.get(case_id)

    def status(self, case_id: str) -> CaseStatus | None:
        return self.statuses.get(case_id)

    def save_ranking(self, case_id: str, top3: list[tuple[str, float]], novel: bool) -> None:
        with self._lock:
            self.rankings[case_id] = {"top3": list(top3), "novel": novel}
            self.statuses[case_id] = CaseStatus.AWAITING_CONFIRMATION

    def save_report(self, report: EightDReport, completeness: float) -> None:
        with self._lock:
            self.reports.setdefault(report.case_id, []).append(report)

    def save_confirmation(self, case_id: str, cause_code: str | None) -> None:
        with self._lock:
            self.confirmations[case_id] = cause_code
            self.statuses[case_id] = CaseStatus.CONFIRMED


# ------------------------------------------------------------------ Postgres

_STATUS_SQL = ", ".join(f"'{s.value}'" for s in CaseStatus)


class CaseRow(Base):
    __tablename__ = "cases"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="status"),
        {"schema": DIAGNOSIS_SCHEMA},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    symptoms: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    status: Mapped[str] = mapped_column(String(32))
    novelty_flag: Mapped[bool | None] = mapped_column()
    confirmed_cause_code: Mapped[str | None] = mapped_column(String(64))
    case_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CaseCauseRow(Base):
    __tablename__ = "case_causes"
    __table_args__ = ({"schema": DIAGNOSIS_SCHEMA},)

    case_id: Mapped[str] = mapped_column(
        ForeignKey(f"{DIAGNOSIS_SCHEMA}.cases.id"), primary_key=True
    )
    cause_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    in_conformal_set: Mapped[bool | None] = mapped_column()


class Report8DRow(Base):
    __tablename__ = "reports_8d"
    __table_args__ = ({"schema": DIAGNOSIS_SCHEMA},)

    case_id: Mapped[str] = mapped_column(
        ForeignKey(f"{DIAGNOSIS_SCHEMA}.cases.id"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    language: Mapped[str] = mapped_column(String(8), primary_key=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    completeness: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SqlCaseRepository:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def add(self, case: DiagnosisCase) -> None:
        with session_scope(self._factory) as s:
            if s.get(CaseRow, case.case_id) is None:
                s.add(
                    CaseRow(
                        id=case.case_id,
                        source=case.source,
                        title=case.title,
                        description=case.description,
                        symptoms=list(case.reported_symptoms),
                        status=CaseStatus.REGISTERED.value,
                        case_json=case.model_dump(mode="json"),
                    )
                )

    def get(self, case_id: str) -> DiagnosisCase | None:
        with session_scope(self._factory) as s:
            row = s.get(CaseRow, case_id)
            return None if row is None else DiagnosisCase.model_validate(row.case_json)

    def status(self, case_id: str) -> CaseStatus | None:
        with session_scope(self._factory) as s:
            row = s.get(CaseRow, case_id)
            return None if row is None else CaseStatus(row.status)

    def save_ranking(self, case_id: str, top3: list[tuple[str, float]], novel: bool) -> None:
        with session_scope(self._factory) as s:
            row = s.get(CaseRow, case_id, with_for_update=True)
            if row is None:
                raise KeyError(case_id)
            s.execute(delete(CaseCauseRow).where(CaseCauseRow.case_id == case_id))
            for rank, (code, score) in enumerate(top3, start=1):
                s.add(CaseCauseRow(case_id=case_id, cause_code=code, score=score, rank=rank))
            row.novelty_flag = novel
            row.status = CaseStatus.AWAITING_CONFIRMATION.value

    def save_report(self, report: EightDReport, completeness: float) -> None:
        with session_scope(self._factory) as s:
            s.merge(
                Report8DRow(
                    case_id=report.case_id,
                    version=report.version,
                    language=report.language,
                    content=report.model_dump(mode="json"),
                    completeness=completeness,
                )
            )

    def save_confirmation(self, case_id: str, cause_code: str | None) -> None:
        with session_scope(self._factory) as s:
            row = s.get(CaseRow, case_id, with_for_update=True)
            if row is None:
                raise KeyError(case_id)
            row.confirmed_cause_code = cause_code
            row.status = CaseStatus.CONFIRMED.value
