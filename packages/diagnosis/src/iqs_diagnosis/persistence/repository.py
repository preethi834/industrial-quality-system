"""Case repository: in memory (tests, evaluation) and Postgres (``diagnosis`` schema).

Tables follow blueprint §2.4 with two simplifications for step 3c: causes are stored by
taxonomy code (the ``cause_taxonomy`` table is not created yet), and
``in_conformal_set`` / ``calibration_run_id`` stay empty until step 4.
``hypothesis_tests`` (migration 0004) holds one row per scientist hypothesis; a new
scientist report for a case replaces that case's rows.
"""

from __future__ import annotations

import threading
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

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
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from iqs_common.db import Base, session_scope
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.reporting.eight_d import EightDReport

if TYPE_CHECKING:
    from iqs_diagnosis.scientist.runner import ScientistReport

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
    def save_calibration_run(self, run: dict[str, Any]) -> None: ...
    def save_cause_set(
        self, case_id: str, cause_set: list[str], calibration_run_id: str
    ) -> None: ...
    def save_hypothesis_tests(self, case_id: str, report: ScientistReport) -> None: ...


def hypothesis_rows(report: ScientistReport) -> list[dict[str, Any]]:
    """Flatten a scientist report into ``diagnosis.hypothesis_tests`` column values."""
    rows = []
    for h in report.hypotheses:
        r = h.result
        rows.append(
            {
                "cause_code": h.cause_code,
                "rank": h.rank,
                "mode": report.mode,
                "sql_template": r.template,
                "sql": r.sql,
                "params": list(r.params),
                "test_name": r.test,
                "target": r.target,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "p_adjusted": h.p_adjusted,
                "q": report.q,
                "effect_size": dict(r.effect),
                "verdict": h.verdict,
                "evidence": {
                    "groups": dict(r.groups),
                    "n_targets": r.n_targets,
                    "p_min_raw": r.p_min_raw,
                    "reason": r.reason,
                    "reference_end": report.reference_end,
                    "affected_start": report.affected_start,
                    "plot": r.plot,
                },
            }
        )
    return rows


class InMemoryCaseRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cases: dict[str, DiagnosisCase] = {}
        self.statuses: dict[str, CaseStatus] = {}
        self.rankings: dict[str, dict[str, Any]] = {}
        self.reports: dict[str, list[EightDReport]] = {}
        self.confirmations: dict[str, str | None] = {}
        self.calibration_runs: dict[str, dict[str, Any]] = {}
        self.cause_sets: dict[str, tuple[list[str], str]] = {}
        self.hypothesis_tests: dict[str, list[dict[str, Any]]] = {}

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

    def save_calibration_run(self, run: dict[str, Any]) -> None:
        with self._lock:
            self.calibration_runs.setdefault(run["run_id"], run)

    def save_cause_set(self, case_id: str, cause_set: list[str], calibration_run_id: str) -> None:
        with self._lock:
            self.cause_sets[case_id] = (list(cause_set), calibration_run_id)

    def save_hypothesis_tests(self, case_id: str, report: ScientistReport) -> None:
        with self._lock:
            self.hypothesis_tests[case_id] = hypothesis_rows(report)


# ------------------------------------------------------------------ Postgres

_STATUS_SQL = ", ".join(f"'{s.value}'" for s in CaseStatus)
VERDICTS = ("supported", "rejected", "inconclusive")
_VERDICT_SQL = ", ".join(f"'{v}'" for v in VERDICTS)


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
    calibration_run_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{DIAGNOSIS_SCHEMA}.calibration_runs.id")
    )


class CalibrationRunRow(Base):
    """Frozen calibration runs (blueprint §2.4); one row per threshold variant."""

    __tablename__ = "calibration_runs"
    __table_args__ = ({"schema": DIAGNOSIS_SCHEMA},)

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    method: Mapped[str] = mapped_column(String(16))
    alpha: Mapped[float] = mapped_column(Float)
    delta: Mapped[float | None] = mapped_column(Float)
    qhat: Mapped[float] = mapped_column(Float)
    n_calibration: Mapped[int] = mapped_column(Integer)
    dataset: Mapped[str] = mapped_column(String(64))
    content: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class HypothesisTestRow(Base):
    """One scientist hypothesis per row (blueprint §2.4 ``hypothesis_tests``)."""

    __tablename__ = "hypothesis_tests"
    __table_args__ = (
        CheckConstraint(f"verdict IN ({_VERDICT_SQL})", name="verdict"),
        CheckConstraint("mode IN ('top3', 'all_causes')", name="mode"),
        {"schema": DIAGNOSIS_SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey(f"{DIAGNOSIS_SCHEMA}.cases.id"), index=True)
    cause_code: Mapped[str] = mapped_column(String(64))
    rank: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(String(16))
    sql_template: Mapped[str] = mapped_column(String(64))
    sql: Mapped[str] = mapped_column(Text)
    params: Mapped[list[str]] = mapped_column(JSONB)
    test_name: Mapped[str] = mapped_column(String(64))
    target: Mapped[str | None] = mapped_column(String(64))
    statistic: Mapped[float | None] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float)
    p_adjusted: Mapped[float | None] = mapped_column(Float)
    q: Mapped[float] = mapped_column(Float)
    effect_size: Mapped[dict[str, float]] = mapped_column(JSONB)
    verdict: Mapped[str] = mapped_column(String(16))
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


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

    def save_calibration_run(self, run: dict[str, Any]) -> None:
        """One row per threshold variant (``<run_id>:<variant>``); existing rows are kept."""
        with session_scope(self._factory) as s:
            for variant, t in run["thresholds"].items():
                key = f"{run['run_id']}:{variant}"
                if s.get(CalibrationRunRow, key) is None:
                    s.add(
                        CalibrationRunRow(
                            id=key,
                            method=t["method"],
                            alpha=run["alpha"],
                            delta=t.get("delta"),
                            qhat=t["qhat"],
                            n_calibration=run["n_calibration"],
                            dataset=run["corpus"],
                            content=run,
                        )
                    )

    def save_cause_set(self, case_id: str, cause_set: list[str], calibration_run_id: str) -> None:
        with session_scope(self._factory) as s:
            rows = s.scalars(select(CaseCauseRow).where(CaseCauseRow.case_id == case_id)).all()
            present = {r.cause_code for r in rows}
            for r in rows:
                r.in_conformal_set = r.cause_code in cause_set
                r.calibration_run_id = calibration_run_id
            for rank, code in enumerate(cause_set, start=len(rows) + 1):
                if code not in present:
                    s.add(
                        CaseCauseRow(
                            case_id=case_id,
                            cause_code=code,
                            score=0.0,
                            rank=rank,
                            in_conformal_set=True,
                            calibration_run_id=calibration_run_id,
                        )
                    )

    def save_hypothesis_tests(self, case_id: str, report: ScientistReport) -> None:
        with session_scope(self._factory) as s:
            s.execute(delete(HypothesisTestRow).where(HypothesisTestRow.case_id == case_id))
            for row in hypothesis_rows(report):
                s.add(HypothesisTestRow(case_id=case_id, **row))
