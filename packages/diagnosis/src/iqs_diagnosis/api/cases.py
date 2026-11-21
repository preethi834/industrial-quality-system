"""Case endpoints (mounted under /v1 behind the API key).

``POST /cases/{id}/run`` runs synchronously up to the engineer's confirmation: without an
LLM a run takes well under a second (measured in step 3d). When LLM nodes arrive (3e)
it becomes ``202 Accepted`` + job ID as in blueprint §2.3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from iqs_diagnosis.agent.graph import DiagnosisAgent
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.persistence.repository import CaseRepository

RUN_ID = re.compile(r"^run-[0-9a-f]{12}$")


@dataclass
class CaseService:
    agent: DiagnosisAgent
    repository: CaseRepository
    runs_root: Path | None
    cause_codes: frozenset[str]


class RegisterCase(BaseModel):
    run_id: str = Field(description="Simulator run directory name (run-<12 hex>) under runs_root")


class ConfirmCause(BaseModel):
    cause_code: str | None = Field(description="Taxonomy code, or null if no listed cause fits")
    engineer: str = Field(min_length=1, max_length=128)


def _ranking(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "top3": [{"cause_code": c, "probability": p} for c, p in state["top3"]],
        "novel": state["novel"],
        "max_similarity": state["max_similarity"],
    }


def build_cases_router(service: CaseService) -> APIRouter:
    router = APIRouter(prefix="/cases", tags=["cases"])

    def _require(case_id: str) -> None:
        if service.repository.get(case_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"case {case_id} not found")

    def _require_run(case_id: str) -> dict[str, Any]:
        _require(case_id)
        if service.agent.phase(case_id) == "not_started":
            raise HTTPException(status.HTTP_409_CONFLICT, "case has not been run yet")
        return service.agent.state(case_id)

    @router.post("", status_code=status.HTTP_201_CREATED)
    def register(body: RegisterCase) -> dict[str, Any]:
        if service.runs_root is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "runs_root not configured")
        if not RUN_ID.match(body.run_id):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid run_id")
        case = load_case(service.runs_root / body.run_id)
        if case is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no case for this run (no alarm)")
        service.repository.add(case)
        return {"case_id": case.case_id, "title": case.title, "status": "registered"}

    @router.get("/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        _require(case_id)
        case = service.repository.get(case_id)
        assert case is not None
        return {
            "case_id": case_id,
            "title": case.title,
            "description": case.description,
            "phase": service.agent.phase(case_id),
        }

    @router.post("/{case_id}/run")
    async def run(case_id: str) -> dict[str, Any]:
        _require(case_id)
        if service.agent.phase(case_id) != "not_started":
            raise HTTPException(status.HTTP_409_CONFLICT, "case has already been run")
        case = service.repository.get(case_id)
        assert case is not None
        state = await service.agent.run(case)
        return {
            "case_id": case_id,
            "phase": service.agent.phase(case_id),
            **_ranking(state),
            "ticket_id": (state.get("ticket") or {}).get("ticket_id"),
            "data_supported_cause": (state.get("scientist") or {}).get("data_supported_cause"),
            "report_completeness": state["completeness"]["score"],
        }

    @router.get("/{case_id}/cause-set")
    def cause_set(case_id: str) -> dict[str, Any]:
        state = _require_run(case_id)
        cs = state.get("cause_set")
        if cs is None:  # no calibration run configured
            return {"case_id": case_id, "kind": "top3", **_ranking(state)}
        return {"case_id": case_id, "kind": "conformal", **cs, **_ranking(state)}

    @router.get("/{case_id}/hypotheses")
    def hypotheses(case_id: str) -> dict[str, Any]:
        """Scientist results: per hypothesis the SQL, test, p-values, verdict and figure."""
        state = _require_run(case_id)
        sci = state.get("scientist")
        if sci is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no hypothesis tests for this case")
        out: dict[str, Any] = sci
        return out

    @router.get("/{case_id}/report-8d")
    def report_8d(case_id: str, lang: str = "en") -> dict[str, Any]:
        state = _require_run(case_id)
        if lang != "en":
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "only lang=en exists before step 3e"
            )
        report: dict[str, Any] = state["report_8d"]
        return report

    @router.post("/{case_id}/confirm")
    async def confirm(case_id: str, body: ConfirmCause) -> dict[str, Any]:
        _require(case_id)
        if service.agent.phase(case_id) != "awaiting_confirmation":
            raise HTTPException(status.HTTP_409_CONFLICT, "case is not waiting for confirmation")
        if body.cause_code is not None and body.cause_code not in service.cause_codes:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "unknown cause code")
        state = await service.agent.confirm(case_id, body.cause_code, body.engineer)
        return {
            "case_id": case_id,
            "phase": service.agent.phase(case_id),
            "confirmation": state["confirmation"],
            "ticket": state.get("ticket"),
        }

    return router
