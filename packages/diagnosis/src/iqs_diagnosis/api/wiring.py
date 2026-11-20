"""Build the case service (agent + stores) from settings."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.agent.checkpoint import memory_checkpointer, postgres_checkpointer
from iqs_diagnosis.agent.graph import AgentDeps, DiagnosisAgent
from iqs_diagnosis.agent.tickets import McpTicketGateway
from iqs_diagnosis.api.cases import CaseService
from iqs_diagnosis.conformal.calibration import CalibrationRun, ConformalPredictor
from iqs_diagnosis.persistence.repository import (
    CaseRepository,
    InMemoryCaseRepository,
    SqlCaseRepository,
)
from iqs_diagnosis.retrieval.library import load_library
from iqs_diagnosis.scoring.ranker import CauseRanker, RankerParams
from iqs_diagnosis.settings import DiagnosisSettings
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary


def build_case_service(settings: DiagnosisSettings) -> CaseService | None:
    """``None`` when no case library is configured (case endpoints stay off)."""
    d = settings.diagnosis
    if d.library_path is None:
        return None
    taxonomy = load_taxonomy(settings.taxonomy_path)
    causes = [c.code for c in taxonomy.causes]
    calibration = CalibrationRun.load(d.calibration_run_path) if d.calibration_run_path else None
    params = calibration.scorer if calibration else RankerParams.load(d.ranker_params_path)
    ranker = CauseRanker(load_library(d.library_path), causes, params)
    # The QMS MCP server runs in-process here; a separate MCP service (stdio/HTTP)
    # is wired in step 7 when the compose stack gains it.
    from iqs_mcp.server import build_server
    from iqs_mcp.store import InMemoryTicketStore, SqlTicketStore, TicketStore

    repository: CaseRepository
    checkpointer: BaseCheckpointSaver[Any]
    tickets: TicketStore
    if d.store == "postgres":
        from iqs_common.db import create_db_engine, create_session_factory

        factory = create_session_factory(create_db_engine(settings.database))
        repository = SqlCaseRepository(factory)
        checkpointer = postgres_checkpointer(settings.database)
        tickets = SqlTicketStore(factory)
    else:
        repository = InMemoryCaseRepository()
        checkpointer = memory_checkpointer()
        tickets = InMemoryTicketStore()
    deps = AgentDeps(
        vocabulary=load_vocabulary(d.vocabulary_path),
        ranker=ranker,
        taxonomy=taxonomy,
        tickets=McpTicketGateway(build_server(tickets)),
        repository=repository,
        conformal=ConformalPredictor(calibration, d.cause_set_variant) if calibration else None,
    )
    if calibration is not None:
        repository.save_calibration_run(calibration.model_dump(mode="json"))
    return CaseService(
        agent=DiagnosisAgent(deps, checkpointer),
        repository=repository,
        runs_root=d.runs_root,
        cause_codes=frozenset(causes),
    )
