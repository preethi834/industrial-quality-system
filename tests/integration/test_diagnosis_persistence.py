"""Step 3c persistence against a real Postgres (testcontainers; run with `make test-integration`).

- migration 0002 creates the diagnosis and qms tables and matches the models;
- SqlTicketStore and SqlCaseRepository round-trip, ticket creation is idempotent;
- with the Postgres checkpointer a case paused at the confirmation survives a "restart"
  (a new agent instance with a new connection resumes it).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import inspect

from iqs_common.db import create_db_engine, create_session_factory
from iqs_common.settings import DatabaseSettings
from iqs_common.taxonomy import load_taxonomy
from iqs_mcp.models import TicketStatus, TicketUpdate
from iqs_mcp.server import build_server
from iqs_mcp.store import SqlTicketStore

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = REPO_ROOT / "configs/taxonomy/causes.yaml"
VOCABULARY = REPO_ROOT / "configs/diagnosis/symptoms.yaml"
SCENARIOS = REPO_ROOT / "configs/simulator/scenarios"


def test_migration_creates_tables(migrated: Config, pg_settings: DatabaseSettings) -> None:
    engine = create_db_engine(pg_settings)
    insp = inspect(engine)
    assert {"cases", "case_causes", "reports_8d"} <= set(insp.get_table_names("diagnosis"))
    assert {"tickets", "ticket_events"} <= set(insp.get_table_names("qms"))
    engine.dispose()


def test_sql_ticket_store(migrated: Config, pg_settings: DatabaseSettings) -> None:
    factory = create_session_factory(create_db_engine(pg_settings))
    store = SqlTicketStore(factory)
    t = store.create(
        case_id="SIM-1",
        title="t",
        severity=2,
        summary="s",
        candidate_causes=("MACHINE.TOOL_WEAR",),
        novel=False,
    )
    again = store.create(
        case_id="SIM-1", title="t", severity=2, summary="s", candidate_causes=(), novel=False
    )
    assert again.ticket_id == t.ticket_id
    u = store.update(
        t.ticket_id,
        TicketUpdate(status=TicketStatus.CAUSE_CONFIRMED, confirmed_cause="MACHINE.TOOL_WEAR"),
    )
    assert u.status is TicketStatus.CAUSE_CONFIRMED and len(u.history) == 2
    assert store.get(t.ticket_id).candidate_causes == ("MACHINE.TOOL_WEAR",)


def test_case_survives_restart_with_postgres_checkpointer(
    migrated: Config, pg_settings: DatabaseSettings, tmp_path: Path
) -> None:
    from iqs_diagnosis.agent.checkpoint import postgres_checkpointer
    from iqs_diagnosis.agent.graph import AgentDeps, DiagnosisAgent
    from iqs_diagnosis.agent.tickets import McpTicketGateway
    from iqs_diagnosis.evaluation.labels import load_case_label
    from iqs_diagnosis.ingestion.simulator import load_case
    from iqs_diagnosis.persistence.repository import CaseStatus, SqlCaseRepository
    from iqs_diagnosis.retrieval.base import LibraryCase
    from iqs_diagnosis.scoring.ranker import CauseRanker, RankerParams
    from iqs_diagnosis.symptoms import extract_profile
    from iqs_diagnosis.symptoms.vocabulary import load_vocabulary
    from iqs_sim.pipeline import run_scenario
    from iqs_sim.scenarios import load_scenario

    taxonomy = load_taxonomy(TAXONOMY)
    vocab = load_vocabulary(VOCABULARY)
    library = []
    for name in ("tool_wear_m1", "coolant_loss_m3"):
        run = tmp_path / name
        run_scenario(load_scenario(SCENARIOS / f"{name}.yaml"), run, taxonomy)
        case = load_case(run)
        label = load_case_label(run)
        assert case is not None and label is not None
        library.append(LibraryCase(extract_profile(case, vocab), label.cause_code))
    params = RankerParams(
        n_neighbours=1,
        alpha=0.01,
        novelty_threshold=None,
        novelty_alpha=0.1,
        n_novelty_calibration=0,
    )
    factory = create_session_factory(create_db_engine(pg_settings))

    def agent() -> DiagnosisAgent:
        deps = AgentDeps(
            vocabulary=vocab,
            ranker=CauseRanker(library, [c.code for c in taxonomy.causes], params),
            taxonomy=taxonomy,
            tickets=McpTicketGateway(build_server(SqlTicketStore(factory))),
            repository=SqlCaseRepository(factory),
        )
        return DiagnosisAgent(deps, postgres_checkpointer(pg_settings))

    case = load_case(tmp_path / "tool_wear_m1")
    assert case is not None
    first = agent()
    asyncio.run(first.run(case))
    assert first.phase(case.case_id) == "awaiting_confirmation"

    second = agent()  # new connection, new graph: only Postgres holds the state
    assert second.phase(case.case_id) == "awaiting_confirmation"
    done = asyncio.run(second.confirm(case.case_id, "MACHINE.TOOL_WEAR", "eng"))
    assert done["ticket"]["status"] == "cause_confirmed"
    repo = SqlCaseRepository(factory)
    assert repo.status(case.case_id) is CaseStatus.CONFIRMED


def test_calibration_run_rows_and_cause_set(
    migrated: Config, pg_settings: DatabaseSettings
) -> None:
    from iqs_diagnosis.domain.case import DiagnosisCase
    from iqs_diagnosis.persistence.repository import SqlCaseRepository

    factory = create_session_factory(create_db_engine(pg_settings))
    repo = SqlCaseRepository(factory)
    run = {
        "run_id": "cal_t",
        "corpus": "c",
        "alpha": 0.1,
        "n_calibration": 20,
        "thresholds": {
            "lac": {"method": "lac", "qhat": 0.5, "k": 19, "delta": None},
            "aps": {"method": "aps", "qhat": 0.9, "k": 19, "delta": None},
        },
    }
    repo.save_calibration_run(run)
    repo.save_calibration_run(run)  # idempotent
    repo.add(DiagnosisCase(case_id="SIM-c", source="simulator", title="t", description="d"))
    repo.save_ranking("SIM-c", [("MACHINE.TOOL_WEAR", 0.9), ("MATERIAL.BAD_BATCH", 0.05)], False)
    repo.save_cause_set("SIM-c", ["MACHINE.TOOL_WEAR"], "cal_t:lac")
