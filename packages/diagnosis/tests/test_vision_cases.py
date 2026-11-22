"""Step 7, diagnosis half of the loop: cluster event -> exactly one case -> agent run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iqs_common.events import EventType
from iqs_common.outbox_worker import InMemoryOutbox
from iqs_diagnosis.conformal.calibration import ConformalPredictor
from iqs_diagnosis.ingestion.vision import (
    VISION_SOURCE,
    VisionCaseError,
    case_from_cluster_event,
    vision_case_id,
)
from iqs_diagnosis.persistence.repository import CaseStatus
from iqs_diagnosis.worker.consumer import VisionCaseConsumer

from .agent_fixtures import make_agent, test_calibration_run
from .vision_fixtures import FIRST_SEEN, RUN, cluster_event, copy_run


@pytest.fixture
def runs_root(reference_runs: dict[str, Path], tmp_path: Path) -> Path:
    return copy_run(reference_runs, tmp_path / "runs")


def test_case_from_event(runs_root: Path) -> None:
    ev = cluster_event()
    case = case_from_cluster_event(ev.payload, runs_root, 12.0)
    assert case.case_id == vision_case_id(ev.payload.cluster_id)
    assert case.case_id == case_from_cluster_event(ev.payload, runs_root, 12.0).case_id
    assert case.source == VISION_SOURCE and case.linkage == "synthetic_simulator"
    assert case.source_cluster_id == str(ev.payload.cluster_id)
    assert case.observed_dir == (runs_root / RUN / "observed").resolve()
    assert case.window_start == FIRST_SEEN and case.baseline_hours == 12.0
    assert "SYNTHETIC" in case.description
    assert case.reported_symptoms[0]["code"] == "new_visual_defect_cluster"


def test_case_from_event_errors(runs_root: Path, tmp_path: Path) -> None:
    ev = cluster_event()
    with pytest.raises(VisionCaseError, match="runs_root"):
        case_from_cluster_event(ev.payload, None, 12.0)
    with pytest.raises(VisionCaseError, match="no observed data"):
        case_from_cluster_event(ev.payload, tmp_path / "empty", 12.0)
    unlinked = ev.payload.model_copy(
        update={"simulator_run_id": None, "part_ids": (), "linkage": "none"}
    )
    with pytest.raises(VisionCaseError, match="no linked process data"):
        case_from_cluster_event(unlinked, runs_root, 12.0)


def _consumer(reference_profiles: dict[str, Any], runs_root: Path, **kw: Any):  # type: ignore[no-untyped-def]
    agent, repo, tickets = make_agent(reference_profiles, **kw)
    return VisionCaseConsumer(agent, repo, runs_root, 12.0), agent, repo, tickets


def test_event_opens_exactly_one_case_and_runs_once(
    reference_profiles: dict[str, Any], runs_root: Path
) -> None:
    consumer, agent, repo, tickets = _consumer(reference_profiles, runs_root)
    box = InMemoryOutbox()
    ev = cluster_event()
    box.append(ev)
    assert box.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer).processed == [ev.event_id]

    cases = repo.list_cases(source="vision")
    assert len(cases) == 1
    case, status = cases[0]
    assert status == CaseStatus.AWAITING_CONFIRMATION
    assert agent.phase(case.case_id) == "awaiting_confirmation"
    state = agent.state(case.case_id)
    assert state["top3"] and state["report_8d"]
    reports_before = len(repo.reports[case.case_id])
    ticket_id = state["ticket"]["ticket_id"]
    assert tickets.get(ticket_id).case_id == case.case_id  # QMS ticket via MCP

    # Redelivery of the same event (lost acknowledgement).
    box.requeue(ev.event_id)
    box.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer)
    # A different event for the same cluster (e.g. a producer retry with a new event ID).
    again = cluster_event(cluster_id=ev.payload.cluster_id)
    box.append(again)
    box.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer)

    assert len(repo.list_cases(source="vision")) == 1
    assert len(repo.reports[case.case_id]) == reports_before  # agent not re-run
    assert {r.status for r in box.rows()} == {"processed"}


def test_crash_between_open_and_run_is_resumed_once(
    reference_profiles: dict[str, Any], runs_root: Path
) -> None:
    consumer, agent, repo, _ = _consumer(reference_profiles, runs_root)
    ev = cluster_event()
    case = case_from_cluster_event(ev.payload, runs_root, 12.0)
    assert repo.open_vision_case(ev.event_id, case)  # ... and the process died here
    assert agent.phase(case.case_id) == "not_started"
    assert consumer.resume_unrun_cases() == 1
    assert agent.phase(case.case_id) == "awaiting_confirmation"
    assert consumer.resume_unrun_cases() == 0


def test_unusable_event_is_retried_then_failed(
    reference_profiles: dict[str, Any], tmp_path: Path
) -> None:
    consumer, _, repo, _ = _consumer(reference_profiles, tmp_path / "no-runs")
    box = InMemoryOutbox()
    ev = cluster_event()
    box.append(ev)
    for _ in range(2):
        box.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer, max_attempts=2)
    assert box.rows()[0].status == "failed"
    assert "no observed data" in (box.rows()[0].last_error or "")
    assert repo.list_cases() == []


def test_vision_case_cause_set_says_guarantee_does_not_apply(
    reference_profiles: dict[str, Any], runs_root: Path
) -> None:
    run = test_calibration_run()
    consumer, agent, _, _ = _consumer(
        reference_profiles, runs_root, conformal=ConformalPredictor(run, "lac")
    )
    ev = cluster_event()
    consumer(ev)
    cs = agent.state(vision_case_id(ev.payload.cluster_id))["cause_set"]
    assert cs["guarantee_applies"] is False
    assert "does not apply" in cs["coverage_note"]


def test_case_list_api(reference_profiles: dict[str, Any], runs_root: Path) -> None:
    from fastapi.testclient import TestClient

    from iqs_common.taxonomy import load_taxonomy
    from iqs_diagnosis.api.cases import CaseService
    from iqs_diagnosis.api.main import create_app
    from iqs_diagnosis.settings import DiagnosisSettings

    from .conftest import TAXONOMY_PATH

    consumer, agent, repo, _ = _consumer(reference_profiles, runs_root)
    ev = cluster_event()
    consumer(ev)
    service = CaseService(
        agent=agent,
        repository=repo,
        runs_root=runs_root,
        cause_codes=frozenset(c.code for c in load_taxonomy(TAXONOMY_PATH).causes),
    )
    settings = DiagnosisSettings(
        api={"keys": "k", "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        neo4j={"uri": "bolt://127.0.0.1:1", "password": "x"},
    )
    h = {"X-API-Key": "k"}
    with TestClient(create_app(settings, case_service=service)) as c:
        all_cases = c.get("/v1/cases", headers=h).json()
        assert len(all_cases) == 1
        assert c.get("/v1/cases", headers=h, params={"source": "simulator"}).json() == []
        cid = str(ev.payload.cluster_id)
        listed = c.get("/v1/cases", headers=h, params={"cluster_id": cid}).json()
        assert [x["case_id"] for x in listed] == [vision_case_id(cid)]
        one = c.get(f"/v1/cases/{listed[0]['case_id']}", headers=h).json()
        assert one["source"] == "vision" and one["linkage"] == "synthetic_simulator"
        assert one["status"] == "awaiting_confirmation" and one["source_cluster_id"] == cid
        waiting = c.get("/v1/cases", headers=h, params={"status": "awaiting_confirmation"})
        assert len(waiting.json()) == 1
        assert c.get("/v1/cases", headers=h, params={"status": "bogus"}).status_code == 422
