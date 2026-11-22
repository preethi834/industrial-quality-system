"""Step 7 feedback: counts, candidate pool, graph, case_confirmed event, idempotency."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from iqs_common.events import CaseConfirmedEvent, EventType, parse_event
from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.feedback.store import InMemoryFeedbackStore
from iqs_diagnosis.feedback.update import FeedbackService
from iqs_diagnosis.graph.repository import ConfirmationWrite, InMemoryGraph
from iqs_diagnosis.ingestion.vision import case_from_cluster_event, vision_case_id

from .conftest import TAXONOMY_PATH
from .vision_fixtures import cluster_event, copy_run

TAX = load_taxonomy(TAXONOMY_PATH)
CAUSE = "MACHINE.TOOL_WEAR"


@pytest.fixture
def vision_case(reference_runs: dict[str, Path], tmp_path: Path) -> DiagnosisCase:
    runs = copy_run(reference_runs, tmp_path / "runs")
    return case_from_cluster_event(cluster_event().payload, runs, 12.0)


def _service(graph: Any = None) -> tuple[FeedbackService, InMemoryFeedbackStore, InMemoryGraph]:
    store = InMemoryFeedbackStore()
    g = graph if graph is not None else InMemoryGraph()
    return FeedbackService(store=store, taxonomy=TAX, graph=g), store, g


def _events(store: InMemoryFeedbackStore) -> list[CaseConfirmedEvent]:
    out = []
    for r in store.outbox.rows(EventType.CASE_CONFIRMED):
        e = parse_event(r.payload)
        assert isinstance(e, CaseConfirmedEvent)
        out.append(e)
    return out


def test_vision_confirmation_feeds_everything_once(vision_case: DiagnosisCase) -> None:
    svc, store, graph = _service()
    r = svc.apply(vision_case, CAUSE, "engineer-1", "scratch_tool_wear")
    ctx = "defect_type:toy/scratch_tool_wear"
    assert r["first_application"] and r["count_context"] == ctx
    assert (r["count_before"], r["count_after"]) == (0, 1)
    assert store.count(ctx, CAUSE) == 1
    assert [c["case_id"] for c in store.candidates()] == [vision_case.case_id]
    assert store.candidates()[0]["used_in_calibration_run"] is None
    assert r["graph"]["status"] == "written"
    assert graph.edge("Case", vision_case.case_id, "CONFIRMED_CAUSE", "Cause", CAUSE)
    assert graph.edge("DefectType", "toy/scratch_tool_wear", "INDICATES", "Cause", CAUSE) == {
        "count": 1
    }
    [ev] = _events(store)
    assert ev.payload.case_id == UUID(vision_case.case_id)
    assert str(ev.payload.source_cluster_id) == vision_case.source_cluster_id
    assert ev.payload.defect_class_name == "scratch_tool_wear"
    assert ev.payload.taxonomy_version == TAX.version

    # Second application (retry, replay): nothing changes.
    again = svc.apply(vision_case, CAUSE, "engineer-1", "scratch_tool_wear")
    assert again["first_application"] is False and again["count_after"] == 1
    assert store.count(ctx, CAUSE) == 1 and len(_events(store)) == 1
    assert len(store.candidates()) == 1


def test_default_class_name_matches_vision(vision_case: DiagnosisCase) -> None:
    svc, store, _ = _service()
    r = svc.apply(vision_case, CAUSE, "e")
    assert r["defect_class_name"] == "toy:tool_wear"
    assert _events(store)[0].payload.defect_class_name == "toy:tool_wear"


def test_second_case_same_defect_type_counts_up(
    reference_runs: dict[str, Path], tmp_path: Path
) -> None:
    runs = copy_run(reference_runs, tmp_path / "runs")
    svc, store, graph = _service()
    for _ in range(2):
        case = case_from_cluster_event(cluster_event().payload, runs, 12.0)
        svc.apply(case, CAUSE, "e", "scratch")
    assert store.count("defect_type:toy/scratch", CAUSE) == 2
    assert graph.edge("DefectType", "toy/scratch", "INDICATES", "Cause", CAUSE) == {"count": 2}


def test_non_vision_case_counts_by_source_without_event() -> None:
    case = DiagnosisCase(case_id="sim-1", source="simulator", title="t", description="d")
    svc, store, graph = _service()
    r = svc.apply(case, CAUSE, "e")
    assert r["count_context"] == "source:simulator" and r["count_after"] == 1
    assert _events(store) == []
    assert graph.edge("Case", "sim-1", "CONFIRMED_CAUSE", "Cause", CAUSE) is not None


def test_no_listed_cause_records_nothing(vision_case: DiagnosisCase) -> None:
    svc, store, graph = _service()
    r = svc.apply(vision_case, None, "e")
    assert r["graph"]["status"] == "skipped" and r["count_after"] is None
    assert store.counts() == [] and store.candidates() == [] and _events(store) == []
    assert graph.writes == 0


class FlakyGraph(InMemoryGraph):
    def __init__(self) -> None:
        super().__init__()
        self.down = True

    def write_confirmation(self, w: ConfirmationWrite) -> None:
        if self.down:
            raise ConnectionError("neo4j down")
        super().write_confirmation(w)


def test_graph_outage_keeps_postgres_feedback_and_retries(vision_case: DiagnosisCase) -> None:
    graph = FlakyGraph()
    svc, store, _ = _service(graph)
    r = svc.apply(vision_case, CAUSE, "e", "scratch")
    assert r["graph"] == {"status": "pending", "error": "ConnectionError"}
    assert store.count("defect_type:toy/scratch", CAUSE) == 1 and len(_events(store)) == 1
    assert svc.flush_graph() == 0
    graph.down = False
    assert svc.flush_graph() == 1
    assert graph.edge("Case", vision_case.case_id, "CONFIRMED_CAUSE", "Cause", CAUSE)
    assert svc.flush_graph() == 0  # nothing pending any more


def test_graph_disabled_is_reported_not_faked(vision_case: DiagnosisCase) -> None:
    svc = FeedbackService(store=InMemoryFeedbackStore(), taxonomy=TAX, graph=None)
    assert svc.apply(vision_case, CAUSE, "e")["graph"]["status"] == "disabled"


def test_agent_confirm_runs_feedback_node(
    reference_profiles: dict[str, Any], vision_case: DiagnosisCase
) -> None:
    from .agent_fixtures import make_agent

    svc, store, graph = _service()
    agent, _, _ = make_agent(reference_profiles, feedback=svc)
    asyncio.run(agent.run(vision_case))
    assert store.counts() == []  # nothing is learned before the engineer confirms
    state = asyncio.run(agent.confirm(vision_case.case_id, CAUSE, "engineer-1", "scratch"))
    assert agent.phase(vision_case.case_id) == "done"
    assert state["feedback"]["count_after"] == 1
    assert state["confirmation"]["defect_class_name"] == "scratch"
    assert "feedback_update" in state["timings_ms"]
    assert store.count("defect_type:toy/scratch", CAUSE) == 1
    assert graph.edge("Case", vision_case.case_id, "CONFIRMED_CAUSE", "Cause", CAUSE)


def test_agent_without_feedback_is_unchanged(
    reference_profiles: dict[str, Any], vision_case: DiagnosisCase
) -> None:
    from .agent_fixtures import make_agent

    agent, _, _ = make_agent(reference_profiles)
    asyncio.run(agent.run(vision_case))
    state = asyncio.run(agent.confirm(vision_case.case_id, CAUSE, "e"))
    assert state["feedback"] is None and agent.phase(vision_case.case_id) == "done"


def test_confirm_api_with_feedback(
    reference_profiles: dict[str, Any], reference_runs: dict[str, Path], tmp_path: Path
) -> None:
    import shutil

    from fastapi.testclient import TestClient

    from iqs_diagnosis.api.cases import CaseService
    from iqs_diagnosis.api.main import create_app
    from iqs_diagnosis.settings import DiagnosisSettings
    from iqs_diagnosis.worker.consumer import VisionCaseConsumer

    from .agent_fixtures import make_agent

    runs = copy_run(reference_runs, tmp_path / "runs")
    shutil.copytree(reference_runs["tool_wear_m1"], runs / "run-00000000000c")  # simulator case
    svc, store, _ = _service()
    agent, repo, _ = make_agent(reference_profiles, feedback=svc)
    ev = cluster_event()
    VisionCaseConsumer(agent, repo, runs, 12.0)(ev)
    service = CaseService(
        agent=agent,
        repository=repo,
        runs_root=runs,
        cause_codes=frozenset(c.code for c in TAX.causes),
        feedback=svc,
    )
    settings = DiagnosisSettings(
        api={"keys": "k", "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        neo4j={"uri": "bolt://127.0.0.1:1", "password": "x"},
    )
    h = {"X-API-Key": "k"}
    case_id = vision_case_id(ev.payload.cluster_id)
    with TestClient(create_app(settings, case_service=service)) as c:
        caps = c.get("/v1/capabilities", headers=h).json()["features"]
        assert caps["feedback"] and caps["graph"]
        bad = c.post(
            f"/v1/cases/{case_id}/confirm",
            json={"cause_code": CAUSE, "engineer": "e", "defect_class_name": "has space"},
            headers=h,
        )
        assert bad.status_code == 422
        r = c.post(
            f"/v1/cases/{case_id}/confirm",
            json={"cause_code": CAUSE, "engineer": "e", "defect_class_name": "scratch"},
            headers=h,
        )
        assert r.status_code == 200, r.text
        fb = r.json()["feedback"]
        assert fb["count_after"] == 1 and fb["graph"]["status"] == "written"
        counts = c.get("/v1/feedback/counts", headers=h).json()["counts"]
        assert counts == [{"context": "defect_type:toy/scratch", "cause_code": CAUSE, "count": 1}]
        assert len(c.get("/v1/feedback/candidates", headers=h).json()["candidates"]) == 1

        # A simulator case cannot be given a defect class name.
        sim = c.post("/v1/cases", json={"run_id": "run-00000000000c"}, headers=h)
        assert sim.status_code == 201
        sid = sim.json()["case_id"]
        assert c.post(f"/v1/cases/{sid}/run", headers=h).status_code == 200
        r2 = c.post(
            f"/v1/cases/{sid}/confirm",
            json={"cause_code": CAUSE, "engineer": "e", "defect_class_name": "x"},
            headers=h,
        )
        assert r2.status_code == 422
