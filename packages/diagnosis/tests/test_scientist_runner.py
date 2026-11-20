"""Scientist runner: Benjamini-Hochberg, verdicts, data-supported cause; agent and API."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.scientist.runner import bh_adjust, test_case

from .agent_fixtures import make_agent
from .conftest import TAXONOMY_PATH

TEMPLATES = {
    c.code: c.test_template for c in load_taxonomy(TAXONOMY_PATH).causes if c.test_template
}


def test_bh_hand_computed() -> None:
    # sorted p: .01 .02 .04 (m = 3): adjusted = min over j >= i of p_j * m / j
    # -> .03, .03, .04 ; None stays None and is not counted in m.
    adj = bh_adjust([0.04, None, 0.01, 0.02], 0.05)
    assert adj[1] is None
    assert adj[0] == pytest.approx(0.04) and adj[2] == pytest.approx(0.03)
    assert adj[3] == pytest.approx(0.03)
    assert bh_adjust([None, None], 0.05) == [None, None]


def test_case_verdicts_on_tool_wear(reference_runs: dict[str, Path]) -> None:
    case = load_case(reference_runs["tool_wear_m1"])
    assert case is not None
    hyps = ["MACHINE.TOOL_WEAR", "MATERIAL.BAD_BATCH", "PERSONNEL.PART_HANDLING_ERROR"]
    rep = test_case(case, hyps, TEMPLATES)
    v = {h.cause_code: h for h in rep.hypotheses}
    assert v["MACHINE.TOOL_WEAR"].verdict == "supported"
    assert v["MACHINE.TOOL_WEAR"].result.target == "M1"
    assert v["MATERIAL.BAD_BATCH"].verdict in ("rejected", "inconclusive")
    assert v["PERSONNEL.PART_HANDLING_ERROR"].verdict == "inconclusive"  # no dimension rejects
    assert rep.data_supported_cause == "MACHINE.TOOL_WEAR"
    with pytest.raises(ValueError, match="duplicate"):
        test_case(case, ["MACHINE.TOOL_WEAR"] * 2, TEMPLATES)


def test_agent_runs_scientist_and_reports_it(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> None:
    agent, _, _ = make_agent(reference_profiles)
    case = load_case(reference_runs["wrong_parameters_m3"])
    assert case is not None
    out = asyncio.run(agent.run(case))
    sci = out["scientist"]
    assert [h["cause_code"] for h in sci["hypotheses"]] == [c for c, _ in out["top3"]]
    assert sci["data_supported_cause"] == "METHOD.WRONG_PROCESS_PARAMETERS"
    d4 = out["report_8d"]["sections"][4]
    assert "Data check (Benjamini-Hochberg" in d4["content"]
    assert d4["data"]["data_supported_cause"] == "METHOD.WRONG_PROCESS_PARAMETERS"
    assert "test_hypotheses" in out["timings_ms"]


def test_hypotheses_endpoint(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any], tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from iqs_diagnosis.api.cases import CaseService
    from iqs_diagnosis.api.main import create_app
    from iqs_diagnosis.settings import DiagnosisSettings

    run_id = "run-0000000000aa"
    shutil.copytree(reference_runs["gauge_drift_q1"], tmp_path / "runs" / run_id)
    agent, repo, _ = make_agent(reference_profiles)
    service = CaseService(
        agent=agent,
        repository=repo,
        runs_root=tmp_path / "runs",
        cause_codes=frozenset(TEMPLATES),
    )
    settings = DiagnosisSettings(
        api={"keys": "k", "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        neo4j={"uri": "bolt://127.0.0.1:1", "password": "x"},
    )
    h = {"X-API-Key": "k"}
    with TestClient(create_app(settings, case_service=service)) as c:
        assert c.get("/v1/capabilities", headers=h).json()["features"]["hypothesis_tests"]
        case_id = c.post("/v1/cases", json={"run_id": run_id}, headers=h).json()["case_id"]
        assert c.get(f"/v1/cases/{case_id}/hypotheses", headers=h).status_code == 409
        run = c.post(f"/v1/cases/{case_id}/run", headers=h).json()
        assert run["data_supported_cause"] == "MEASUREMENT.SENSOR_DRIFT"
        body = c.get(f"/v1/cases/{case_id}/hypotheses", headers=h).json()
    first = body["hypotheses"][0]
    assert {"sql", "params", "plot", "test", "p_value"} <= set(first["result"])
    assert first["verdict"] in ("supported", "rejected", "inconclusive")


def test_hypotheses_for_known_and_novel() -> None:
    from iqs_diagnosis.scientist.runner import hypotheses_for

    allc = ["A", "B", "C", "D", "E"]
    assert hypotheses_for(["C", "A", "E"], allc, novel=False) == (["C", "A", "E"], "top3")
    assert hypotheses_for(["C", "A", "E"], allc, novel=True) == (
        ["C", "A", "E", "B", "D"],
        "all_causes",
    )


def test_all_causes_mode_picks_smallest_adjusted_p(reference_runs: dict[str, Path]) -> None:
    case = load_case(reference_runs["tool_wear_m1"])
    assert case is not None
    allc = list(TEMPLATES)
    # Put tool wear last so a rank-based pick would choose something else if supported.
    hyps = [c for c in allc if c != "MACHINE.TOOL_WEAR"] + ["MACHINE.TOOL_WEAR"]
    rep = test_case(case, hyps, TEMPLATES, mode="all_causes")
    assert rep.mode == "all_causes" and len(rep.hypotheses) == len(allc)
    sup = [h for h in rep.hypotheses if h.verdict == "supported"]
    assert sup and rep.supported_causes == [h.cause_code for h in sup]
    best = min(sup, key=lambda h: (h.p_adjusted or 1.0, h.rank))
    assert rep.data_supported_cause == best.cause_code
    # BH over all conclusive hypotheses, not three.
    rank_pick = test_case(case, hyps, TEMPLATES).data_supported_cause
    assert rank_pick == sup[0].cause_code


def test_novel_case_tests_every_cause_and_persists(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> None:
    agent, repo, _ = make_agent(reference_profiles)
    case = load_case(reference_runs["drive_fault_m1"])  # left out of the library
    assert case is not None
    out = asyncio.run(agent.run(case))
    assert out["novel"] is True
    sci = out["scientist"]
    assert sci["mode"] == "all_causes"
    tested = [h["cause_code"] for h in sci["hypotheses"]]
    assert tested[:3] == [c for c, _ in out["top3"]]
    assert sorted(tested) == sorted(TEMPLATES)
    d4 = out["report_8d"]["sections"][4]
    assert "all causes tested" in d4["content"]
    assert d4["data"]["data_check_mode"] == "all_causes"
    rows = repo.hypothesis_tests[case.case_id]
    assert len(rows) == len(TEMPLATES) and {r["mode"] for r in rows} == {"all_causes"}
    assert {r["verdict"] for r in rows} <= {"supported", "rejected", "inconclusive"}


def test_known_case_persists_top3_rows(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> None:
    agent, repo, _ = make_agent(reference_profiles)
    case = load_case(reference_runs["wrong_parameters_m3"])
    assert case is not None
    out = asyncio.run(agent.run(case))
    assert out["scientist"]["mode"] == "top3"
    rows = repo.hypothesis_tests[case.case_id]
    assert [r["cause_code"] for r in rows] == [c for c, _ in out["top3"]]
    assert all(r["sql"] and r["test_name"] for r in rows)
