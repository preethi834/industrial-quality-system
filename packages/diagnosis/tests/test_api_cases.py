"""Case endpoints over an injected in-memory case service."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.api.cases import CaseService
from iqs_diagnosis.api.main import create_app
from iqs_diagnosis.settings import DiagnosisSettings

from .agent_fixtures import make_agent
from .conftest import TAXONOMY_PATH

KEY = "k"
H = {"X-API-Key": KEY}
RUN = "run-0123456789ab"


@pytest.fixture
def client(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any], tmp_path: Path
) -> TestClient:
    runs_root = tmp_path / "runs"
    shutil.copytree(reference_runs["tool_wear_m1"], runs_root / RUN)
    shutil.copytree(reference_runs["baseline"], runs_root / "run-00000000000b")
    agent, repo, _ = make_agent(reference_profiles)
    service = CaseService(
        agent=agent,
        repository=repo,
        runs_root=runs_root,
        cause_codes=frozenset(c.code for c in load_taxonomy(TAXONOMY_PATH).causes),
    )
    settings = DiagnosisSettings(
        api={"keys": KEY, "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        neo4j={"uri": "bolt://127.0.0.1:1", "password": "x"},
    )
    return TestClient(create_app(settings, case_service=service))


def test_full_case_flow(client: TestClient) -> None:
    with client as c:
        caps = c.get("/v1/capabilities", headers=H).json()["features"]
        assert caps["cases"] and caps["retrieval"] and caps["report_8d"]

        r = c.post("/v1/cases", json={"run_id": RUN}, headers=H)
        assert r.status_code == 201
        case_id = r.json()["case_id"]
        assert c.get(f"/v1/cases/{case_id}", headers=H).json()["phase"] == "not_started"
        assert c.get(f"/v1/cases/{case_id}/cause-set", headers=H).status_code == 409

        run = c.post(f"/v1/cases/{case_id}/run", headers=H)
        assert run.status_code == 200
        body = run.json()
        assert body["phase"] == "awaiting_confirmation"
        assert body["top3"][0]["cause_code"] == "MACHINE.TOOL_WEAR"
        assert body["ticket_id"].startswith("QMS-") and body["report_completeness"] == 1.0
        assert c.post(f"/v1/cases/{case_id}/run", headers=H).status_code == 409

        cs = c.get(f"/v1/cases/{case_id}/cause-set", headers=H).json()
        assert cs["kind"] == "top3" and len(cs["top3"]) == 3
        assert caps["conformal_cause_set"] is False
        rep = c.get(f"/v1/cases/{case_id}/report-8d", headers=H).json()
        assert [s["key"] for s in rep["sections"]][0] == "D0"
        assert c.get(f"/v1/cases/{case_id}/report-8d?lang=de", headers=H).status_code == 422

        bad = c.post(
            f"/v1/cases/{case_id}/confirm", json={"cause_code": "X.Y", "engineer": "e"}, headers=H
        )
        assert bad.status_code == 422
        ok = c.post(
            f"/v1/cases/{case_id}/confirm",
            json={"cause_code": "MACHINE.TOOL_WEAR", "engineer": "e"},
            headers=H,
        )
        assert ok.status_code == 200 and ok.json()["phase"] == "done"
        assert ok.json()["ticket"]["status"] == "cause_confirmed"
        again = c.post(
            f"/v1/cases/{case_id}/confirm", json={"cause_code": None, "engineer": "e"}, headers=H
        )
        assert again.status_code == 409


def test_register_errors(client: TestClient) -> None:
    with client as c:
        assert c.post("/v1/cases", json={"run_id": RUN}).status_code == 401
        for bad in ("../etc", "run-XYZ", "run-0123456789ab/../x"):
            assert c.post("/v1/cases", json={"run_id": bad}, headers=H).status_code == 422
        assert (
            c.post("/v1/cases", json={"run_id": "run-ffffffffffff"}, headers=H).status_code == 404
        )
        # The baseline run raised no alarm, so there is no case.
        assert (
            c.post("/v1/cases", json={"run_id": "run-00000000000b"}, headers=H).status_code == 404
        )
        assert c.get("/v1/cases/SIM-nope", headers=H).status_code == 404
        assert c.post("/v1/cases/SIM-nope/run", headers=H).status_code == 404


def test_cause_set_is_conformal_when_calibrated(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any], tmp_path: Path
) -> None:
    from iqs_diagnosis.conformal.calibration import ConformalPredictor

    from .agent_fixtures import test_calibration_run

    runs_root = tmp_path / "runs"
    shutil.copytree(reference_runs["tool_wear_m1"], runs_root / RUN)
    agent, repo, _ = make_agent(
        reference_profiles, conformal=ConformalPredictor(test_calibration_run(0.75), "lac")
    )
    service = CaseService(
        agent=agent,
        repository=repo,
        runs_root=runs_root,
        cause_codes=frozenset(c.code for c in load_taxonomy(TAXONOMY_PATH).causes),
    )
    settings = DiagnosisSettings(
        api={"keys": KEY, "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        neo4j={"uri": "bolt://127.0.0.1:1", "password": "x"},
    )
    with TestClient(create_app(settings, case_service=service)) as c:
        assert c.get("/v1/capabilities", headers=H).json()["features"]["conformal_cause_set"]
        case_id = c.post("/v1/cases", json={"run_id": RUN}, headers=H).json()["case_id"]
        c.post(f"/v1/cases/{case_id}/run", headers=H)
        cs = c.get(f"/v1/cases/{case_id}/cause-set", headers=H).json()
    assert cs["kind"] == "conformal" and cs["causes"][0] == "MACHINE.TOOL_WEAR"
    assert cs["alpha"] == 0.1 and "marginally" in cs["guarantee"]
