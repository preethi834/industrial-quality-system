"""Step 7 gate 2: features that are not enabled show "not installed / disabled" with the
reason, and the quality-loop page offers no action for them. Runs against the real service
apps in-process (no Docker)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from iqs_dashboard import loop_client
from iqs_dashboard.capabilities import status_table, unavailable_message
from iqs_diagnosis.api.main import create_app as create_diagnosis
from iqs_diagnosis.settings import DiagnosisSettings
from iqs_vision.api.main import create_app as create_vision
from iqs_vision.settings import VisionSettings

PAGE = Path(__file__).resolve().parents[1] / "packages/dashboard/src/iqs_dashboard/pages"
KEY = "dash-key"
REPO = Path(__file__).resolve().parents[1]


def _vision(discovery: bool) -> TestClient:
    return TestClient(
        create_vision(
            VisionSettings(
                _env_file=None,
                api={"keys": KEY},
                database={"host": "127.0.0.1", "port": 1},
                vision={
                    "models_dir": Path("no-models"),
                    "discovery": {
                        "enabled": discovery,
                        "store": "memory",
                        "config_path": REPO / "configs/vision/discovery.yaml",
                    },
                },
            )
        )
    )


def _diagnosis() -> TestClient:  # no case library -> cases / feedback not available
    return TestClient(
        create_diagnosis(
            DiagnosisSettings(
                _env_file=None,
                api={"keys": KEY},
                database={"host": "127.0.0.1", "port": 1, "password": "x"},
                neo4j={"uri": "bolt://127.0.0.1:1", "password": "x"},
            )
        )
    )


def _wire(monkeypatch: pytest.MonkeyPatch, vision: TestClient, diagnosis: TestClient) -> None:
    c = loop_client.QualityLoopClient(
        cast(httpx.Client, vision), cast(httpx.Client, diagnosis), KEY
    )
    monkeypatch.setattr(loop_client, "get_loop_client", lambda: c)


def _texts(at: AppTest) -> list[str]:
    return [m.value for m in at.info]


def test_page_shows_reasons_when_everything_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, _vision(discovery=False), _diagnosis())
    at = AppTest.from_file(str(PAGE / "5_Quality_Loop.py")).run(timeout=60)
    assert not at.exception
    infos = _texts(at)
    assert any("discovery_trigger: disabled" in t and "IQS_VISION__DISCOVERY" in t for t in infos)
    assert any("cases: missing artifact" in t and "LIBRARY_PATH" in t for t in infos)
    assert len(at.button) == 0 and len(at.selectbox) == 0  # no action offered


def test_page_stops_when_only_diagnosis_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, _vision(discovery=True), _diagnosis())
    at = AppTest.from_file(str(PAGE / "5_Quality_Loop.py")).run(timeout=60)
    assert not at.exception
    infos = _texts(at)
    assert not any("discovery_trigger" in t for t in infos)
    assert any("cases: missing artifact" in t for t in infos)
    assert len(at.button) == 0


def test_page_reports_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    dead = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.5)
    c = loop_client.QualityLoopClient(dead, dead, KEY)
    monkeypatch.setattr(loop_client, "get_loop_client", lambda: c)
    at = AppTest.from_file(str(PAGE / "5_Quality_Loop.py")).run(timeout=60)
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)


def test_helper_messages() -> None:
    caps = {
        "service": "vision",
        "features": {"a": True, "edge": False, "old": False},
        "status": {"a": "enabled", "edge": "not_installed"},
        "reason": {"edge": "planned: step 8e"},
    }
    assert unavailable_message(caps, "a") is None
    assert unavailable_message(caps, "edge") == "vision · edge: not installed — planned: step 8e"
    assert unavailable_message(caps, "old") == "vision · old: disabled"  # pre-step-7 service
    assert "not installed" in (unavailable_message(caps, "missing") or "")
    assert status_table(caps)["status"] == ["enabled", "not installed", "disabled"]


class StubLoopClient:
    """API-shaped answers for the happy path (shapes from the service tests)."""

    def __init__(self) -> None:
        self.confirmed: list[tuple[str, str | None, str, str | None]] = []
        on: dict[str, object] = {"status": {}, "reason": {}}
        self.vcaps: dict[str, object] = {
            "service": "vision",
            "features": {"discovery_trigger": True},
            **on,
        }
        self.dcaps: dict[str, object] = {
            "service": "diagnosis",
            "features": {"cases": True, "feedback": True},
            **on,
        }

    def vision_capabilities(self) -> dict[str, object]:
        return self.vcaps

    def diagnosis_capabilities(self) -> dict[str, object]:
        return self.dcaps

    def clusters(self) -> list[dict[str, object]]:
        return [
            {
                "cluster_id": "0193a1b2-0000-7000-8000-0000000000c1",
                "category": "toy",
                "status": "alerted",
                "size": 5,
                "first_seen": "2026-01-06T07:26:00+00:00",
                "defect_class_name": None,
                "cause_code": None,
            }
        ]

    def vision_cases(self) -> list[dict[str, object]]:
        return [
            {
                "case_id": "6b0c7c8e-3a59-5d43-9a1f-0c3c1b8e5f01",
                "source": "vision",
                "status": "awaiting_confirmation",
                "title": "New visual defect cluster: toy (5 images)",
                "source_cluster_id": "0193a1b2-0000-7000-8000-0000000000c1",
                "linkage": "synthetic_simulator",
                "phase": "awaiting_confirmation",
            }
        ]

    def cause_set(self, case_id: str) -> dict[str, object]:
        return {
            "kind": "conformal",
            "causes": ["MACHINE.TOOL_WEAR", "MACHINE.SPINDLE_OVERLOAD"],
            "guarantee_applies": False,
            "coverage_note": "the guarantee does not apply to it.",
            "top3": [
                {"cause_code": "MACHINE.TOOL_WEAR", "probability": 0.7},
                {"cause_code": "MACHINE.SPINDLE_OVERLOAD", "probability": 0.2},
                {"cause_code": "MACHINE.DRIVE_FAULT", "probability": 0.1},
            ],
            "novel": False,
        }

    def confirm(
        self, case_id: str, cause: str | None, engineer: str, name: str | None
    ) -> dict[str, object]:
        self.confirmed.append((case_id, cause, engineer, name))
        return {
            "feedback": {
                "count_before": 0,
                "count_after": 1,
                "count_context": "defect_type:toy/scratch",
                "graph": {"status": "written"},
            }
        }


def test_page_happy_path_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubLoopClient()
    monkeypatch.setattr(loop_client, "get_loop_client", lambda: stub)
    at = AppTest.from_file(str(PAGE / "5_Quality_Loop.py")).run(timeout=60)
    assert not at.exception
    assert any("does not apply" in w.value for w in at.warning)
    assert at.selectbox(key="cause").options[-1] == "none of these"
    at.text_input(key="engineer").input("engineer-1")
    at.text_input(key="class_name").input("scratch")
    at.button[0].click().run(timeout=60)
    assert not at.exception
    assert stub.confirmed == [
        ("6b0c7c8e-3a59-5d43-9a1f-0c3c1b8e5f01", "MACHINE.TOOL_WEAR", "engineer-1", "scratch")
    ]
    assert any("Count 0 → 1" in s.value for s in at.success)


def test_feedback_off_means_no_confirm_button(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubLoopClient()
    stub.dcaps = {
        "service": "diagnosis",
        "features": {"cases": True, "feedback": False},
        "status": {"cases": "enabled", "feedback": "disabled"},
        "reason": {"feedback": "feedback off"},
    }
    monkeypatch.setattr(loop_client, "get_loop_client", lambda: stub)
    at = AppTest.from_file(str(PAGE / "5_Quality_Loop.py")).run(timeout=60)
    assert not at.exception
    assert any("feedback: disabled" in m.value for m in at.info)
    assert len(at.button) == 0
