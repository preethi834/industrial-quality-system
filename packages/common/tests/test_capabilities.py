from __future__ import annotations

from iqs_common.capabilities import WHEN_OFF, FeatureStatus, feature_status


def test_status_and_reason() -> None:
    st, why = feature_status(
        {"inspections": False, "discovery_trigger": True, "edge": False, "x": False}
    )
    assert st == {
        "inspections": "missing_artifact",
        "discovery_trigger": "enabled",
        "edge": "not_installed",
        "x": "disabled",
    }
    assert "discovery_trigger" not in why and "8e" in why["edge"]


def test_every_service_flag_has_an_entry() -> None:
    import pytest

    dx = pytest.importorskip("iqs_diagnosis.api.main")
    vx = pytest.importorskip("iqs_vision.api.main")
    DX, VX = dx.FEATURES, vx.FEATURES

    assert set(VX) | set(DX) <= set(WHEN_OFF)
    assert all(isinstance(s, FeatureStatus) and r for s, r in WHEN_OFF.values())
