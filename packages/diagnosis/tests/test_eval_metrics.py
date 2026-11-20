"""Wilson intervals and latency summaries."""

from __future__ import annotations

import math

import pytest

from iqs_diagnosis.evaluation.metrics import latency, wilson


def test_wilson_known_values() -> None:
    p = wilson(5, 10)
    assert p.value == 0.5
    assert p.low == pytest.approx(0.2366, abs=1e-4) and p.high == pytest.approx(0.7634, abs=1e-4)
    zero = wilson(0, 10)
    assert zero.low == 0.0 and zero.high == pytest.approx(0.2775, abs=1e-4)
    full = wilson(74, 74)
    assert full.high == 1.0 and full.low == pytest.approx(0.9507, abs=1e-4)


def test_wilson_edge_cases() -> None:
    assert math.isnan(wilson(0, 0).value)
    with pytest.raises(ValueError):
        wilson(3, 2)
    assert "3/4" in str(wilson(3, 4))


def test_latency_nearest_rank() -> None:
    lat = latency([float(x) for x in range(1, 101)])
    assert (lat.p50_ms, lat.p95_ms, lat.max_ms, lat.n) == (50.0, 95.0, 100.0, 100)
    with pytest.raises(ValueError):
        latency([])
