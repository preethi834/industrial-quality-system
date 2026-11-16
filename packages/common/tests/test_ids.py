from __future__ import annotations

import uuid
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from iqs_common import ids
from iqs_common.ids import deterministic_id, uuid7, uuid7_timestamp_ms


def test_version_and_variant() -> None:
    value = uuid7()
    assert value.version == 7
    assert value.variant == uuid.RFC_4122


def test_embedded_timestamp_matches_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ids.time, "time_ns", lambda: 1_700_000_000_123_456_789)
    assert uuid7_timestamp_ms(uuid7()) == 1_700_000_000_123


def test_ids_sort_by_creation_time_across_milliseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Iterator[int] = iter(range(1_700_000_000_000, 1_700_000_000_050))
    monkeypatch.setattr(ids.time, "time_ns", lambda: next(clock) * 1_000_000)
    generated = [uuid7() for _ in range(50)]
    assert generated == sorted(generated)


def test_ten_thousand_ids_are_unique() -> None:
    assert len({uuid7() for _ in range(10_000)}) == 10_000


def test_timestamp_rejects_non_v7() -> None:
    with pytest.raises(ValueError, match="not a UUIDv7"):
        uuid7_timestamp_ms(uuid.uuid4())


@hypothesis_settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.integers(min_value=0, max_value=2**48 - 1))
def test_timestamp_round_trip_for_any_millisecond(unix_ms: int) -> None:
    with patch.object(ids.time, "time_ns", return_value=unix_ms * 1_000_000):
        value = uuid7()
    assert uuid7_timestamp_ms(value) == unix_ms
    assert value.version == 7


def test_deterministic_id_is_repeatable_and_order_sensitive() -> None:
    first = deterministic_id("sim", "seed-42", "part-7")
    assert first == deterministic_id("sim", "seed-42", "part-7")
    assert deterministic_id("a", "b") != deterministic_id("b", "a")
    assert deterministic_id("x").version == 5


def test_deterministic_id_separator_prevents_join_collisions() -> None:
    assert deterministic_id("a", "bc") != deterministic_id("ab", "c")


def test_deterministic_id_requires_parts() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        deterministic_id()


def test_deterministic_id_rejects_parts_containing_separator() -> None:
    with pytest.raises(ValueError, match="unit separator"):
        deterministic_id("a\x1fb")
    with pytest.raises(ValueError, match="unit separator"):
        deterministic_id("ok", "bad\x1f")
