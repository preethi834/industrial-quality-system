from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from iqs_common.events import (
    CaseConfirmedEvent,
    CaseConfirmedPayload,
    EventType,
    NewDefectClusterEvent,
    NewDefectClusterPayload,
    parse_event,
)
from iqs_common.ids import uuid7

T0 = datetime(2026, 11, 16, 10, 0, tzinfo=UTC)


def _cluster_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cluster_id": uuid7(),
        "category": "bottle",
        "size": 5,
        "first_seen": T0,
        "last_seen": T0 + timedelta(hours=1),
        "inspection_ids": [uuid7() for _ in range(3)],
        "model_version": "patchcore-v1",
    }
    data.update(overrides)
    return data


def _confirmed_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "case_id": uuid7(),
        "cause_code": "MACHINE.FIXTURE_MISALIGNMENT",
        "taxonomy_version": "v0",
        "confirmed_by": "engineer-1",
        "confirmed_at": T0,
        "source_cluster_id": uuid7(),
        "defect_class_name": "scratch_new",
    }
    data.update(overrides)
    return data


def _cluster_event() -> NewDefectClusterEvent:
    return NewDefectClusterEvent(payload=NewDefectClusterPayload(**_cluster_payload()))


def _confirmed_event() -> CaseConfirmedEvent:
    return CaseConfirmedEvent(payload=CaseConfirmedPayload(**_confirmed_payload()))


# --- round trips and dispatch ------------------------------------------------------


def test_cluster_event_json_round_trip() -> None:
    event = _cluster_event()
    parsed = parse_event(event.model_dump_json())
    assert isinstance(parsed, NewDefectClusterEvent)
    assert parsed == event
    assert parsed.type is EventType.NEW_DEFECT_CLUSTER


def test_confirmed_event_dict_and_bytes_round_trip() -> None:
    event = _confirmed_event()
    from_dict = parse_event(event.model_dump(mode="json"))
    from_bytes = parse_event(event.model_dump_json().encode("utf-8"))
    assert isinstance(from_dict, CaseConfirmedEvent)
    assert from_dict == event == from_bytes
    assert from_dict.type is EventType.CASE_CONFIRMED


def test_type_serialises_as_plain_string() -> None:
    assert _cluster_event().model_dump(mode="json")["type"] == "new_defect_cluster"
    assert _confirmed_event().model_dump(mode="json")["type"] == "case_confirmed"


def test_defaults_event_id_v7_and_aware_timestamp() -> None:
    event = _cluster_event()
    assert event.event_id.version == 7
    assert event.occurred_at.tzinfo is not None
    assert event.schema_version == 1
    assert event.producer == "vision"
    assert _confirmed_event().producer == "diagnosis"


# --- envelope validation -----------------------------------------------------------


def test_unknown_type_rejected() -> None:
    data = _cluster_event().model_dump(mode="json")
    data["type"] = "something_else"
    with pytest.raises(ValidationError):
        parse_event(data)


def test_wrong_producer_for_type_rejected() -> None:
    data = _cluster_event().model_dump(mode="json")
    data["producer"] = "diagnosis"
    with pytest.raises(ValidationError):
        parse_event(data)


def test_extra_fields_rejected_in_envelope_and_payload() -> None:
    envelope = _cluster_event().model_dump(mode="json")
    envelope["unexpected"] = 1
    with pytest.raises(ValidationError):
        parse_event(envelope)
    with pytest.raises(ValidationError):
        NewDefectClusterPayload(**_cluster_payload(unexpected=1))


def test_unsupported_schema_version_rejected() -> None:
    data = _cluster_event().model_dump(mode="json")
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        parse_event(data)


def test_events_are_immutable() -> None:
    event = _cluster_event()
    with pytest.raises(ValidationError):
        event.producer = "diagnosis"  # type: ignore[misc, assignment]


# --- payload validation ------------------------------------------------------------


def test_naive_datetimes_rejected() -> None:
    with pytest.raises(ValidationError):
        NewDefectClusterPayload(**_cluster_payload(first_seen=datetime(2026, 11, 16, 10, 0)))
    with pytest.raises(ValidationError):
        CaseConfirmedPayload(**_confirmed_payload(confirmed_at=datetime(2026, 11, 16, 10, 0)))


def test_last_seen_before_first_seen_rejected() -> None:
    with pytest.raises(ValidationError, match="last_seen"):
        NewDefectClusterPayload(**_cluster_payload(last_seen=T0 - timedelta(seconds=1)))


def test_more_inspection_ids_than_cluster_size_rejected() -> None:
    with pytest.raises(ValidationError, match="cluster size"):
        NewDefectClusterPayload(**_cluster_payload(size=2))


def test_inspection_ids_limits() -> None:
    with pytest.raises(ValidationError):
        NewDefectClusterPayload(**_cluster_payload(inspection_ids=[]))
    at_limit = [uuid7() for _ in range(1000)]
    NewDefectClusterPayload(**_cluster_payload(size=5000, inspection_ids=at_limit))
    with pytest.raises(ValidationError):
        NewDefectClusterPayload(**_cluster_payload(size=5000, inspection_ids=[*at_limit, uuid7()]))


@pytest.mark.parametrize("bad_code", ["tool wear", "machine.tool_wear", "", "MACHINE..X"])
def test_malformed_cause_code_rejected(bad_code: str) -> None:
    with pytest.raises(ValidationError):
        CaseConfirmedPayload(**_confirmed_payload(cause_code=bad_code))


def test_defect_class_name_requires_source_cluster() -> None:
    with pytest.raises(ValidationError, match="requires source_cluster_id"):
        CaseConfirmedPayload(**_confirmed_payload(source_cluster_id=None))
    manual_case = CaseConfirmedPayload(
        **_confirmed_payload(source_cluster_id=None, defect_class_name=None)
    )
    assert manual_case.source_cluster_id is None
