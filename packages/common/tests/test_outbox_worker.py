"""Outbox delivery semantics on the in-memory store (SQL store: integration tests, 7g)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from iqs_common.events import (
    CaseConfirmedEvent,
    CaseConfirmedPayload,
    Event,
    EventType,
    NewDefectClusterEvent,
    NewDefectClusterPayload,
)
from iqs_common.outbox import OutboxStatus
from iqs_common.outbox_worker import InMemoryOutbox, run_worker

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
CLUSTER = {EventType.NEW_DEFECT_CLUSTER}


def _cluster() -> NewDefectClusterEvent:
    return NewDefectClusterEvent(
        payload=NewDefectClusterPayload(
            cluster_id=uuid4(),
            category="bottle",
            size=2,
            first_seen=NOW,
            last_seen=NOW,
            inspection_ids=(uuid4(),),
            model_version="m",
        )
    )


def _confirmed() -> CaseConfirmedEvent:
    return CaseConfirmedEvent(
        payload=CaseConfirmedPayload(
            case_id=uuid4(),
            cause_code="MACHINE.TOOL_WEAR",
            taxonomy_version="0.1.0",
            confirmed_by="e",
            confirmed_at=NOW,
        )
    )


def test_delivers_only_requested_types_in_order() -> None:
    box = InMemoryOutbox()
    events: list[Event] = [_cluster(), _confirmed(), _cluster()]
    for e in events:
        box.append(e)
    seen: list[Event] = []
    result = box.process_pending(CLUSTER, seen.append)
    assert [e.event_id for e in seen] == [events[0].event_id, events[2].event_id]
    assert result.processed == [events[0].event_id, events[2].event_id]
    statuses = {r.id: r.status for r in box.rows()}
    assert statuses[events[1].event_id] == OutboxStatus.PENDING.value
    # Processed rows are not delivered again.
    assert box.process_pending(CLUSTER, seen.append).total == 0


def test_duplicate_append_rejected_like_primary_key() -> None:
    box = InMemoryOutbox()
    e = _cluster()
    box.append(e)
    with pytest.raises(ValueError, match="duplicate"):
        box.append(e)


def test_failing_handler_retries_then_fails() -> None:
    box = InMemoryOutbox()
    e = _cluster()
    box.append(e)

    def boom(_: Event) -> None:
        raise RuntimeError("down")

    for attempt in range(1, 3):
        r = box.process_pending(CLUSTER, boom, max_attempts=3)
        assert r.retried == [e.event_id]
        assert box.rows()[0].attempts == attempt
    r = box.process_pending(CLUSTER, boom, max_attempts=3)
    assert r.failed == [e.event_id]
    row = box.rows()[0]
    assert row.status == OutboxStatus.FAILED.value
    assert row.last_error == "RuntimeError: down"
    assert box.process_pending(CLUSTER, boom).total == 0  # failed rows are parked


def test_one_failure_does_not_block_the_batch() -> None:
    box = InMemoryOutbox()
    bad, good = _cluster(), _cluster()
    box.append(bad)
    box.append(good)

    def handler(e: Event) -> None:
        if e.event_id == bad.event_id:
            raise ValueError("x")

    r = box.process_pending(CLUSTER, handler)
    assert r.retried == [bad.event_id] and r.processed == [good.event_id]


def test_invalid_payload_is_recorded_not_raised() -> None:
    box = InMemoryOutbox()
    e = _cluster()
    box.append(e)
    box.rows()[0].payload["payload"] = {"nonsense": True}
    r = box.process_pending(CLUSTER, lambda _: None, max_attempts=1)
    assert r.failed == [e.event_id]
    assert (box.rows()[0].last_error or "").startswith("invalid event")


def test_requeue_redelivers_same_event() -> None:
    box = InMemoryOutbox()
    e = _cluster()
    box.append(e)
    seen: list[Event] = []
    box.process_pending(CLUSTER, seen.append)
    box.requeue(e.event_id)
    box.process_pending(CLUSTER, seen.append)
    assert [x.event_id for x in seen] == [e.event_id, e.event_id]  # at-least-once


def test_run_worker_stops() -> None:
    box = InMemoryOutbox()
    box.append(_cluster())
    assert run_worker(box, CLUSTER, lambda _: None, max_batches=2) == 1
    stop = threading.Event()
    stop.set()
    assert run_worker(box, CLUSTER, lambda _: None, stop=stop) == 0
