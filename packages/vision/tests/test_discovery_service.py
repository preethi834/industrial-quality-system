"""Discovery service + case_confirmed consumer on the in-memory stores (step 7 core loop,
vision half): inspect -> unknown buffer -> cluster -> exactly one outbox event -> named."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from iqs_common.events import (
    CaseConfirmedEvent,
    CaseConfirmedPayload,
    EventType,
    NewDefectClusterEvent,
    parse_event,
)
from iqs_common.outbox_worker import InMemoryOutbox
from iqs_vision.discovery.consumer import make_case_confirmed_handler
from iqs_vision.discovery.service import DiscoveryService, PartContext
from iqs_vision.discovery.store import (
    ClusterStatus,
    InMemoryDiscoveryStore,
    Routing,
    UnknownClusterError,
)
from iqs_vision.discovery.trigger import TriggerParams

from .discovery_fakes import IMG, FakeScorer, ts, unit

RUN = "run-0123456789ab"


@pytest.fixture
def setup() -> tuple[DiscoveryService, FakeScorer, InMemoryOutbox, InMemoryDiscoveryStore]:
    box = InMemoryOutbox()
    store = InMemoryDiscoveryStore(box)
    scorer = FakeScorer()
    return DiscoveryService(scorer, store, TriggerParams()), scorer, box, store


def _ctx(i: int) -> PartContext:
    return PartContext(part_id=f"P{i:06d}", run_id=RUN, line_id="L1", observed_at=ts(i))


def _cluster_events(box: InMemoryOutbox) -> list[NewDefectClusterEvent]:
    out = []
    for r in box.rows(EventType.NEW_DEFECT_CLUSTER):
        e = parse_event(r.payload)
        assert isinstance(e, NewDefectClusterEvent)
        out.append(e)
    return out


def _confirm(cluster_id, name: str | None = "scratch_new") -> CaseConfirmedEvent:  # type: ignore[no-untyped-def]
    return CaseConfirmedEvent(
        payload=CaseConfirmedPayload(
            case_id=uuid4(),
            cause_code="MACHINE.TOOL_WEAR",
            taxonomy_version="0.1.0",
            confirmed_by="engineer-1",
            confirmed_at=datetime.now(UTC),
            source_cluster_id=cluster_id,
            defect_class_name=name,
        )
    )


def test_normal_and_undecided_images_are_not_buffered(setup) -> None:  # type: ignore[no-untyped-def]
    svc, scorer, box, store = setup
    scorer.queue = [(0.5, None)]
    assert svc.route_observation(IMG, "toy").observation.routing == Routing.NORMAL
    scorer.threshold = None
    scorer.queue = [(9.0, None)]
    assert svc.route_observation(IMG, "toy").observation.routing == Routing.UNDECIDED
    assert store.buffer("toy") == [] and box.rows() == []


def test_core_loop_vision_half(setup) -> None:  # type: ignore[no-untyped-def]
    svc, scorer, box, store = setup
    rng = np.random.default_rng(0)
    scratch = unit(rng)
    # Good parts, two scattered unrelated anomalies, then the new defect type.
    scorer.queue = [(0.2, None)] * 20 + [(2.0, unit(rng)), (2.0, unit(rng))]
    scorer.queue += [(2.0 + 0.1 * k, unit(rng, scratch)) for k in range(5)]
    results = [svc.route_observation(IMG, "toy", _ctx(i)) for i in range(27)]

    alerts = [r for r in results if r.alert_event_id is not None]
    assert len(alerts) == 1 and alerts[0] is results[-1]  # the 5th scratch triggers it
    events = _cluster_events(box)
    assert len(events) == 1
    p = events[0].payload
    assert p.size == 5 and p.simulator_run_id == RUN and p.linkage == "synthetic_simulator"
    assert p.first_seen == ts(22) and p.last_seen == ts(26)
    assert p.part_ids == tuple(f"P{i:06d}" for i in range(22, 27))
    assert p.representative_inspection_id in p.inspection_ids
    assert p.score_max == pytest.approx(2.4) and p.line_id == "L1"
    cluster = store.get_cluster(p.cluster_id)
    assert cluster is not None and cluster.status == ClusterStatus.ALERTED
    assert cluster.alerted_event_id == events[0].event_id

    # More of the same before confirmation: the cluster grows, no second event.
    scorer.queue = [(2.0, unit(rng, scratch)) for _ in range(4)]
    for i in range(27, 31):
        r = svc.route_observation(IMG, "toy", _ctx(i))
        assert r.alert_event_id is None and r.cluster_id == p.cluster_id
    assert len(_cluster_events(box)) == 1
    grown = store.get_cluster(p.cluster_id)
    assert grown is not None and grown.member_count == 9 and grown.last_seen == ts(30)

    # Engineer confirmation arrives (twice: at-least-once delivery).
    handler = make_case_confirmed_handler(store)
    confirm = _confirm(p.cluster_id)
    handler(confirm)
    handler(confirm)
    named = store.get_cluster(p.cluster_id)
    assert named is not None and named.status == ClusterStatus.NAMED
    assert named.defect_class_name == "scratch_new"
    assert named.cause_code == "MACHINE.TOOL_WEAR"

    # Afterwards the same defect is a known class: no buffer, no event.
    scorer.queue = [(2.0, unit(rng, scratch)) for _ in range(6)]
    for i in range(31, 37):
        r = svc.route_observation(IMG, "toy", _ctx(i))
        assert r.observation.routing == Routing.KNOWN and r.known_name == "scratch_new"
    assert len(_cluster_events(box)) == 1
    assert all(o.cluster_id != p.cluster_id for o in store.buffer("toy"))


def test_categories_are_clustered_separately(setup) -> None:  # type: ignore[no-untyped-def]
    svc, scorer, box, _ = setup
    rng = np.random.default_rng(1)
    c = unit(rng)
    scorer.queue = [(2.0, unit(rng, c)) for _ in range(8)]
    for i in range(8):
        svc.route_observation(IMG, "a" if i % 2 else "b", _ctx(i))
    assert box.rows() == []  # 4 per category < min_cluster_size


def test_consumer_ignores_non_vision_and_rejects_unknown_cluster(setup) -> None:  # type: ignore[no-untyped-def]
    _, _, _, store = setup
    handler = make_case_confirmed_handler(store)
    handler(_confirm(None, name=None))  # not a vision case: acknowledged, no change
    with pytest.raises(UnknownClusterError):
        handler(_confirm(uuid4()))  # retried by the worker, then parked as failed


def test_default_class_name(setup) -> None:  # type: ignore[no-untyped-def]
    svc, scorer, box, store = setup
    rng = np.random.default_rng(2)
    c = unit(rng)
    scorer.queue = [(2.0, unit(rng, c)) for _ in range(5)]
    for i in range(5):
        svc.route_observation(IMG, "toy", _ctx(i))
    cid = _cluster_events(box)[0].payload.cluster_id
    make_case_confirmed_handler(store)(_confirm(cid, name=None))
    named = store.get_cluster(cid)
    assert named is not None and named.defect_class_name == "toy:tool_wear"
