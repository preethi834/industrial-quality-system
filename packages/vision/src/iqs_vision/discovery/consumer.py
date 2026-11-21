"""Vision consumer of ``case_confirmed`` (step 7): the cluster becomes a named defect class.

Idempotent on ``event_id`` (``vision.processed_events``) and on the cluster status
(``alerted -> named`` happens once). Confirmations of cases that did not come from a vision
cluster (``source_cluster_id is None``) are not vision's business and are acknowledged.
"""

from __future__ import annotations

from iqs_common.events import CaseConfirmedEvent, Event, default_defect_class_name
from iqs_common.log import get_logger
from iqs_common.outbox_worker import Handler
from iqs_vision.discovery.store import DiscoveryStore

_log = get_logger("iqs_vision.discovery.consumer")


def make_case_confirmed_handler(store: DiscoveryStore) -> Handler:
    def handle(event: Event) -> None:
        if not isinstance(event, CaseConfirmedEvent):
            raise TypeError(f"unexpected event type {event.type}")
        p = event.payload
        if p.source_cluster_id is None:
            return
        cluster = store.get_cluster(p.source_cluster_id)
        name = p.defect_class_name or default_defect_class_name(
            cluster.category if cluster else "unknown", p.cause_code
        )
        changed = store.name_cluster(
            event.event_id, p.source_cluster_id, name, p.cause_code, p.case_id
        )
        _log.info(
            "cluster_named" if changed else "cluster_name_skipped",
            cluster_id=str(p.source_cluster_id),
            name=name,
            cause_code=p.cause_code,
            event_id=str(event.event_id),
        )

    return handle
