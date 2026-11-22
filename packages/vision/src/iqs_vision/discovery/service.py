"""Discovery application service (step 7): score -> route -> buffer -> maybe alert.

``route_observation`` is the one entry point. ``POST /v1/discovery/inspect`` calls it today;
after step 2 merges, ``POST /v1/inspections`` can call it too (merge follow-up; the
inspection route is not edited on this branch).

Concurrency: one lock per service instance serialises the buffer -> cluster -> alert step,
so two requests cannot alert the same group twice in one process. Across processes the
unique ``alerted_event_id`` and the overlap rule keep a duplicate alert visible but not
silent; the API runs one worker process (compose), which is the supported setup.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from statistics import median
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from iqs_common.events import NewDefectClusterEvent, NewDefectClusterPayload
from iqs_common.ids import uuid7
from iqs_common.log import get_logger
from iqs_vision.data.preprocess import image_to_tensor
from iqs_vision.discovery.embed import region_embedding
from iqs_vision.discovery.store import (
    ClusterRecord,
    ClusterStatus,
    DiscoveryStore,
    Observation,
    Routing,
)
from iqs_vision.discovery.trigger import (
    BufferedPoint,
    ClusterView,
    TriggerParams,
    assign_known,
    find_clusters,
)
from iqs_vision.inference.registry import ModelRegistry

_log = get_logger("iqs_vision.discovery")
MAX_EVENT_IDS = 1000  # NewDefectClusterPayload.inspection_ids limit


@dataclass(frozen=True)
class Scored:
    score: float
    threshold: float | None
    model_version: str
    embedding: np.ndarray | None  # only computed for anomalous images


class Scorer(Protocol):
    def score(self, image: Image.Image, category: str) -> Scored: ...


class PatchCoreScorer:
    """Registry model: image score vs the registry threshold, embedding when anomalous."""

    def __init__(self, registry: ModelRegistry, top_k: int) -> None:
        self.registry = registry
        self.top_k = top_k

    def score(self, image: Image.Image, category: str) -> Scored:
        detector, info = self.registry.get(category)  # UnknownModelError -> 404 in the API
        x = image_to_tensor(image, detector.image_size, normalize=detector.normalized_input)
        x = x.unsqueeze(0)
        with torch.inference_mode():
            out = detector.predict(x)
        score, threshold = float(out.scores[0]), info.threshold
        embedding = None
        if threshold is not None and score > threshold:
            with torch.inference_mode():
                feats = detector.patch_features(x)[0]
            embedding = region_embedding(feats, out.maps[0], self.top_k)
        return Scored(score, threshold, info.version, embedding)


@dataclass(frozen=True)
class PartContext:
    part_id: str | None = None
    run_id: str | None = None
    line_id: str | None = None
    observed_at: datetime | None = None


@dataclass(frozen=True)
class RoutingResult:
    observation: Observation
    cluster_id: uuid.UUID | None = None
    alert_event_id: uuid.UUID | None = None
    known_name: str | None = None


def _view(c: ClusterRecord, members: list[uuid.UUID]) -> ClusterView:
    return ClusterView(
        cluster_id=c.id,
        members=frozenset(members),
        centroid=c.centroid,
        radius=c.radius,
        named=c.status == ClusterStatus.NAMED,
        name=c.defect_class_name,
    )


class DiscoveryService:
    def __init__(
        self,
        scorer: Scorer,
        store: DiscoveryStore,
        params: TriggerParams,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.scorer = scorer
        self.store = store
        self.params = params
        self.clock = clock
        self._lock = threading.Lock()

    def route_observation(
        self, image: Image.Image, category: str, ctx: PartContext | None = None
    ) -> RoutingResult:
        ctx = ctx or PartContext()
        s = self.scorer.score(image, category)
        base = Observation(
            id=uuid7(),
            category=category,
            observed_at=ctx.observed_at or self.clock(),
            score=s.score,
            threshold=s.threshold,
            decision=(
                "undecided"
                if s.threshold is None
                else "anomalous"
                if s.score > s.threshold
                else "normal"
            ),
            routing=Routing.NORMAL,
            model_version=s.model_version,
            part_id=ctx.part_id,
            run_id=ctx.run_id,
            line_id=ctx.line_id,
        )
        if base.decision != "anomalous" or s.embedding is None:
            undecided = base.decision == "undecided"
            obs = replace(base, routing=Routing.UNDECIDED if undecided else Routing.NORMAL)
            self.store.add_observation(obs)
            return RoutingResult(obs)

        with self._lock:
            views = [_view(c, self.store.members(c.id)) for c in self.store.clusters(category)]
            known = assign_known(s.embedding, views, self.params.known_radius_factor)
            if known is not None:
                obs = replace(
                    base,
                    routing=Routing.KNOWN,
                    embedding=s.embedding,
                    known_cluster_id=known.cluster_id,
                )
                self.store.add_observation(obs)
                return RoutingResult(obs, cluster_id=known.cluster_id, known_name=known.name)

            obs = replace(base, routing=Routing.UNKNOWN_BUFFERED, embedding=s.embedding)
            self.store.add_observation(obs)
            alert_id, cluster_id = self._recluster(category, views)
        # The new observation may have joined a cluster (alerted or grown).
        joined = self.store.observations([obs.id])[0].cluster_id
        return RoutingResult(obs, cluster_id=joined or cluster_id, alert_event_id=alert_id)

    def _recluster(
        self, category: str, views: list[ClusterView]
    ) -> tuple[uuid.UUID | None, uuid.UUID | None]:
        buffer = self.store.buffer(category)
        points = [BufferedPoint(o.id, o.embedding) for o in buffer if o.embedding is not None]
        result = find_clusters(points, views, self.params)
        by_id = {o.id: o for o in buffer}
        for cluster_id, members, geo in result.grown:
            last = max(by_id[m].observed_at for m in members if m in by_id)
            self.store.grow_cluster(
                cluster_id, sorted(members), geo.centroid, geo.radius, last, geo.representative
            )
        for named_id, members in result.near_named:
            _log.info(
                "group_near_named_class",
                category=category,
                named_cluster_id=str(named_id),
                size=len(members),
            )
        alert: tuple[uuid.UUID | None, uuid.UUID | None] = (None, None)
        for members, geo in result.new:
            obs = sorted((by_id[m] for m in members), key=lambda o: (o.observed_at, o.id))
            event = self._event(category, obs, geo.representative)
            cluster = ClusterRecord(
                id=event.payload.cluster_id,
                category=category,
                status=ClusterStatus.ALERTED,
                member_count=len(obs),
                centroid=geo.centroid,
                radius=geo.radius,
                first_seen=obs[0].observed_at,
                last_seen=obs[-1].observed_at,
                representative_id=geo.representative,
                alerted_event_id=event.event_id,
            )
            self.store.create_alerted_cluster(cluster, [o.id for o in obs], event)
            _log.info(
                "new_defect_cluster",
                category=category,
                cluster_id=str(cluster.id),
                size=len(obs),
                event_id=str(event.event_id),
            )
            alert = (event.event_id, cluster.id)
        return alert

    def _event(
        self, category: str, obs: list[Observation], representative: uuid.UUID
    ) -> NewDefectClusterEvent:
        parts = tuple(o.part_id for o in obs if o.part_id is not None)
        runs = {o.run_id for o in obs if o.run_id is not None}
        lines = {o.line_id for o in obs if o.line_id is not None}
        linked = bool(parts) or bool(runs)
        # Most recent members first if the cluster is larger than the event limit.
        ids = tuple(o.id for o in obs[-MAX_EVENT_IDS:])
        if representative not in ids:
            representative = ids[-1]
        scores = [o.score for o in obs]
        return NewDefectClusterEvent(
            payload=NewDefectClusterPayload(
                cluster_id=uuid7(),
                category=category,
                size=len(obs),
                first_seen=obs[0].observed_at,
                last_seen=obs[-1].observed_at,
                inspection_ids=ids,
                part_ids=parts[-MAX_EVENT_IDS:],
                model_version=obs[-1].model_version,
                line_id=lines.pop() if len(lines) == 1 else None,
                simulator_run_id=runs.pop() if len(runs) == 1 else None,
                linkage="synthetic_simulator" if linked else "none",
                representative_inspection_id=representative,
                score_max=max(scores),
                score_median=float(median(scores)),
            )
        )
