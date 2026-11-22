"""Discovery persistence (step 7): observations, clusters, processed events.

Two implementations of one protocol:

- :class:`InMemoryDiscoveryStore` (unit tests, in-process loop): writes to an
  :class:`~iqs_common.outbox_worker.InMemoryOutbox` under the same lock as the cluster;
- :class:`SqlDiscoveryStore` (Postgres, tables from migration 0005): the cluster row and
  the ``platform.events`` row are written in **one transaction** (transactional outbox).

Cluster membership is ``observation.cluster_id``: an observation belongs to at most one
cluster. Embeddings are ``real[]``; the trigger works in NumPy on at most a few hundred rows
per category, so a vector index would add nothing.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

import numpy as np
from sqlalchemy import (
    REAL,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Uuid,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from iqs_common.db import Base, session_scope
from iqs_common.events import NewDefectClusterEvent
from iqs_common.outbox_worker import InMemoryOutbox, SqlOutbox

VISION_SCHEMA = "vision"


class Routing(StrEnum):
    NORMAL = "normal"
    UNDECIDED = "undecided"  # no threshold for this model
    KNOWN = "known"  # matched a named defect class
    UNKNOWN_BUFFERED = "unknown_buffered"


class ClusterStatus(StrEnum):
    ALERTED = "alerted"
    NAMED = "named"


@dataclass(frozen=True)
class Observation:
    id: uuid.UUID
    category: str
    observed_at: datetime  # inspection time (simulated time for simulator parts)
    score: float
    threshold: float | None
    decision: str
    routing: Routing
    model_version: str
    embedding: np.ndarray | None = None
    # SHA-256 of the decoded pixels (mode, size, bytes): repeats of one image count once
    # in the trigger (KNOWN_ISSUES #46). None for rows written before migration 0006.
    image_sha256: str | None = None
    part_id: str | None = None
    run_id: str | None = None
    line_id: str | None = None
    known_cluster_id: uuid.UUID | None = None
    cluster_id: uuid.UUID | None = None


@dataclass(frozen=True)
class ClusterRecord:
    id: uuid.UUID
    category: str
    status: ClusterStatus
    member_count: int
    centroid: np.ndarray
    radius: float
    first_seen: datetime
    last_seen: datetime
    representative_id: uuid.UUID
    alerted_event_id: uuid.UUID
    defect_class_name: str | None = None
    cause_code: str | None = None
    case_id: uuid.UUID | None = None
    named_at: datetime | None = None


class DiscoveryStore(Protocol):
    def add_observation(self, obs: Observation) -> None: ...
    def observations(self, ids: Iterable[uuid.UUID]) -> list[Observation]: ...
    def buffer(self, category: str) -> list[Observation]:
        """Unknown-buffered observations not in a named cluster."""
        ...

    def clusters(self, category: str | None = None) -> list[ClusterRecord]: ...
    def get_cluster(self, cluster_id: uuid.UUID) -> ClusterRecord | None: ...
    def members(self, cluster_id: uuid.UUID) -> list[uuid.UUID]: ...
    def create_alerted_cluster(
        self, cluster: ClusterRecord, members: Sequence[uuid.UUID], event: NewDefectClusterEvent
    ) -> None:
        """Cluster row + membership + outbox event, atomically."""
        ...

    def grow_cluster(
        self,
        cluster_id: uuid.UUID,
        members: Sequence[uuid.UUID],
        centroid: np.ndarray,
        radius: float,
        last_seen: datetime,
        representative_id: uuid.UUID,
    ) -> None: ...
    def name_cluster(
        self,
        event_id: uuid.UUID,
        cluster_id: uuid.UUID,
        name: str,
        cause_code: str,
        case_id: uuid.UUID,
    ) -> bool:
        """``alerted -> named`` once per event; ``False`` if the event was already handled."""
        ...


class UnknownClusterError(KeyError):
    pass


# --- in memory -------------------------------------------------------------------------


class InMemoryDiscoveryStore:
    def __init__(self, outbox: InMemoryOutbox) -> None:
        self.outbox = outbox
        self._obs: dict[uuid.UUID, Observation] = {}
        self._clusters: dict[uuid.UUID, ClusterRecord] = {}
        self._processed: set[uuid.UUID] = set()
        self._lock = threading.RLock()

    def add_observation(self, obs: Observation) -> None:
        with self._lock:
            if obs.id in self._obs:
                raise ValueError(f"duplicate observation {obs.id}")
            self._obs[obs.id] = obs

    def observations(self, ids: Iterable[uuid.UUID]) -> list[Observation]:
        with self._lock:
            return [self._obs[i] for i in ids if i in self._obs]

    def all_observations(self) -> list[Observation]:
        with self._lock:
            return list(self._obs.values())

    def buffer(self, category: str) -> list[Observation]:
        with self._lock:
            named = {c.id for c in self._clusters.values() if c.status == ClusterStatus.NAMED}
            return [
                o
                for o in self._obs.values()
                if o.category == category
                and o.routing == Routing.UNKNOWN_BUFFERED
                and o.cluster_id not in named
            ]

    def clusters(self, category: str | None = None) -> list[ClusterRecord]:
        with self._lock:
            out = [c for c in self._clusters.values() if category in (None, c.category)]
            return sorted(out, key=lambda c: (c.first_seen, c.id))

    def get_cluster(self, cluster_id: uuid.UUID) -> ClusterRecord | None:
        with self._lock:
            return self._clusters.get(cluster_id)

    def members(self, cluster_id: uuid.UUID) -> list[uuid.UUID]:
        with self._lock:
            return sorted(o.id for o in self._obs.values() if o.cluster_id == cluster_id)

    def _assign(self, cluster_id: uuid.UUID, members: Sequence[uuid.UUID]) -> None:
        for m in members:
            self._obs[m] = replace(self._obs[m], cluster_id=cluster_id)

    def create_alerted_cluster(
        self, cluster: ClusterRecord, members: Sequence[uuid.UUID], event: NewDefectClusterEvent
    ) -> None:
        with self._lock:
            if cluster.id in self._clusters:
                raise ValueError(f"duplicate cluster {cluster.id}")
            self.outbox.append(event)  # raises before any state change on a duplicate
            self._clusters[cluster.id] = cluster
            self._assign(cluster.id, members)

    def grow_cluster(
        self,
        cluster_id: uuid.UUID,
        members: Sequence[uuid.UUID],
        centroid: np.ndarray,
        radius: float,
        last_seen: datetime,
        representative_id: uuid.UUID,
    ) -> None:
        with self._lock:
            c = self._clusters.get(cluster_id)
            if c is None:
                raise UnknownClusterError(cluster_id)
            if c.status != ClusterStatus.ALERTED:
                return  # a named cluster is frozen
            self._assign(cluster_id, members)
            self._clusters[cluster_id] = replace(
                c,
                member_count=len(self.members(cluster_id)),
                centroid=centroid,
                radius=radius,
                last_seen=max(c.last_seen, last_seen),
                representative_id=representative_id,
            )

    def name_cluster(
        self,
        event_id: uuid.UUID,
        cluster_id: uuid.UUID,
        name: str,
        cause_code: str,
        case_id: uuid.UUID,
    ) -> bool:
        with self._lock:
            if event_id in self._processed:
                return False
            c = self._clusters.get(cluster_id)
            if c is None:
                raise UnknownClusterError(cluster_id)
            self._processed.add(event_id)
            if c.status == ClusterStatus.NAMED:
                return False
            self._clusters[cluster_id] = replace(
                c,
                status=ClusterStatus.NAMED,
                defect_class_name=name,
                cause_code=cause_code,
                case_id=case_id,
                named_at=datetime.now(UTC),
            )
            return True


# --- Postgres --------------------------------------------------------------------------

_ROUTING_SQL = ", ".join(f"'{r.value}'" for r in Routing)
_STATUS_SQL = ", ".join(f"'{s.value}'" for s in ClusterStatus)


class ClusterRow(Base):
    __tablename__ = "clusters"
    __table_args__ = (
        CheckConstraint(f"status IN ({_STATUS_SQL})", name="status"),
        {"schema": VISION_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    member_count: Mapped[int] = mapped_column(Integer)
    centroid: Mapped[list[float]] = mapped_column(ARRAY(REAL))
    radius: Mapped[float] = mapped_column(Float)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    representative_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    alerted_event_id: Mapped[uuid.UUID] = mapped_column(Uuid, unique=True)
    defect_class_name: Mapped[str | None] = mapped_column(String(64))
    cause_code: Mapped[str | None] = mapped_column(String(64))
    case_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    named_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ObservationRow(Base):
    __tablename__ = "discovery_observations"
    __table_args__ = (
        CheckConstraint(f"routing IN ({_ROUTING_SQL})", name="routing"),
        {"schema": VISION_SCHEMA},
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16))
    routing: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(128))
    embedding: Mapped[list[float] | None] = mapped_column(ARRAY(REAL))
    image_sha256: Mapped[str | None] = mapped_column(String(64))
    part_id: Mapped[str | None] = mapped_column(String(64))
    run_id: Mapped[str | None] = mapped_column(String(32))
    line_id: Mapped[str | None] = mapped_column(String(64))
    known_cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{VISION_SCHEMA}.clusters.id")
    )
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{VISION_SCHEMA}.clusters.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VisionProcessedEventRow(Base):
    __tablename__ = "processed_events"
    __table_args__ = ({"schema": VISION_SCHEMA},)

    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def _vec(v: list[float] | None) -> np.ndarray | None:
    return None if v is None else np.asarray(v, dtype=np.float32)


def _obs_from_row(r: ObservationRow) -> Observation:
    return Observation(
        id=r.id,
        category=r.category,
        observed_at=r.observed_at,
        score=r.score,
        threshold=r.threshold,
        decision=r.decision,
        routing=Routing(r.routing),
        model_version=r.model_version,
        embedding=_vec(r.embedding),
        image_sha256=r.image_sha256,
        part_id=r.part_id,
        run_id=r.run_id,
        line_id=r.line_id,
        known_cluster_id=r.known_cluster_id,
        cluster_id=r.cluster_id,
    )


def _cluster_from_row(r: ClusterRow) -> ClusterRecord:
    return ClusterRecord(
        id=r.id,
        category=r.category,
        status=ClusterStatus(r.status),
        member_count=r.member_count,
        centroid=np.asarray(r.centroid, dtype=np.float32),
        radius=r.radius,
        first_seen=r.first_seen,
        last_seen=r.last_seen,
        representative_id=r.representative_id,
        alerted_event_id=r.alerted_event_id,
        defect_class_name=r.defect_class_name,
        cause_code=r.cause_code,
        case_id=r.case_id,
        named_at=r.named_at,
    )


class SqlDiscoveryStore:
    """Postgres implementation (exercised by the integration tests, step 7g)."""

    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def add_observation(self, obs: Observation) -> None:
        with session_scope(self._factory) as s:
            s.add(
                ObservationRow(
                    id=obs.id,
                    category=obs.category,
                    observed_at=obs.observed_at,
                    score=obs.score,
                    threshold=obs.threshold,
                    decision=obs.decision,
                    routing=obs.routing.value,
                    model_version=obs.model_version,
                    embedding=None if obs.embedding is None else obs.embedding.tolist(),
                    image_sha256=obs.image_sha256,
                    part_id=obs.part_id,
                    run_id=obs.run_id,
                    line_id=obs.line_id,
                    known_cluster_id=obs.known_cluster_id,
                    cluster_id=obs.cluster_id,
                )
            )

    def observations(self, ids: Iterable[uuid.UUID]) -> list[Observation]:
        wanted = list(ids)
        with session_scope(self._factory) as s:
            rows = s.scalars(select(ObservationRow).where(ObservationRow.id.in_(wanted))).all()
            by_id = {r.id: _obs_from_row(r) for r in rows}
        return [by_id[i] for i in wanted if i in by_id]

    def buffer(self, category: str) -> list[Observation]:
        named = select(ClusterRow.id).where(ClusterRow.status == ClusterStatus.NAMED.value)
        with session_scope(self._factory) as s:
            rows = s.scalars(
                select(ObservationRow)
                .where(
                    ObservationRow.category == category,
                    ObservationRow.routing == Routing.UNKNOWN_BUFFERED.value,
                    (ObservationRow.cluster_id.is_(None))
                    | (ObservationRow.cluster_id.not_in(named)),
                )
                .order_by(ObservationRow.observed_at, ObservationRow.id)
            ).all()
            return [_obs_from_row(r) for r in rows]

    def clusters(self, category: str | None = None) -> list[ClusterRecord]:
        q = select(ClusterRow).order_by(ClusterRow.first_seen, ClusterRow.id)
        if category is not None:
            q = q.where(ClusterRow.category == category)
        with session_scope(self._factory) as s:
            return [_cluster_from_row(r) for r in s.scalars(q).all()]

    def get_cluster(self, cluster_id: uuid.UUID) -> ClusterRecord | None:
        with session_scope(self._factory) as s:
            r = s.get(ClusterRow, cluster_id)
            return None if r is None else _cluster_from_row(r)

    def members(self, cluster_id: uuid.UUID) -> list[uuid.UUID]:
        with session_scope(self._factory) as s:
            return sorted(
                s.scalars(
                    select(ObservationRow.id).where(ObservationRow.cluster_id == cluster_id)
                ).all()
            )

    def create_alerted_cluster(
        self, cluster: ClusterRecord, members: Sequence[uuid.UUID], event: NewDefectClusterEvent
    ) -> None:
        with session_scope(self._factory) as s:
            s.add(
                ClusterRow(
                    id=cluster.id,
                    category=cluster.category,
                    status=cluster.status.value,
                    member_count=cluster.member_count,
                    centroid=cluster.centroid.tolist(),
                    radius=cluster.radius,
                    first_seen=cluster.first_seen,
                    last_seen=cluster.last_seen,
                    representative_id=cluster.representative_id,
                    alerted_event_id=cluster.alerted_event_id,
                )
            )
            s.flush()
            s.execute(
                update(ObservationRow)
                .where(ObservationRow.id.in_(list(members)))
                .values(cluster_id=cluster.id)
            )
            SqlOutbox.append_in(s, event)  # same transaction as the cluster row

    def grow_cluster(
        self,
        cluster_id: uuid.UUID,
        members: Sequence[uuid.UUID],
        centroid: np.ndarray,
        radius: float,
        last_seen: datetime,
        representative_id: uuid.UUID,
    ) -> None:
        with session_scope(self._factory) as s:
            c = s.get(ClusterRow, cluster_id, with_for_update=True)
            if c is None:
                raise UnknownClusterError(cluster_id)
            if c.status != ClusterStatus.ALERTED.value:
                return
            s.execute(
                update(ObservationRow)
                .where(ObservationRow.id.in_(list(members)))
                .values(cluster_id=cluster_id)
            )
            s.flush()
            c.member_count = len(
                s.scalars(
                    select(ObservationRow.id).where(ObservationRow.cluster_id == cluster_id)
                ).all()
            )
            c.centroid = centroid.tolist()
            c.radius = radius
            c.last_seen = max(c.last_seen, last_seen)
            c.representative_id = representative_id

    def name_cluster(
        self,
        event_id: uuid.UUID,
        cluster_id: uuid.UUID,
        name: str,
        cause_code: str,
        case_id: uuid.UUID,
    ) -> bool:
        with session_scope(self._factory) as s:
            inserted = s.execute(
                insert(VisionProcessedEventRow)
                .values(event_id=event_id)
                .on_conflict_do_nothing(index_elements=["event_id"])
                .returning(VisionProcessedEventRow.event_id)
            ).first()
            if inserted is None:
                return False
            c = s.get(ClusterRow, cluster_id, with_for_update=True)
            if c is None:
                raise UnknownClusterError(cluster_id)  # rolls back the processed row too
            if c.status == ClusterStatus.NAMED.value:
                return False
            c.status = ClusterStatus.NAMED.value
            c.defect_class_name = name
            c.cause_code = cause_code
            c.case_id = case_id
            c.named_at = datetime.now(UTC)
            return True
