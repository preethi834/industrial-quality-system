"""Minimal discovery trigger (step 7): pure domain logic, no I/O.

Blueprint §5 sequencing note: step 7 needs "camera finds a new defect cluster" before the
evaluated discovery layer of step 8c exists. This is the minimal version:

1. an anomalous image whose embedding is within ``radius * known_radius_factor`` of a
   **named** cluster's centroid is a known defect class (no buffering, no alert);
2. otherwise it goes to the category's unknown buffer;
3. HDBSCAN (scikit-learn, core install) over the buffer; a group of at least
   ``min_cluster_size`` images that does not overlap an existing cluster is a **new
   cluster** -> one alert. A group that overlaps an existing cluster (``overlap`` of the
   smaller of the two member sets) is that cluster growing: its members, centroid and
   radius are updated, no new alert.
4. a group is only a cluster if it is **compact**: its radius (``radius_quantile`` of the
   member-to-centroid distances) is at most ``max_radius``. Without this, HDBSCAN with
   ``allow_single_cluster`` turns a handful of unrelated anomalies into one "cluster"
   (checked: 5 random unit vectors in 64-d and 512-d form a single cluster).
5. HDBSCAN can split one defect type into dense parts or label its fringe as noise, so
   close compact parts are merged and buffer points within ``known_radius_factor`` x the
   radius of a core join it (regression test: one growing blob alerts exactly once).
6. (found by the 7g e2e test) a grown cluster's centroid and radius are computed over **all**
   its members, not only the points near the current core; otherwise the radius understates
   the class and, once named, its own fringe falls outside the known-class reach. And a group
   close to a **named** class (same rule as 5) is that class's fringe, not a new defect type:
   it raises no alert and stays buffered (``near_named``).
7. (KNOWN_ISSUES #46, measured in 7h) **distinct images** count, not observations: repeats of
   one image (same pixel hash, ``image_key``) are one point for HDBSCAN, the size rule and the
   geometry, and all their observations join the cluster together. A camera stream that shows
   the same few false-alarm images again and again is not a population of a new defect type.

The parameters are stated starting values, not tuned or evaluated (KNOWN_ISSUES #43).
Embeddings are L2-normalised, so Euclidean distance is a monotone function of cosine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import HDBSCAN

RADIUS_FLOOR = 1e-3  # identical repeated images would otherwise give a zero radius


class TriggerParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: int = Field(default=3, ge=1)
    overlap: float = Field(default=0.5, gt=0.0, le=1.0)
    known_radius_factor: float = Field(default=1.5, gt=0.0)
    radius_quantile: float = Field(default=0.95, gt=0.0, le=1.0)
    # Unit-sphere distance; random unrelated unit vectors sit ~0.9 from their centroid.
    max_radius: float = Field(default=0.5, gt=0.0, le=2.0)
    top_k_patches: int = Field(default=10, ge=1)

    @classmethod
    def load(cls, path: Path) -> TriggerParams:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw.get("trigger", {}))


@dataclass(frozen=True)
class BufferedPoint:
    observation_id: UUID
    embedding: np.ndarray  # (D,), L2-normalised
    # Identity of the image (SHA-256 of its pixels); None = treat as a distinct image.
    image_key: str | None = None


def distinct_images(
    buffer: Sequence[BufferedPoint],
) -> tuple[list[BufferedPoint], dict[UUID, frozenset[UUID]]]:
    """One point per image (the first observation of it, in buffer order) and, per kept
    point, the IDs of every observation of that image."""
    first: dict[str, BufferedPoint] = {}
    ids: dict[str, list[UUID]] = {}
    for p in buffer:
        key = p.image_key if p.image_key is not None else f"obs:{p.observation_id}"
        first.setdefault(key, p)
        ids.setdefault(key, []).append(p.observation_id)
    uniq = list(first.values())
    expand = {first[k].observation_id: frozenset(v) for k, v in ids.items()}
    return uniq, expand


@dataclass(frozen=True)
class ClusterView:
    """What the trigger needs to know about an existing cluster."""

    cluster_id: UUID
    members: frozenset[UUID]
    centroid: np.ndarray
    radius: float
    named: bool
    name: str | None = None


@dataclass(frozen=True)
class ClusterGeometry:
    centroid: np.ndarray
    radius: float
    representative: UUID  # the member closest to the centroid


@dataclass
class ClusteringResult:
    new: list[tuple[frozenset[UUID], ClusterGeometry]] = field(default_factory=list)
    grown: list[tuple[UUID, frozenset[UUID], ClusterGeometry]] = field(default_factory=list)
    # Groups suppressed because they lie close to a named class: (named cluster id, members).
    near_named: list[tuple[UUID, frozenset[UUID]]] = field(default_factory=list)


def l2_normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero embedding cannot be normalised")
    return (v / n).astype(np.float32)


def geometry(points: Sequence[BufferedPoint], quantile: float) -> ClusterGeometry:
    if not points:
        raise ValueError("a cluster needs at least one point")
    x = np.stack([p.embedding for p in points])
    centroid = l2_normalise(x.mean(axis=0))
    d = np.linalg.norm(x - centroid, axis=1)
    radius = max(float(np.quantile(d, quantile)), RADIUS_FLOOR)
    return ClusterGeometry(centroid, radius, points[int(np.argmin(d))].observation_id)


def assign_known(
    embedding: np.ndarray, clusters: Sequence[ClusterView], factor: float
) -> ClusterView | None:
    """The nearest named cluster whose (scaled) radius contains the embedding, if any."""
    best: tuple[float, ClusterView] | None = None
    for c in clusters:
        if not c.named:
            continue
        d = float(np.linalg.norm(embedding - c.centroid))
        if d <= c.radius * factor and (best is None or d < best[0]):
            best = (d, c)
    return None if best is None else best[1]


def _overlaps(a: frozenset[UUID], b: frozenset[UUID], share: float) -> bool:
    smaller = min(len(a), len(b))
    return smaller > 0 and len(a & b) / smaller >= share


def _close(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, factor: float) -> bool:
    return float(np.linalg.norm(c1 - c2)) <= factor * max(r1, r2)


def _merge_close_groups(
    groups: list[list[BufferedPoint]], params: TriggerParams
) -> list[list[BufferedPoint]]:
    """HDBSCAN can split one defect type into several dense parts; merge parts whose
    centroids lie within ``known_radius_factor`` x the larger radius (repeat to a fixpoint)."""
    merged = [list(g) for g in groups]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        geos = [geometry(g, params.radius_quantile) for g in merged]
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                gi, gj = geos[i], geos[j]
                if _close(
                    gi.centroid, gi.radius, gj.centroid, gj.radius, params.known_radius_factor
                ):
                    merged[i] = merged[i] + merged[j]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def find_clusters(
    buffer: Sequence[BufferedPoint], existing: Sequence[ClusterView], params: TriggerParams
) -> ClusteringResult:
    """Cluster the unknown buffer; split the groups into new clusters and grown ones.

    A group continues an existing (alerted, not yet named) cluster if it shares
    ``overlap`` of the smaller member set **or** its centroid is close to that cluster's
    centroid (same rule as the merge above), so a cluster that HDBSCAN splits as it grows
    does not alert twice.
    """
    result = ClusteringResult()
    # Repeats of one image are one point (step 7 above); ``expand`` maps back to observations.
    buffer, expand = distinct_images(buffer)
    if len(buffer) < params.min_cluster_size:
        return result
    x = np.stack([p.embedding for p in buffer])
    labels = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=min(params.min_samples, len(buffer) - 1),
        # One coherent group must be able to form a cluster on its own: early on the
        # buffer holds a single new defect type and nothing else.
        allow_single_cluster=True,
        copy=True,
    ).fit_predict(x)
    groups = [
        [p for p, lab in zip(buffer, labels, strict=True) if lab == label]
        for label in sorted(set(labels.tolist()) - {-1})
    ]
    # Loose groups are not defect types (step 4 below): drop them before merging, or their
    # large radius would make every other group look "close".
    compact = [g for g in groups if geometry(g, params.radius_quantile).radius <= params.max_radius]
    unnamed = [c for c in existing if not c.named]
    named = [c for c in existing if c.named]
    for core in _merge_close_groups(compact, params):
        # HDBSCAN keeps only the densest core of a homogeneous blob and calls its fringe
        # noise; points within the class-match distance of the core belong to it.
        core_geo = geometry(core, params.radius_quantile)
        reach = params.known_radius_factor * core_geo.radius
        pts = [p for p in buffer if float(np.linalg.norm(p.embedding - core_geo.centroid)) <= reach]
        if len(pts) < params.min_cluster_size:  # distinct images
            continue
        members = frozenset().union(*(expand[p.observation_id] for p in pts))
        geo = geometry(pts, params.radius_quantile)
        if geo.radius > params.max_radius:
            continue  # not compact: stays in the buffer
        match = next(
            (
                c
                for c in unnamed
                if _overlaps(members, c.members, params.overlap)
                or _close(
                    geo.centroid, geo.radius, c.centroid, c.radius, params.known_radius_factor
                )
            ),
            None,
        )
        if match is not None:
            union = members | match.members
            # Geometry over every member still in the buffer, not just the core's reach.
            all_pts = [p for p in buffer if expand[p.observation_id] & union]
            result.grown.append(
                (match.cluster_id, union, geometry(all_pts, params.radius_quantile))
            )
            continue
        near = next(
            (
                c
                for c in named
                if _close(
                    geo.centroid, geo.radius, c.centroid, c.radius, params.known_radius_factor
                )
            ),
            None,
        )
        if near is not None:
            result.near_named.append((near.cluster_id, members))
            continue
        result.new.append((members, geo))
    return result
