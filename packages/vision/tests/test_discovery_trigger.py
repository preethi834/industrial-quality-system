"""Discovery trigger domain (step 7): pure functions, synthetic embeddings."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
import torch

from iqs_vision.discovery.embed import region_embedding
from iqs_vision.discovery.trigger import (
    BufferedPoint,
    ClusterView,
    TriggerParams,
    assign_known,
    find_clusters,
    geometry,
    l2_normalise,
)

from .discovery_fakes import unit

REPO = Path(__file__).resolve().parents[3]
P = TriggerParams()


def _points(embs: list[np.ndarray]) -> list[BufferedPoint]:
    return [BufferedPoint(uuid4(), e) for e in embs]


def test_repo_config_matches_defaults() -> None:
    assert TriggerParams.load(REPO / "configs" / "vision" / "discovery.yaml") == TriggerParams()


def test_no_cluster_below_min_size() -> None:
    rng = np.random.default_rng(0)
    c = unit(rng)
    assert find_clusters(_points([unit(rng, c) for _ in range(4)]), [], P).new == []


def test_one_tight_group_alerts_once_at_min_size() -> None:
    rng = np.random.default_rng(1)
    c = unit(rng)
    r = find_clusters(_points([unit(rng, c) for _ in range(5)]), [], P)
    assert len(r.new) == 1 and len(r.new[0][0]) == 5 and r.grown == []
    assert r.new[0][1].radius < P.max_radius


@pytest.mark.parametrize("n", [5, 8, 15, 30])
def test_pure_noise_never_alerts(n: int) -> None:
    """Regression for the allow_single_cluster pitfall (compactness gate)."""
    rng = np.random.default_rng(n)
    assert find_clusters(_points([unit(rng) for _ in range(n)]), [], P).new == []


def test_group_amid_noise_is_found_without_noise_members() -> None:
    rng = np.random.default_rng(2)
    c = unit(rng)
    group = _points([unit(rng, c) for _ in range(6)])
    noise = _points([unit(rng) for _ in range(8)])
    r = find_clusters(group + noise, [], P)
    assert len(r.new) == 1
    assert r.new[0][0] == frozenset(p.observation_id for p in group)


def test_growing_alerted_cluster_does_not_realert() -> None:
    rng = np.random.default_rng(3)
    c = unit(rng)
    first = _points([unit(rng, c) for _ in range(5)])
    members, geo = find_clusters(first, [], P).new[0]
    view = ClusterView(uuid4(), members, geo.centroid, geo.radius, named=False)
    more = first + _points([unit(rng, c) for _ in range(10)])
    r = find_clusters(more, [view], P)
    assert r.new == []
    assert len(r.grown) == 1 and r.grown[0][0] == view.cluster_id
    # All original members stay; a fringe point may remain in the buffer for now.
    assert members <= r.grown[0][1] and len(r.grown[0][1]) >= 12


def test_second_defect_type_alerts_separately() -> None:
    rng = np.random.default_rng(4)
    a, b = unit(rng), unit(rng)
    pa = _points([unit(rng, a) for _ in range(6)])
    members, geo = find_clusters(pa, [], P).new[0]
    view = ClusterView(uuid4(), members, geo.centroid, geo.radius, named=False)
    pb = _points([unit(rng, b) for _ in range(6)])
    r = find_clusters(pa + pb, [view], P)
    assert [m for m, _ in r.new] == [frozenset(p.observation_id for p in pb)]


def test_assign_known_uses_named_clusters_and_radius() -> None:
    rng = np.random.default_rng(5)
    c = unit(rng)
    geo = geometry(_points([unit(rng, c) for _ in range(10)]), 0.95)
    named = ClusterView(UUID(int=1), frozenset(), geo.centroid, geo.radius, named=True, name="x")
    unnamed = ClusterView(UUID(int=2), frozenset(), geo.centroid, geo.radius, named=False)
    near = geo.centroid
    assert assign_known(near, [unnamed], 1.0) is None  # only named classes count
    assert assign_known(near, [unnamed, named], 1.0) is named
    assert assign_known(unit(rng), [named], 1.0) is None  # far away


def test_region_embedding_follows_anomaly_map() -> None:
    feats = torch.zeros(4, 2, 2)
    feats[:, 0, 0] = torch.tensor([1.0, 0, 0, 0])  # top-left patch
    feats[:, 1, 1] = torch.tensor([0, 0, 0, 1.0])  # bottom-right patch
    amap = np.zeros((8, 8), np.float32)
    amap[6:, 6:] = 5.0  # anomaly bottom-right
    e = region_embedding(feats, amap, top_k=1)
    assert np.allclose(e, [0, 0, 0, 1])
    assert np.isclose(np.linalg.norm(region_embedding(torch.rand(16, 4, 4), amap, 3)), 1.0)
    with pytest.raises(ValueError):
        region_embedding(torch.rand(4, 4), amap, 1)


@pytest.mark.parametrize("seed", range(10))
def test_one_blob_never_alerts_twice_as_it_grows(seed: int) -> None:
    """Feed one defect type point by point; at most one new cluster over the whole run."""
    rng = np.random.default_rng(100 + seed)
    c = unit(rng)
    pts: list[BufferedPoint] = []
    views: list[ClusterView] = []
    alerts = 0
    for _ in range(40):
        pts.append(BufferedPoint(uuid4(), unit(rng, c)))
        r = find_clusters(pts, views, P)
        alerts += len(r.new)
        for members, geo in r.new:
            views.append(ClusterView(uuid4(), members, geo.centroid, geo.radius, named=False))
        for cid, members, geo in r.grown:
            views = [
                ClusterView(cid, members, geo.centroid, geo.radius, named=False)
                if v.cluster_id == cid
                else v
                for v in views
            ]
    assert alerts == 1


def _at_distance(rng: np.random.Generator, c: np.ndarray, t: float) -> np.ndarray:
    """A unit vector at angle atan(t) from ``c`` (distance ~ t for small t)."""
    u = rng.normal(size=c.shape)
    u -= u.dot(c) * c
    return l2_normalise(c + t * u / np.linalg.norm(u))


def test_group_near_a_named_class_does_not_alert() -> None:
    """Regression (7g e2e): the fringe of a named class formed a 'new' cluster."""
    rng = np.random.default_rng(7)
    c = unit(rng)
    named = ClusterView(uuid4(), frozenset(), c, 0.1, named=True, name="scratch")
    centre = _at_distance(rng, c, 0.14)  # within 1.5 x max(radius) of the named centroid
    fringe = _points([unit(rng, centre, spread=0.005) for _ in range(6)])
    r = find_clusters(fringe, [named], P)
    assert r.new == [] and r.grown == []
    assert [cid for cid, _ in r.near_named] == [named.cluster_id]
    # The same group with the class far away is a new defect type.
    far = ClusterView(uuid4(), frozenset(), unit(rng), 0.1, named=True, name="other")
    assert len(find_clusters(fringe, [far], P).new) == 1


def test_grown_cluster_geometry_covers_all_members() -> None:
    """Regression (7g e2e): the radius was computed over the core's reach only."""
    rng = np.random.default_rng(8)
    c = unit(rng)
    core = _points([unit(rng, c, spread=0.005) for _ in range(6)])
    edge = _points([_at_distance(rng, c, 0.2) for _ in range(2)])
    members = frozenset(p.observation_id for p in core + edge)
    view = ClusterView(uuid4(), members, c, 0.05, named=False)
    r = find_clusters(core + edge, [view], P)
    assert len(r.grown) == 1
    _, grown_members, geo = r.grown[0]
    assert grown_members == members
    expected = geometry(core + edge, P.radius_quantile)
    assert np.isclose(geo.radius, expected.radius)
    assert geo.radius > 0.15  # the edge members count
