"""BM25, vector search and reciprocal-rank fusion on toy data."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from iqs_diagnosis.retrieval.base import Hit, ranked
from iqs_diagnosis.retrieval.bm25 import BM25Index
from iqs_diagnosis.retrieval.fusion import rrf
from iqs_diagnosis.retrieval.hybrid import HybridRetriever
from iqs_diagnosis.retrieval.vector import Standardizer, VectorIndex

from .synthetic import library, profile


def test_bm25_ranks_matching_document_first() -> None:
    idx = BM25Index(
        [
            ("a", "reject_rate_increase gauge_check_drift dimension_mean_shift"),
            ("b", "reject_rate_increase power_deviation power_deviation@M1"),
            ("c", "reject_rate_increase vibration_rise tool_wear_rate_rise"),
        ]
    )
    hits = idx.search("power_deviation", 3)
    assert hits[0].case_id == "b" and hits[0].rank == 1
    assert [h.rank for h in hits] == [1, 2, 3]


def test_bm25_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        BM25Index([])
    with pytest.raises(ValueError, match="duplicate"):
        BM25Index([("a", "x"), ("a", "y")])


def test_ranked_breaks_ties_by_id() -> None:
    assert [h.case_id for h in ranked([("b", 1.0), ("a", 1.0), ("c", 2.0)], 3)] == ["c", "a", "b"]


def test_vector_identical_profile_has_similarity_one() -> None:
    names = ["x", "y"]
    idx = VectorIndex(names, [("a", {"x": 0.0, "y": 1.0}), ("b", {"x": 5.0, "y": -1.0})])
    assert idx.max_similarity({"x": 5.0, "y": -1.0}) == pytest.approx(1.0)
    hits = idx.search({"x": 4.0, "y": -1.0}, 2)
    assert hits[0].case_id == "b" and hits[0].score > hits[1].score


def test_standardizer_floors_constant_features() -> None:
    s = Standardizer.fit(["x"], [{"x": 1.0}, {"x": 1.0}])
    z = s.transform([{"x": 2.0}])
    assert np.isfinite(z).all()
    with pytest.raises(KeyError):
        s.transform([{"y": 1.0}])


def test_rrf_hand_computed() -> None:
    a = [Hit("x", 9, 1), Hit("y", 8, 2)]
    b = [Hit("y", 0.9, 1), Hit("z", 0.8, 2)]
    fused = rrf([a, b], k=60)
    scores = {h.case_id: h.score for h in fused}
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["z"] == pytest.approx(1 / 62)
    assert [h.case_id for h in fused] == ["y", "x", "z"]


@given(
    st.lists(
        st.permutations(["a", "b", "c", "d", "e"]),
        min_size=1,
        max_size=4,
    )
)
def test_rrf_item_first_everywhere_is_first(orders: list[list[str]]) -> None:
    lists = [[Hit(cid, 0.0, i + 1) for i, cid in enumerate(o)] for o in orders]
    fused = rrf(lists)
    assert {h.case_id for h in fused} == {"a", "b", "c", "d", "e"}
    firsts = {o[0] for o in orders}
    if len(firsts) == 1:
        assert fused[0].case_id == firsts.pop()
    assert all(math.isfinite(h.score) for h in fused)


def test_hybrid_finds_same_cause_neighbours() -> None:
    lib = library(["MACHINE.TOOL_WEAR", "MACHINE.DRIVE_FAULT", "MATERIAL.BAD_BATCH"], 10)
    r = HybridRetriever(lib, depth=10)
    q = profile("Q", "MACHINE.DRIVE_FAULT", np.random.default_rng(99))
    res = r.search(q)
    assert len(res.fused) == 10
    assert all(r.cause_of(h.case_id) == "MACHINE.DRIVE_FAULT" for h in res.fused[:5])
    assert 0 < res.max_similarity <= 1


def test_hybrid_rejects_duplicate_library_ids() -> None:
    lib = library(["MACHINE.TOOL_WEAR"], 2)
    with pytest.raises(ValueError, match="duplicate"):
        HybridRetriever([lib[0], lib[0]])
