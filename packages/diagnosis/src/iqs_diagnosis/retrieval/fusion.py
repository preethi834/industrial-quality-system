"""Reciprocal-rank fusion (Cormack et al. 2009): score = sum over lists of 1 / (k + rank)."""

from __future__ import annotations

from collections.abc import Sequence

from iqs_diagnosis.retrieval.base import Hit, ranked

RRF_K = 60


def rrf(rankings: Sequence[Sequence[Hit]], *, k: int = RRF_K, top: int | None = None) -> list[Hit]:
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[str, float] = {}
    for hits in rankings:
        for h in hits:
            scores[h.case_id] = scores.get(h.case_id, 0.0) + 1.0 / (k + h.rank)
    return ranked(list(scores.items()), top if top is not None else len(scores))
