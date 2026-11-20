"""Cause probabilities from retrieval votes (no LLM involved).

The ``n_neighbours`` best fused hits vote for their confirmed cause, weighted by
their fused (RRF) score; weights are normalised to sum to 1. With additive smoothing
``alpha`` over all taxonomy causes::

    p(c) = (W_c + alpha) / (1 + alpha * C)

so a cause that no neighbour has keeps a small, non-zero probability (it can still
enter a conformal set in step 4). ``n_neighbours`` and ``alpha`` are chosen on the dev
split by negative log-likelihood (``fit_vote_params``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from iqs_diagnosis.retrieval.base import Hit

DEFAULT_NEIGHBOURS = (3, 5, 10, 20, 40)
DEFAULT_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
PROB_FLOOR = 1e-12


@dataclass(frozen=True)
class VoteParams:
    n_neighbours: int = 10
    alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.n_neighbours < 1 or self.alpha <= 0:
            raise ValueError("n_neighbours must be >= 1 and alpha > 0")


def vote_probabilities(
    hits: Sequence[Hit],
    cause_of: Callable[[str], str],
    causes: Sequence[str],
    params: VoteParams,
) -> dict[str, float]:
    """Probability for every cause in ``causes`` (same order), summing to 1."""
    if len(set(causes)) != len(causes) or not causes:
        raise ValueError("causes must be a non-empty list of unique codes")
    votes = dict.fromkeys(causes, 0.0)
    used = list(hits)[: params.n_neighbours]
    total = sum(h.score for h in used)
    for h in used:
        cause = cause_of(h.case_id)
        if cause not in votes:
            raise ValueError(f"library cause {cause!r} is not in the cause list")
        votes[cause] += h.score / total if total > 0 else 1.0 / len(used)
    denom = 1.0 + params.alpha * len(causes)
    return {c: (votes[c] + params.alpha) / denom for c in causes}


def top_k(probabilities: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    """Highest probabilities first; ties broken by cause code."""
    return sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def mean_nll(
    hit_lists: Sequence[Sequence[Hit]],
    truths: Sequence[str],
    cause_of: Callable[[str], str],
    causes: Sequence[str],
    params: VoteParams,
) -> float:
    if len(hit_lists) != len(truths) or not truths:
        raise ValueError("need one truth per hit list (and at least one)")
    total = 0.0
    for hits, truth in zip(hit_lists, truths, strict=True):
        p = vote_probabilities(hits, cause_of, causes, params)[truth]
        total -= math.log(max(p, PROB_FLOOR))
    return total / len(truths)


def fit_vote_params(
    hit_lists: Sequence[Sequence[Hit]],
    truths: Sequence[str],
    cause_of: Callable[[str], str],
    causes: Sequence[str],
    *,
    neighbours: Sequence[int] = DEFAULT_NEIGHBOURS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[VoteParams, float]:
    """Grid search for the lowest mean NLL on labelled dev queries (known causes only).

    Ties prefer more neighbours, then larger alpha (smoother, more conservative).
    """
    best: tuple[float, int, float] | None = None
    for n in neighbours:
        for a in alphas:
            nll = mean_nll(hit_lists, truths, cause_of, causes, VoteParams(n, a))
            key = (round(nll, 12), -n, -a)
            if best is None or key < (round(best[0], 12), -best[1], -best[2]):
                best = (nll, n, a)
    assert best is not None
    return VoteParams(n_neighbours=best[1], alpha=best[2]), best[0]
