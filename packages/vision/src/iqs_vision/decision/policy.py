"""Expected-cost decision rules: accept / reject / review (blueprint Rev 2 §2.5 P2-C).

With p = P(defect | score) and the costs of `CostConfig`, the expected cost per part is

    accept = p * C_esc
    reject = (1 - p) * C_fr
    review = C_rev + eps_miss * p * C_esc + eps_fr * (1 - p) * C_fr

All three are linear in p, so each pairwise comparison is a single cut-off on p:

    review <= accept  iff  p >= p_ra = (C_rev + eps_fr C_fr) / (C_esc (1 - eps_miss) + eps_fr C_fr)
    review <= reject  iff  p <= p_rr = (C_fr (1 - eps_fr) - C_rev)
                                       / (C_fr (1 - eps_fr) + eps_miss C_esc)
    reject <= accept  iff  p >= p_ar = C_fr / (C_esc + C_fr)

Unconstrained policy: accept iff p <= lo = min(p_ra, p_ar), reject iff p >= hi = max(p_rr, p_ar),
review in between. When review never wins, lo = hi = p_ar and the rule is two-way. It has
**no** false-alarm guarantee.

Guaranteed policy: a part may be flagged (rejected or reviewed) only if its anomaly score is
above the conformal threshold q_hat; at or below q_hat it is accepted. Above q_hat the cost rule
only chooses reject (p >= p_rr) or review. So P(good part flagged) = P(good score > q_hat) <= alpha
holds exactly as in step 2, whatever the costs and the probability calibration are.

At a cut-off the tied actions have equal expected cost; the rules above fix one of them.
Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from iqs_vision.decision.costs import CostConfig

Action = Literal["accept", "reject", "review"]
ACCEPT, REJECT, REVIEW = 0, 1, 2
ACTIONS: tuple[Action, Action, Action] = ("accept", "reject", "review")


@dataclass(frozen=True, slots=True)
class CutOffs:
    p_review_vs_accept: float  # p_ra
    p_review_vs_reject: float  # p_rr
    p_reject_vs_accept: float  # p_ar
    lo: float  # accept iff p <= lo (unconstrained)
    hi: float  # reject iff p >= hi (both policies use p_rr above q_hat)
    review_possible: bool  # review wins somewhere in the unconstrained policy

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def cut_offs(c: CostConfig) -> CutOffs:
    p_ra = (c.c_rev + c.eps_fr * c.c_fr) / (c.c_esc * (1 - c.eps_miss) + c.eps_fr * c.c_fr)
    p_rr = (c.c_fr * (1 - c.eps_fr) - c.c_rev) / (c.c_fr * (1 - c.eps_fr) + c.eps_miss * c.c_esc)
    p_ar = c.c_fr / (c.c_esc + c.c_fr)
    review = p_ra < p_rr
    lo, hi = (min(p_ra, p_ar), max(p_rr, p_ar)) if review else (p_ar, p_ar)
    return CutOffs(p_ra, p_rr, p_ar, lo, hi, review)


def expected_costs(p: np.ndarray | float, c: CostConfig) -> np.ndarray:
    """Expected cost of (accept, reject, review); shape (..., 3)."""
    q = np.asarray(p, dtype=np.float64)
    accept = q * c.c_esc
    reject = (1 - q) * c.c_fr
    review = c.c_rev + c.eps_miss * q * c.c_esc + c.eps_fr * (1 - q) * c.c_fr
    return np.stack([accept, reject, review], axis=-1)


def _check_p(p: np.ndarray) -> None:
    if p.size and (not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1):
        raise ValueError("p must be in [0, 1]")


def decide_unconstrained(p: np.ndarray, c: CostConfig) -> np.ndarray:
    """Cost-optimal three-way action codes (0 accept, 1 reject, 2 review). No guarantee."""
    q = np.asarray(p, dtype=np.float64)
    _check_p(q)
    k = cut_offs(c)
    out = np.full(q.shape, REVIEW, dtype=np.int8)
    out[q <= k.lo] = ACCEPT
    out[q >= k.hi] = REJECT  # with lo == hi this wins the tie: two-way rule p < p_ar accept
    return out


def decide_guaranteed(
    p: np.ndarray, score: np.ndarray, threshold: float, c: CostConfig
) -> np.ndarray:
    """Accept at or below q_hat; above it, reject or review by expected cost."""
    q = np.asarray(p, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    _check_p(q)
    if q.shape != s.shape:
        raise ValueError("p and score must have the same shape")
    k = cut_offs(c)
    out = np.where(q >= k.p_review_vs_reject, REJECT, REVIEW).astype(np.int8)
    out[s <= threshold] = ACCEPT
    return out


def decide_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    """Fixed-threshold baseline: reject above the threshold, else accept (no review)."""
    s = np.asarray(score, dtype=np.float64)
    return np.where(s > threshold, REJECT, ACCEPT).astype(np.int8)
