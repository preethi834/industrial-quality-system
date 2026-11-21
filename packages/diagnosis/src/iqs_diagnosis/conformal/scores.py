"""Nonconformity scores and prediction sets over taxonomy causes (blueprint §2.8).

Probabilities are a mapping cause -> p over the full taxonomy (summing to 1).

- **LAC** (least ambiguous set-valued classifier): ``s(x, y) = 1 - p(y | x)``;
  set = every cause with ``s <= q``. Smallest sets on average; can be **empty**, which the
  agent shows as "no cause reaches the threshold: treat as new / escalate".
- **APS** (adaptive prediction sets, non-randomised): causes sorted by probability
  (descending, ties by cause code so the order is fixed in advance); ``s(x, y)`` = total
  probability of y and every cause ranked before it. The set adds causes in that order
  until the cumulative probability reaches ``q`` (the cause that crosses ``q`` is
  included), so it is never empty and grows with the case's ambiguity. It contains
  ``{y : s(x, y) <= q}``, so the marginal guarantee holds and is slightly conservative.

Tied probabilities (frequent with retrieval votes: every cause no neighbour voted for
gets the same smoothing mass) make both scores discrete, and coverage conservative
(see ``iqs_diagnosis.conformal.split``). The fixed tie order in APS is part of the score
function and is chosen before calibration, as the guarantee requires.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

TOL = 1e-12  # float tolerance for "score <= q" (probabilities are sums of floats)


class Method(StrEnum):
    LAC = "lac"
    APS = "aps"


def _ordered(probs: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))


def lac_score(probs: Mapping[str, float], label: str) -> float:
    return 1.0 - float(probs[label])


def aps_score(probs: Mapping[str, float], label: str) -> float:
    total = 0.0
    for code, p in _ordered(probs):
        total += p
        if code == label:
            return min(total, 1.0)
    raise KeyError(label)


def score(method: Method, probs: Mapping[str, float], label: str) -> float:
    return lac_score(probs, label) if method is Method.LAC else aps_score(probs, label)


def prediction_set(method: Method, probs: Mapping[str, float], qhat: float) -> list[str]:
    """Causes in the set, most probable first."""
    ordered = _ordered(probs)
    if method is Method.LAC:
        return [c for c, p in ordered if 1.0 - p <= qhat + TOL]
    out: list[str] = []
    total = 0.0
    for code, p in ordered:
        out.append(code)
        total += p
        if total >= qhat - TOL:
            break
    return out
