"""Evaluation metrics: proportions with Wilson score intervals, latency summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

Z95 = 1.959963984540054


class Proportion(BaseModel):
    """k successes out of n with a 95 % Wilson score interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int
    n: int
    value: float
    low: float
    high: float

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a (n=0)"
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}] ({self.k}/{self.n})"


def wilson(k: int, n: int, z: float = Z95) -> Proportion:
    if n < 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if n == 0:
        return Proportion(k=0, n=0, value=math.nan, low=math.nan, high=math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Exact endpoints at 0 and n (avoid -1e-17 / 1+1e-16 from rounding).
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return Proportion(k=k, n=n, value=p, low=low, high=high)


class Latency(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    p50_ms: float
    p95_ms: float
    max_ms: float


def latency(values_ms: Sequence[float]) -> Latency:
    if not values_ms:
        raise ValueError("no latency values")
    xs = sorted(values_ms)

    def q(p: float) -> float:
        # Nearest-rank percentile.
        return xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]

    return Latency(n=len(xs), p50_ms=q(0.5), p95_ms=q(0.95), max_ms=xs[-1])
