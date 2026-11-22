"""Statistical validity check of the conformal false-alarm guarantee (step 2 gate 3).

Repeated random splits: the good images the detector never saw (calibration + good test
images of the regular condition) are pooled; each of R resplits draws a calibration set of
the original size n, sets q_hat at alpha and measures the false-alarm rate on the remaining
m good images (and the miss rate on all defective test images). Gate: the mean realised
false-alarm rate <= alpha + 3 standard errors (sd / sqrt(R)).

Pooling makes calibration and test exchangeable by construction, so this checks the
implementation (index rule, strict flagging), not whether the factory's good
parts are exchangeable with the calibration images. The fixed split (official calibration
images -> official good test images) is reported next to it for that question. Pooling also
hides calibration leakage (leaked low scores are shuffled into both sides); leakage is caught
by check_calibration_disjoint and shows up in the fixed split.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from iqs_vision.calibration.beta_bound import predicted_false_alarm
from iqs_vision.calibration.conformal import conformal_index

DEFAULT_RESPLITS = 100


@dataclass(frozen=True, slots=True)
class ResplitResult:
    alpha: float
    n: int
    k: int
    m: int
    resplits: int
    mean_false_alarm: float
    sd_false_alarm: float
    se_false_alarm: float
    gate_limit: float
    passed: bool
    mean_miss: float | None
    predicted_mean: float
    predicted_sd: float
    false_alarm_rates: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resplit_check(
    calib: np.ndarray,
    test_good: np.ndarray,
    test_defect: np.ndarray,
    alpha: float,
    *,
    resplits: int = DEFAULT_RESPLITS,
    seed: int = 0,
    index_rule: Callable[[int, float], int] = conformal_index,
) -> ResplitResult:
    """index_rule is injectable so tests can show that a wrong rank fails the gate."""
    calib, test_good = np.asarray(calib, float), np.asarray(test_good, float)
    n, m = calib.size, test_good.size
    if m < 1:
        raise ValueError("need at least one good test image")
    k = index_rule(n, alpha)  # conformal_index raises when n is too small
    pool = np.concatenate([calib, test_good])
    rng = np.random.default_rng(seed)
    far = np.empty(resplits)
    miss = np.empty(resplits)
    for r in range(resplits):
        perm = rng.permutation(pool.size)
        q = np.sort(pool[perm[:n]])[k - 1]
        far[r] = np.mean(pool[perm[n:]] > q)
        miss[r] = np.mean(test_defect <= q) if test_defect.size else np.nan
    sd = float(far.std(ddof=1)) if resplits > 1 else 0.0
    se = sd / np.sqrt(resplits)
    limit = alpha + 3 * se
    pred = predicted_false_alarm(n, k, m)
    return ResplitResult(
        alpha=alpha,
        n=n,
        k=k,
        m=m,
        resplits=resplits,
        mean_false_alarm=float(far.mean()),
        sd_false_alarm=sd,
        se_false_alarm=float(se),
        gate_limit=float(limit),
        passed=bool(far.mean() <= limit),
        mean_miss=float(np.nanmean(miss)) if test_defect.size else None,
        predicted_mean=pred.mean,
        predicted_sd=pred.sd,
        false_alarm_rates=[float(x) for x in far],
    )
