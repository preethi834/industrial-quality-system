"""Anomaly score -> P(defect): Platt scaling (used for decisions) and isotonic (comparison).

Fitted per category on labelled images (sup_train ∪ sup_val) that are disjoint from both the
conformal calibration set and sup_test (step 6 gate 2; the leak check is in the evaluation
script and its tests).

Prior shift. The fit images are about two thirds defective, a plant has far fewer. If the
calibrator estimates the posterior under the fit prior pi_fit, Bayes' rule gives the posterior
under a production prior pi by rescaling the odds (the class-conditional score densities stay
the same):

    odds_pi(s) = odds_fit(s) * [pi / (1 - pi)] / [pi_fit / (1 - pi_fit)]

Platt uses the smoothed targets of Platt (1999): (N+ + 1) / (N+ + 2) for defects and
1 / (N- + 2) for good images, so perfectly separated categories still give a finite slope.
The slope is constrained to be >= 0, so p never decreases as the score increases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import optimize, special
from sklearn.isotonic import IsotonicRegression

Method = Literal["platt", "isotonic"]
_P_CLIP = 1e-6


def _xy(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be 0 (good) or 1 (defect)")
    if not (y == 0).any() or not (y == 1).any():
        raise ValueError("need both good and defective images to fit")
    return s, y


def shift_prior(p: np.ndarray, pi_fit: float, pi: float) -> np.ndarray:
    """Posterior under prior pi from a posterior under prior pi_fit (odds rescaling)."""
    for name, v in (("pi_fit", pi_fit), ("pi", pi)):
        if not 0 < v < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    q = np.clip(np.asarray(p, dtype=np.float64), _P_CLIP, 1 - _P_CLIP)
    log_odds = special.logit(q) + special.logit(pi) - special.logit(pi_fit)
    return np.asarray(special.expit(log_odds), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """p_fit(s) = expit(a * (s - mean) / std + b), a >= 0."""

    a: float
    b: float
    mean: float
    std: float
    pi_fit: float  # defect share of the fit images
    n_good: int
    n_defect: int
    method: Literal["platt"] = "platt"

    def predict_fit_prior(self, scores: np.ndarray) -> np.ndarray:
        z = (np.asarray(scores, dtype=np.float64) - self.mean) / self.std
        return np.asarray(special.expit(self.a * z + self.b), dtype=np.float64)

    def predict(self, scores: np.ndarray, pi: float) -> np.ndarray:
        """P(defect | score) under the production defect rate pi."""
        return shift_prior(self.predict_fit_prior(scores), self.pi_fit, pi)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlattCalibrator:
        if d.get("method") != "platt":
            raise ValueError("not a Platt calibrator")
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            mean=float(d["mean"]),
            std=float(d["std"]),
            pi_fit=float(d["pi_fit"]),
            n_good=int(d["n_good"]),
            n_defect=int(d["n_defect"]),
        )


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    s, y = _xy(scores, labels)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    mean = float(s.mean())
    std = float(s.std()) or 1.0
    z = (s - mean) / std

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        f = w[0] * z + w[1]
        # cross-entropy with soft targets: sum log(1 + e^f) - t f
        val = float(np.sum(np.logaddexp(0.0, f) - t * f))
        r = special.expit(f) - t
        return val, np.array([float(r @ z), float(r.sum())])

    res = optimize.minimize(
        loss,
        x0=np.array([1.0, 0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None)],
    )
    if not res.success:
        raise RuntimeError(f"Platt fit did not converge: {res.message}")
    return PlattCalibrator(
        a=float(res.x[0]),
        b=float(res.x[1]),
        mean=mean,
        std=std,
        pi_fit=n_pos / (n_pos + n_neg),
        n_good=n_neg,
        n_defect=n_pos,
    )


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    model: IsotonicRegression
    pi_fit: float
    method: Literal["isotonic"] = "isotonic"

    def predict_fit_prior(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(scores, dtype=np.float64)), np.float64)

    def predict(self, scores: np.ndarray, pi: float) -> np.ndarray:
        return shift_prior(self.predict_fit_prior(scores), self.pi_fit, pi)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    s, y = _xy(scores, labels)
    m = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    m.fit(s, y.astype(np.float64))
    return IsotonicCalibrator(model=m, pi_fit=float(y.mean()))


# --- calibration quality --------------------------------------------------------------------


def brier(p: np.ndarray, labels: np.ndarray) -> float:
    q = np.asarray(p, dtype=np.float64)
    return float(np.mean((q - np.asarray(labels, dtype=np.float64)) ** 2))


def ece(p: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    q = np.asarray(p, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    idx = np.minimum((q * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(q[m].mean() - y[m].mean())
    return float(total)


def reliability(p: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Per-bin (mean predicted, observed defect share, count) for a reliability curve."""
    q = np.asarray(p, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    idx = np.minimum((q * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append(
                {
                    "bin_lo": b / bins,
                    "mean_p": float(q[m].mean()),
                    "observed": float(y[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return out
