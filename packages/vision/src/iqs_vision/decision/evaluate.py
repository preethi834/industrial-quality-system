"""Evaluation of the decision policies in expected cost per 10,000 parts (step 6b).

Data per category (all from committed manifests + saved scores, no inference):

- fit set   = sup_train ∪ sup_val (labelled test images) -> Platt calibrator
- test set  = sup_test -> action rates per class
- calib set = good images held out from the detector -> conformal threshold q_hat (step 2)

The fit set must be disjoint (by path and content) from calib and sup_test: gate 2.

Expected cost per 10,000 parts uses the class-conditional action rates measured on sup_test
and the assumed production defect rate pi, so the test-set defect share does not leak in:

    cost = 10,000 * [pi * E(cost | defect) + (1 - pi) * E(cost | good)]

    good:   accept 0       reject C_fr   review C_rev + eps_fr * C_fr
    defect: accept C_esc   reject 0      review C_rev + eps_miss * C_esc

Intervals: stratified bootstrap over sup_test images, calibrator and q_hat held fixed, so they
cover test-sample noise only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iqs_vision.calibration.beta_bound import clopper_pearson
from iqs_vision.calibration.calibrate import ScoreRecord
from iqs_vision.data.splits import SplitManifest
from iqs_vision.decision.calibrate_proba import PlattCalibrator, fit_platt
from iqs_vision.decision.costs import CostConfig
from iqs_vision.decision.policy import (
    ACCEPT,
    REJECT,
    REVIEW,
    decide_guaranteed,
    decide_threshold,
    decide_unconstrained,
)

POLICIES = ("guaranteed", "unconstrained", "fixed_conformal", "fixed_provisional")
FIT_SPLITS = ("sup_train", "sup_val")
MIN_PER_CLASS = 4  # sup_test needs >= 4 good and >= 4 defective images to be evaluated


class DecisionLeakageError(ValueError):
    """The probability-calibration images overlap the conformal calibration or test set."""


@dataclass(frozen=True)
class CategoryData:
    dataset: str
    category: str
    fit_scores: np.ndarray
    fit_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    good_test_all: np.ndarray  # every official good test image (for the flag rate)
    calib_scores: np.ndarray

    @property
    def n_test_good(self) -> int:
        return int((self.test_labels == 0).sum())

    @property
    def n_test_defect(self) -> int:
        return int((self.test_labels == 1).sum())


def check_fit_disjoint(manifest: SplitManifest) -> None:
    """Gate 2: fit set ∩ (calib ∪ sup_test) = ∅ by path and by sha256."""
    fit = [e for e in manifest.entries if e.sup_split in FIT_SPLITS]
    other = [e for e in manifest.entries if e.split == "calib" or e.sup_split == "sup_test"]
    paths = {e.path for e in fit} & {e.path for e in other}
    hashes = {e.sha256 for e in fit} & {e.sha256 for e in other}
    if paths or hashes:
        raise DecisionLeakageError(
            f"{manifest.dataset}/{manifest.category}: {len(paths)} paths and {len(hashes)} "
            "hashes shared between the probability-fit set and calib / sup_test"
        )


def category_data(
    manifest: SplitManifest, meta: dict[str, Any], records: Sequence[ScoreRecord]
) -> CategoryData:
    """Join saved scores to the manifest sub-splits (path and sha256 must both match)."""
    if meta.get("manifest_fingerprint") != manifest.fingerprint():
        raise ValueError(f"{manifest.category}: scores were computed from another manifest")
    check_fit_disjoint(manifest)
    by_path = {r.path: r for r in records}
    fit_s, fit_y, test_s, test_y = [], [], [], []
    for e in manifest.entries:
        if e.split != "test":
            continue
        r = by_path.get(e.path)
        if r is None or r.sha256 != e.sha256 or r.label != e.label:
            raise ValueError(f"{e.path}: no matching score record")
        if e.sup_split in FIT_SPLITS:
            fit_s.append(r.score)
            fit_y.append(e.label)
        elif e.sup_split == "sup_test":
            test_s.append(r.score)
            test_y.append(e.label)
    good_all = [r.score for r in records if r.split == "test" and r.label == 0]
    calib = [r.score for r in records if r.split == "calib"]
    return CategoryData(
        dataset=manifest.dataset,
        category=manifest.category,
        fit_scores=np.array(fit_s),
        fit_labels=np.array(fit_y, dtype=np.int64),
        test_scores=np.array(test_s),
        test_labels=np.array(test_y, dtype=np.int64),
        good_test_all=np.array(good_all),
        calib_scores=np.array(calib),
    )


def image_costs(actions: np.ndarray, labels: np.ndarray, c: CostConfig) -> np.ndarray:
    """Expected cost of each (action, true label) pair, human-review errors included."""
    good = np.array([0.0, c.c_fr, c.c_rev + c.eps_fr * c.c_fr])
    defect = np.array([c.c_esc, 0.0, c.c_rev + c.eps_miss * c.c_esc])
    a = actions.astype(np.int64)
    return np.where(labels == 1, defect[a], good[a])


def cost_per_10k(actions: np.ndarray, labels: np.ndarray, c: CostConfig) -> float:
    ic = image_costs(actions, labels, c)
    return float(1e4 * (c.pi * ic[labels == 1].mean() + (1 - c.pi) * ic[labels == 0].mean()))


def _rates(actions: np.ndarray, labels: np.ndarray, pi: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, y in (("good", 0), ("defect", 1)):
        a = actions[labels == y]
        out[name] = {
            "accept": float(np.mean(a == ACCEPT)),
            "reject": float(np.mean(a == REJECT)),
            "review": float(np.mean(a == REVIEW)),
        }
    for act in ("review", "reject"):
        # production-weighted share of parts with this action
        out[f"{act}_share"] = pi * out["defect"][act] + (1 - pi) * out["good"][act]
    out["flag_share"] = out["review_share"] + out["reject_share"]
    return out


def policy_actions(
    d: CategoryData,
    cal: PlattCalibrator,
    q_hat: float,
    c: CostConfig,
    scores: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    s = d.test_scores if scores is None else scores
    p = cal.predict(s, c.pi)
    return {
        "guaranteed": decide_guaranteed(p, s, q_hat, c),
        "unconstrained": decide_unconstrained(p, c),
        "fixed_conformal": decide_threshold(s, q_hat),
        "fixed_provisional": decide_threshold(s, float(d.calib_scores.max())),
    }


def bootstrap_indices(
    labels: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified resamples: (n_boot, n_good) and (n_boot, n_defect) index arrays."""
    g, dft = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    return (
        g[rng.integers(0, g.size, (n_boot, g.size))],
        dft[rng.integers(0, dft.size, (n_boot, dft.size))],
    )


def boot_cost(ic: np.ndarray, idx: tuple[np.ndarray, np.ndarray], pi: float) -> np.ndarray:
    ig, idf = idx
    return 1e4 * (pi * ic[idf].mean(axis=1) + (1 - pi) * ic[ig].mean(axis=1))


def ci95(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def flag_rate(good_scores: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
    x, m = int(flagged.sum()), int(good_scores.size)
    lo, hi = clopper_pearson(x, m)
    return {"flagged": x, "n_good": m, "rate": x / m, "ci95": [lo, hi]}


def evaluate_category(
    d: CategoryData,
    q_hat: float,
    alpha: float,
    c: CostConfig,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Headline numbers for one category, plus the bootstrap replicates for pooling."""
    cal = fit_platt(d.fit_scores, d.fit_labels)
    acts = policy_actions(d, cal, q_hat, c)
    idx = bootstrap_indices(d.test_labels, n_boot, rng)
    reps: dict[str, np.ndarray] = {}
    pol: dict[str, Any] = {}
    for name, a in acts.items():
        ic = image_costs(a, d.test_labels, c)
        reps[name] = boot_cost(ic, idx, c.pi)
        pol[name] = {
            "cost_per_10k": cost_per_10k(a, d.test_labels, c),
            "cost_per_10k_ci95": ci95(reps[name]),
            **_rates(a, d.test_labels, c.pi),
        }
    saving = {
        name: {
            "vs_fixed_conformal": pol["fixed_conformal"]["cost_per_10k"]
            - pol[name]["cost_per_10k"],
            "ci95": ci95(reps["fixed_conformal"] - reps[name]),
        }
        for name in ("guaranteed", "unconstrained")
    }
    sup_good = d.test_labels == 0
    false_alarm = {
        # the guaranteed policy flags iff score > q_hat, independent of p: every official
        # good test image can be used without leakage
        "guaranteed_all_test_good": flag_rate(d.good_test_all, d.good_test_all > q_hat),
        "guaranteed_sup_test": flag_rate(
            d.test_scores[sup_good], acts["guaranteed"][sup_good] != ACCEPT
        ),
        "unconstrained_sup_test_no_guarantee": flag_rate(
            d.test_scores[sup_good], acts["unconstrained"][sup_good] != ACCEPT
        ),
    }
    row = {
        "dataset": d.dataset,
        "category": d.category,
        "alpha": alpha,
        "q_hat": q_hat,
        "provisional_threshold": float(d.calib_scores.max()),
        "n": {
            "calib": int(d.calib_scores.size),
            "fit_good": int((d.fit_labels == 0).sum()),
            "fit_defect": int((d.fit_labels == 1).sum()),
            "test_good": d.n_test_good,
            "test_defect": d.n_test_defect,
        },
        "platt": cal.to_dict(),
        "policies": pol,
        "saving": saving,
        "false_alarm": false_alarm,
    }
    return row, reps


def point_costs(
    d: CategoryData, cal: PlattCalibrator, q_hat: float, c: CostConfig
) -> dict[str, dict[str, float]]:
    """Cost per 10k and review share of every policy, no bootstrap (sensitivity sweeps)."""
    out = {}
    for name, a in policy_actions(d, cal, q_hat, c).items():
        r = _rates(a, d.test_labels, c.pi)
        out[name] = {
            "cost_per_10k": cost_per_10k(a, d.test_labels, c),
            "review_share": r["review_share"],
            "flag_share": r["flag_share"],
        }
    return out
