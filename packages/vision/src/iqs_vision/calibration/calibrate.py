"""Per-image score records and calibration results for one deployed detector (step 2b).

Scores are computed once per model (scripts/calibrate.py) and saved next to it as
scores.json; thresholds, the repeated-split analysis and the API all work from these records,
so changing alpha or delta never needs the images or the model again.

alpha policy (owner decision D1, 2026-11-20): 0.02 where n >= 49 calibration images, else 0.05
(n >= 19), else 0.10. delta = 0.1 for the (alpha, delta) threshold (D2), reported as
"infeasible" where even k = n does not reach it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqs_vision.calibration.beta_bound import (
    InfeasibleAlphaDeltaError,
    alpha_delta_threshold,
    clopper_pearson,
    predicted_false_alarm,
)
from iqs_vision.calibration.conformal import (
    InsufficientCalibrationError,
    min_calibration_size,
    split_conformal_threshold,
)
from iqs_vision.data.splits import Protocol, SplitLeakageError, SplitManifest

ALPHA_LEVELS = (0.02, 0.05, 0.10)
DEFAULT_DELTA = 0.1
SCORES_FILE = "scores.json"
CALIBRATION_FILE = "calibration.json"
SCORES_VERSION = 1


def policy_alpha(n: int) -> float:
    """Smallest alpha of ALPHA_LEVELS with a finite marginal threshold for n images (D1)."""
    for alpha in ALPHA_LEVELS:
        if n >= min_calibration_size(alpha):
            return alpha
    raise InsufficientCalibrationError(
        f"n = {n} calibration images: no finite threshold even at alpha = {ALPHA_LEVELS[-1]}"
    )


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    path: str
    sha256: str
    split: str
    label: int
    defect_type: str
    score: float
    source: str | None = None
    condition: str | None = None


def check_calibration_disjoint(manifest: SplitManifest, protocol: Protocol) -> None:
    """Refuse when a calibration image was used to fit the detector (by path or content)."""
    fit = manifest.fit_entries(protocol)
    calib = manifest.select("calib")
    fit_paths, fit_hashes = {e.path for e in fit}, {e.sha256 for e in fit}
    overlap = [e.path for e in calib if e.path in fit_paths or e.sha256 in fit_hashes]
    if overlap:
        raise SplitLeakageError(
            f"{manifest.category}/{protocol}: {len(overlap)} calibration images were fitted "
            f"(e.g. {overlap[0]}); protocol {protocol!r} cannot be calibrated"
        )


def save_scores(path: Path, meta: dict[str, Any], records: Sequence[ScoreRecord]) -> None:
    body = {"version": SCORES_VERSION, **meta, "records": [asdict(r) for r in records]}
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_scores(path: Path) -> tuple[dict[str, Any], list[ScoreRecord]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("version") != SCORES_VERSION:
        raise ValueError(f"{path}: unsupported scores version {body.get('version')}")
    records = [ScoreRecord(**r) for r in body.pop("records")]
    return body, records


def select(
    records: Iterable[ScoreRecord],
    split: str,
    *,
    label: int | None = None,
    source: str | None = "any",
) -> np.ndarray:
    """Scores of one split (optionally one label; source "any", None or "validation")."""
    return np.array(
        [
            r.score
            for r in records
            if r.split == split
            and (label is None or r.label == label)
            and (source == "any" or r.source == source)
        ],
        dtype=np.float64,
    )


def _rates(good: np.ndarray, defect: np.ndarray, q: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if good.size:
        x = int(np.sum(good > q))
        lo, hi = clopper_pearson(x, int(good.size))
        out |= {"n_good": int(good.size), "false_alarms": x, "false_alarm_rate": x / good.size}
        out["false_alarm_ci95"] = [lo, hi]
    if defect.size:
        missed = int(np.sum(defect <= q))
        out |= {"n_defect": int(defect.size), "misses": missed, "miss_rate": missed / defect.size}
    return out


def _level(
    calib: np.ndarray, alpha: float, delta: float, good: np.ndarray, bad: np.ndarray
) -> dict[str, Any]:
    """Marginal + (alpha, delta) thresholds at one alpha, with test rates; None if infeasible."""
    try:
        marginal = split_conformal_threshold(calib, alpha)
    except InsufficientCalibrationError as exc:
        return {"alpha": alpha, "feasible": False, "reason": str(exc)}
    level: dict[str, Any] = {
        "alpha": alpha,
        "feasible": True,
        "marginal": marginal.to_dict() | {"test": _rates(good, bad, marginal.threshold)},
    }
    if good.size:
        level["marginal"]["predicted_false_alarm"] = asdict(
            predicted_false_alarm(marginal.n, marginal.k, int(good.size))
        )
    try:
        ad = alpha_delta_threshold(calib, alpha, delta)
        level["alpha_delta"] = ad.to_dict() | {"test": _rates(good, bad, ad.threshold)}
    except InfeasibleAlphaDeltaError as exc:
        level["alpha_delta"] = {"feasible": False, "delta": delta, "reason": str(exc)}
    return level


def build_calibration(
    records: Sequence[ScoreRecord],
    meta: dict[str, Any],
    *,
    alpha: float | None = None,
    delta: float = DEFAULT_DELTA,
) -> dict[str, Any]:
    """calibration.json content: the active threshold (policy or given alpha) + all levels."""
    calib = select(records, "calib")
    good, bad = select(records, "test", label=0), select(records, "test", label=1)
    active_alpha = policy_alpha(calib.size) if alpha is None else alpha
    active = split_conformal_threshold(calib, active_alpha)  # raises if n is too small
    result: dict[str, Any] = {
        "dataset": meta["dataset"],
        "category": meta["category"],
        "protocol": meta["protocol"],
        "model_run_id": meta.get("model_run_id"),
        "manifest_fingerprint": meta["manifest_fingerprint"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "alpha_policy": "D1: 0.02 if n >= 49, else 0.05 if n >= 19, else 0.10"
        if alpha is None
        else "given",
        "delta": delta,
        "n_calib": int(calib.size),
        "threshold": active.threshold,
        "active": active.to_dict(),
        "levels": [_level(calib, a, delta, good, bad) for a in ALPHA_LEVELS],
    }
    val = select(records, "calib", source="validation")
    if val.size:
        result["validation_only"] = {
            "n_calib": int(val.size),
            "levels": [_level(val, a, delta, good, bad) for a in ALPHA_LEVELS],
        }
    return result
