"""Distribution-shift experiment helpers (step 2 gate 4, blueprint Rev 2 §2.8).

The in-distribution threshold q_hat is applied unchanged to images taken under another
condition. Exchangeability (A1) does not hold there, so the false-alarm guarantee is not
expected to hold; we measure how far the realised rate moves from alpha.

Two sources:
- real: MVTec AD 2 test_public lighting conditions (overexposed, underexposed, shift_1..4),
  from the saved scores of the shift split; no rescoring needed;
- synthetic: deterministic photometric transforms of MVTec AD good test images.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from iqs_vision.calibration.beta_bound import clopper_pearson
from iqs_vision.calibration.calibrate import ScoreRecord
from iqs_vision.evaluation.metrics import image_auroc

Transform = Callable[[Image.Image], Image.Image]


def _channel_gain(r: float, b: float) -> Transform:
    def apply(img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        arr[..., 0] *= r
        arr[..., 2] *= b
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return apply


def _enhance(kind: Any, factor: float) -> Transform:  # an ImageEnhance class
    def apply(img: Image.Image) -> Image.Image:
        return kind(img.convert("RGB")).enhance(factor)

    return apply


SYNTHETIC_SHIFTS: dict[str, Transform] = {
    "brightness_0.7": _enhance(ImageEnhance.Brightness, 0.7),
    "brightness_1.3": _enhance(ImageEnhance.Brightness, 1.3),
    "contrast_0.7": _enhance(ImageEnhance.Contrast, 0.7),
    "contrast_1.3": _enhance(ImageEnhance.Contrast, 1.3),
    "warm": _channel_gain(1.1, 0.9),  # colour temperature: more red, less blue
    "cool": _channel_gain(0.9, 1.1),
}


def false_alarm_summary(good: np.ndarray, threshold: float) -> dict[str, Any]:
    good = np.asarray(good, dtype=np.float64)
    x, m = int(np.sum(good > threshold)), int(good.size)
    lo, hi = clopper_pearson(x, m)
    return {
        "n_good": m,
        "false_alarms": x,
        "false_alarm_rate": x / m,
        "false_alarm_ci95": [lo, hi],
        "median_good_score_over_threshold": float(np.median(good) / threshold),
    }


def condition_table(records: Sequence[ScoreRecord], threshold: float) -> list[dict[str, Any]]:
    """Per lighting condition (regular from the test split, others from shift): rates + AUROC."""
    rows = []
    conds = sorted(
        {r.condition for r in records if r.split in ("test", "shift") and r.condition},
        key=lambda c: (c != "regular", c),
    )
    for cond in conds:
        rs = [r for r in records if r.split in ("test", "shift") and r.condition == cond]
        good = np.array([r.score for r in rs if r.label == 0])
        bad = np.array([r.score for r in rs if r.label == 1])
        row: dict[str, Any] = {"condition": cond}
        if good.size:
            row |= false_alarm_summary(good, threshold)
        if bad.size:
            row["n_defect"] = int(bad.size)
            row["miss_rate"] = float(np.mean(bad <= threshold))
        if good.size and bad.size:
            labels = np.r_[np.zeros(good.size), np.ones(bad.size)]
            row["image_auroc"] = image_auroc(labels, np.r_[good, bad])
        rows.append(row)
    return rows


def all_conditions_auroc(records: Sequence[ScoreRecord]) -> float:
    rs = [r for r in records if r.split in ("test", "shift")]
    return image_auroc(np.array([r.label for r in rs]), np.array([r.score for r in rs]))
