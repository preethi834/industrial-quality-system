"""Shift experiment helpers: deterministic transforms, per-condition table."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from iqs_vision.calibration.calibrate import ScoreRecord
from iqs_vision.calibration.shift import (
    SYNTHETIC_SHIFTS,
    all_conditions_auroc,
    condition_table,
    false_alarm_summary,
)


def _img(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(40, 200, (24, 32, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(SYNTHETIC_SHIFTS))
def test_transforms_are_deterministic_and_keep_shape(name: str) -> None:
    t = SYNTHETIC_SHIFTS[name]
    a, b = np.asarray(t(_img())), np.asarray(t(_img()))
    assert a.shape == (24, 32, 3) and np.array_equal(a, b)
    assert not np.array_equal(a, np.asarray(_img()))


def test_transforms_change_the_intended_property() -> None:
    base = np.asarray(_img(), dtype=float)
    assert np.asarray(SYNTHETIC_SHIFTS["brightness_0.7"](_img())).mean() < base.mean()
    assert np.asarray(SYNTHETIC_SHIFTS["contrast_1.3"](_img())).std() > base.std()
    warm = np.asarray(SYNTHETIC_SHIFTS["warm"](_img()), dtype=float)
    assert warm[..., 0].mean() > base[..., 0].mean() and warm[..., 2].mean() < base[..., 2].mean()
    gray = Image.fromarray(np.full((8, 8), 100, np.uint8))
    assert np.asarray(SYNTHETIC_SHIFTS["cool"](gray)).shape == (8, 8, 3)


def test_false_alarm_summary() -> None:
    s = false_alarm_summary(np.array([0.5, 1.5, 2.5, 0.2]), 1.0)
    assert (s["false_alarms"], s["n_good"], s["false_alarm_rate"]) == (2, 4, 0.5)
    assert s["false_alarm_ci95"][0] < 0.5 < s["false_alarm_ci95"][1]
    assert s["median_good_score_over_threshold"] == pytest.approx(1.0)


def test_condition_table_orders_regular_first_and_uses_both_splits() -> None:
    recs = []
    for cond, split, shift in (("regular", "test", 0.0), ("overexposed", "shift", 2.0)):
        for i in range(5):
            recs.append(
                ScoreRecord(f"{cond}g{i}", "0", split, 0, "good", i * 0.1 + shift, condition=cond)
            )
            recs.append(ScoreRecord(f"{cond}b{i}", "0", split, 1, "bad", 3.0 + i, condition=cond))
    recs.append(ScoreRecord("c0", "0", "calib", 0, "good", 0.0))
    rows = condition_table(recs, threshold=1.0)
    assert [r["condition"] for r in rows] == ["regular", "overexposed"]
    assert rows[0]["false_alarm_rate"] == 0.0 and rows[1]["false_alarm_rate"] == 1.0
    assert rows[0]["image_auroc"] == 1.0 and rows[1]["miss_rate"] == 0.0
    assert 0.5 < all_conditions_auroc(recs) <= 1.0
