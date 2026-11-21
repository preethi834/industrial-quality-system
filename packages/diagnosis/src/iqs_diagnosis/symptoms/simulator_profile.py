"""Symptom profile of a simulator case from its observable data.

Two periods are compared:

- **reference**: the first ``baseline_hours`` of production (the period the alarm rule
  used as its baseline), and
- **affected**: from the start of the alarm window to the end of the data.

Every feature is a descriptive effect size (shift in standard deviations, ratio,
share or lift). No p-values and no decisions about causes are made here: that is the
scientist agent's job (step 5b). Features are aggregated over machines / batches /
shifts (maximum and, for temperature, minimum), so the vector does not depend on
*which* machine is affected; the location is kept as evidence (``where``).

Turning features into discrete symptom codes is done by the vocabulary
(``configs/diagnosis/symptoms.yaml``), not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Final

import numpy as np
import pandas as pd

from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.domain.case import DiagnosisCase

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "reject_rate_log2_ratio",
    "reject_share_dimension",
    "reject_share_surface",
    "reject_lift_s2_machine",
    "reject_lift_batch",
    "reject_lift_shift",
    "dim_mean_shift_z",
    "dim_sd_log2_ratio",
    "dim_shift_s2_gap_z",
    "gauge_check_shift_um",
    "temp_z_max",
    "temp_z_min",
    "load_z_max",
    "load_batch_z_max",
    "vibration_z_max",
    "wear_rate_log2_ratio_max",
    "power_residual_z_max",
    "setpoint_changed_share_max",
    "ambient_shift_c",
)

# Minimum group sizes for rates and lifts: smaller groups are too noisy to report.
MIN_GROUP_PARTS: Final = 30
MIN_REJECTS: Final = 3
# Floor for the reference reject rate (the line's known long-run rate, as in iqs_sim).
MIN_REFERENCE_RATE: Final = 0.02
# Floor for reference standard deviations, so a near-constant signal cannot explode z.
SD_FLOOR: Final[dict[str, float]] = {
    "temperature_c": 0.1,
    "spindle_load_pct": 0.1,
    "vibration_mm_s": 0.01,
    "power_residual": 0.001,
    "measured_dimension_mm": 0.001,
}


@dataclass
class ProfileFeatures:
    values: dict[str, float]
    where: dict[str, str] = field(default_factory=dict)


def _sd(x: pd.Series, name: str) -> float:
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return max(sd if math.isfinite(sd) else 0.0, SD_FLOOR.get(name, 1e-9))


def _log2_lift(rate_group: float, rate_all: float) -> float:
    return math.log2(max(rate_group, 1e-3) / max(rate_all, 1e-3))


def _max_lift(aff: pd.DataFrame, column: str) -> tuple[float, str | None]:
    """Largest log2 lift of the reject rate of one group over the whole affected period."""
    rate_all = float(aff["is_reject"].mean()) if len(aff) else 0.0
    best, where = 0.0, None
    if rate_all == 0.0 or int(aff["is_reject"].sum()) < MIN_REJECTS:
        return best, where
    grouped = aff.groupby(column, observed=True)["is_reject"].agg(["mean", "size"])
    for key, row in grouped.iterrows():
        if row["size"] < MIN_GROUP_PARTS:
            continue
        lift = _log2_lift(float(row["mean"]), rate_all)
        if lift > best:
            best, where = lift, str(key)
    return best, where


def _machine_shift(ref: pd.DataFrame, aff: pd.DataFrame, column: str) -> dict[str, float]:
    """Per machine: (affected mean - reference mean) / reference sd."""
    out: dict[str, float] = {}
    for machine, r in ref.groupby("machine_id"):
        a = aff[aff["machine_id"] == machine]
        if len(r) < MIN_GROUP_PARTS or len(a) < MIN_GROUP_PARTS:
            continue
        out[str(machine)] = float((a[column].mean() - r[column].mean()) / _sd(r[column], column))
    return out


def _argmax(d: dict[str, float], absolute: bool = False) -> tuple[float, str | None]:
    if not d:
        return 0.0, None
    key = max(d, key=lambda k: (abs(d[k]) if absolute else d[k], k))
    return (abs(d[key]) if absolute else d[key]), key


def _wear_rate(readings: pd.DataFrame) -> pd.Series:
    """Median wear increment per cut, per machine (within one tool, so tool changes drop out)."""
    r = readings.sort_values("ts")
    inc = r.groupby("tool_id")["tool_wear_mm"].diff()
    r = r.assign(inc=inc)
    r = r[r["inc"] > 0]
    return r.groupby("machine_id")["inc"].median()


def compute_features(case: DiagnosisCase, store: ProcessStore) -> ProfileFeatures:
    if case.window_start is None or case.baseline_hours is None:
        raise ValueError(f"{case.case_id}: simulator profile needs window_start and baseline_hours")
    parts = store.table("parts")
    readings = store.table("sensor_readings")
    events = store.table("station_events")
    shifts = store.table("shifts")
    machines = store.table("machines")

    t0 = parts["completed_ts"].min().floor("h")
    ref_end = t0 + timedelta(hours=case.baseline_hours)
    win = pd.Timestamp(case.window_start)

    parts = parts.assign(
        is_reject=(parts["final_quality"] == "reject").astype(float),
        shift_code=parts["shift_id"].map(
            dict(zip(shifts["shift_id"], shifts["shift_code"], strict=True))
        ),
    )
    p_ref = parts[parts["completed_ts"] < ref_end]
    p_aff = parts[parts["completed_ts"] >= win]
    v: dict[str, float] = {}
    where: dict[str, str] = {}

    # --- reject rate and reasons -------------------------------------------------------
    rate_ref = max(float(p_ref["is_reject"].mean()) if len(p_ref) else 0.0, MIN_REFERENCE_RATE)
    rate_aff = float(p_aff["is_reject"].mean()) if len(p_aff) else 0.0
    v["reject_rate_log2_ratio"] = math.log2(max(rate_aff, 1e-3) / rate_ref)
    reasons = p_aff.loc[p_aff["is_reject"] > 0, "reject_reason"].fillna("")
    n_rej = len(reasons)
    v["reject_share_dimension"] = float(reasons.str.contains("dimension").mean()) if n_rej else 0.0
    v["reject_share_surface"] = float(reasons.str.contains("surface").mean()) if n_rej else 0.0

    # --- where rejects concentrate (descriptive lifts) ------------------------------------
    for feature, column in (
        ("reject_lift_s2_machine", "s2_machine_id"),
        ("reject_lift_batch", "batch_id"),
        ("reject_lift_shift", "shift_code"),
    ):
        v[feature], w = _max_lift(p_aff, column)
        if w is not None:
            where[feature] = w

    # --- measured dimension and gauge checks --------------------------------------------
    dim = "measured_dimension_mm"
    sd_dim = _sd(p_ref[dim], dim)
    v["dim_mean_shift_z"] = abs(float(p_aff[dim].mean() - p_ref[dim].mean())) / sd_dim
    sd_aff = float(p_aff[dim].std(ddof=1)) if len(p_aff) > 1 else sd_dim
    v["dim_sd_log2_ratio"] = math.log2(max(sd_aff, 1e-9) / sd_dim)
    s2_shift = {
        str(m): float(
            p_aff.loc[p_aff["s2_machine_id"] == m, dim].mean()
            - p_ref.loc[p_ref["s2_machine_id"] == m, dim].mean()
        )
        / sd_dim
        for m in sorted(p_ref["s2_machine_id"].unique())
        if (p_aff["s2_machine_id"] == m).sum() >= MIN_GROUP_PARTS
    }
    if len(s2_shift) >= 2:
        vals = sorted(s2_shift.values())
        v["dim_shift_s2_gap_z"] = vals[-1] - vals[0]
        where["dim_shift_s2_gap_z"] = max(s2_shift, key=lambda k: (abs(s2_shift[k]), k))
    else:
        v["dim_shift_s2_gap_z"] = 0.0
    checks = events[events["event_type"] == "gauge_check"]
    c_ref = checks.loc[checks["ts"] < ref_end, "value"]
    c_aff = checks.loc[checks["ts"] >= win, "value"]
    v["gauge_check_shift_um"] = (
        abs(float(c_aff.mean() - c_ref.mean())) * 1000.0 if len(c_ref) and len(c_aff) else 0.0
    )

    # --- machine sensors --------------------------------------------------------------
    readings = readings.assign(
        power_residual=readings["power_actual_kw"] / readings["power_commanded_kw"] - 1.0
    )
    r_ref = readings[readings["ts"] < ref_end]
    r_aff = readings[readings["ts"] >= win]

    temp = _machine_shift(r_ref, r_aff, "temperature_c")
    v["temp_z_max"], w = _argmax(temp)
    if w:
        where["temp_z_max"] = w
    v["temp_z_min"] = min(temp.values()) if temp else 0.0
    for feature, column, absolute in (
        ("load_z_max", "spindle_load_pct", False),
        ("vibration_z_max", "vibration_mm_s", False),
        ("power_residual_z_max", "power_residual", True),
    ):
        v[feature], w = _argmax(_machine_shift(r_ref, r_aff, column), absolute=absolute)
        if w:
            where[feature] = w

    # Spindle load by batch (all machines), standardised per machine on the reference.
    stats = r_ref.groupby("machine_id")["spindle_load_pct"].agg(["mean", "std"])
    load_z = (r_aff["spindle_load_pct"] - r_aff["machine_id"].map(stats["mean"])) / r_aff[
        "machine_id"
    ].map(stats["std"].clip(lower=SD_FLOOR["spindle_load_pct"]))
    by_batch = (
        pd.DataFrame({"part_id": r_aff["part_id"], "z": load_z})
        .merge(parts[["part_id", "batch_id"]], on="part_id")
        .groupby("batch_id")["z"]
        .agg(["mean", "size"])
    )
    by_batch = by_batch[by_batch["size"] >= 3 * MIN_GROUP_PARTS]
    if len(by_batch):
        v["load_batch_z_max"] = float(by_batch["mean"].max())
        where["load_batch_z_max"] = str(by_batch["mean"].idxmax())
    else:
        v["load_batch_z_max"] = 0.0

    wr_ref, wr_aff = _wear_rate(r_ref), _wear_rate(r_aff)
    wear = {
        str(m): math.log2(float(wr_aff[m]) / float(wr_ref[m]))
        for m in wr_ref.index
        if m in wr_aff.index and wr_ref[m] > 0 and wr_aff[m] > 0
    }
    v["wear_rate_log2_ratio_max"], w = _argmax(wear)
    if w:
        where["wear_rate_log2_ratio_max"] = w

    # Setpoints: share of affected cuts with a parameter set not used in the reference
    # period or a speed outside the validated window (master data in ``machines``).
    win_tbl = machines.set_index("machine_id")[["speed_window_low_rpm", "speed_window_high_rpm"]]
    ref_sets = set(r_ref["parameter_set_id"].unique())
    lo = r_aff["machine_id"].map(win_tbl["speed_window_low_rpm"])
    hi = r_aff["machine_id"].map(win_tbl["speed_window_high_rpm"])
    changed = (~r_aff["parameter_set_id"].isin(ref_sets)) | (
        (r_aff["setpoint_speed_rpm"] < lo) | (r_aff["setpoint_speed_rpm"] > hi)
    )
    share = changed.groupby(r_aff["machine_id"]).mean()
    share_d = {str(k): float(x) for k, x in share.items()}
    v["setpoint_changed_share_max"], w = _argmax(share_d)
    if w and v["setpoint_changed_share_max"] > 0:
        where["setpoint_changed_share_max"] = w

    v["ambient_shift_c"] = (
        float(r_aff["ambient_temperature_c"].mean() - r_ref["ambient_temperature_c"].mean())
        if len(r_ref) and len(r_aff)
        else 0.0
    )

    clean = {k: (float(v[k]) if np.isfinite(v[k]) else 0.0) for k in FEATURE_NAMES}
    return ProfileFeatures(values=clean, where=where)
