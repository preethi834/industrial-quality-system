"""Sensor and quality physics: pure functions, no SimPy, no randomness of their own.

Every function takes its noise values as arguments (drawn by the caller from a
named stream, always the same number per call), so a fault that changes one input
leaves every unrelated output bit-identical.

Signal graph (``->`` = "is computed from"):

- power_commanded  -> setpoints
- power_actual     -> power_commanded, drive deviation
- spindle_load     -> setpoints, true hardness, tool wear, overload factor
- temperature      -> ambient, spindle_load, coolant loss
- vibration        -> tool wear
- tool_wear        -> wear rate (x accelerated-wear multiplier)
- ambient          -> time of day, hall excursion
- measured dim.    -> true dimension (fixture offset, handling error), gauge drift
- surface defect   -> logistic in excess wear, load, temperature, power residual,
                      hardness deviation and out-of-window parameters (hidden truth)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from iqs_sim.machines import NOMINAL_HARDNESS_HB, MachineSpec

# Noise standard deviations (observable sensor noise).
SD_POWER_KW = 0.08
SD_LOAD_PCT = 1.5
SD_TEMP_C = 0.6
SD_VIBRATION = 0.08
SD_AMBIENT_C = 0.15
SD_WEAR_REL = 0.15
N_CUT_NOISE = 5  # power, load, temperature, vibration, ambient

# Quality model thresholds and weights (hidden truth, documented in docs/notes).
WEAR_FRACTION_ONSET = 0.8
LOAD_THRESHOLD_PCT = 60.0
TEMP_THRESHOLD_C = 46.0
POWER_RESIDUAL_FREE = 0.04
HARDNESS_FREE_HB = 15.0
W_WEAR, W_LOAD, W_TEMP, W_POWER, W_HARD, W_PARAM = 6.0, 0.12, 0.35, 10.0, 0.06, 1.5

PLANNED_WEAR_AT_CHANGE = 0.6  # a normal tool reaches 60 % of its limit at the scheduled change


@dataclass(frozen=True)
class CutInputs:
    speed_rpm: float
    feed_mm_rev: float
    hardness_hb: float
    wear_mm: float
    ambient_c: float
    overload_factor: float = 1.0
    coolant_loss: float = 0.0
    drive_deviation: float = 0.0


@dataclass(frozen=True)
class CutReading:
    power_commanded_kw: float
    power_actual_kw: float
    spindle_load_pct: float
    temperature_c: float
    vibration_mm_s: float


def ambient_temperature(hour_of_day: float, excursion_c: float, noise: float) -> float:
    """Hall temperature: daily cycle (peak 15:00) around 21 degC, plus excursion and noise."""
    daily = 21.0 + 1.5 * math.sin(2 * math.pi * (hour_of_day - 9.0) / 24.0)
    return daily + excursion_c + SD_AMBIENT_C * noise


def wear_increment(spec: MachineSpec, multiplier: float, noise: float) -> float:
    assert spec.tool_wear_limit_mm is not None and spec.tool_life_parts is not None
    base = spec.tool_wear_limit_mm * PLANNED_WEAR_AT_CHANGE / spec.tool_life_parts
    return max(0.0, base * multiplier * (1.0 + SD_WEAR_REL * noise))


def cut_reading(spec: MachineSpec, x: CutInputs, noise: tuple[float, ...]) -> CutReading:
    """Sensor values for one cut. ``noise`` holds standard-normal draws (power, load, temp, vib)."""
    assert spec.nominal_speed_rpm is not None and spec.nominal_feed_mm_rev is not None
    assert spec.nominal_power_kw is not None and spec.nominal_load_pct is not None
    assert spec.tool_wear_limit_mm is not None
    speed_ratio = x.speed_rpm / spec.nominal_speed_rpm
    feed_ratio = x.feed_mm_rev / spec.nominal_feed_mm_rev
    wear_frac = x.wear_mm / spec.tool_wear_limit_mm

    p_cmd = spec.nominal_power_kw * speed_ratio * feed_ratio
    p_act = p_cmd * (1.0 + x.drive_deviation) + SD_POWER_KW * noise[0]
    load = (
        spec.nominal_load_pct
        * speed_ratio**0.6
        * feed_ratio**0.8
        * (x.hardness_hb / NOMINAL_HARDNESS_HB)
        * (1.0 + 0.3 * wear_frac)
        * x.overload_factor
        + SD_LOAD_PCT * noise[1]
    )
    heat = (6.0 + 0.3 * load) * (1.0 + 1.5 * x.coolant_loss)
    temp = x.ambient_c + heat + SD_TEMP_C * noise[2]
    vib = 1.0 + 2.5 * wear_frac**2 + SD_VIBRATION * noise[3]
    return CutReading(p_cmd, p_act, load, temp, vib)


def cut_defect_logit(
    spec: MachineSpec, x: CutInputs, r: CutReading, *, out_of_window: bool
) -> float:
    """Contribution of one cut to the part's surface-defect logit (0 under normal conditions)."""
    assert spec.tool_wear_limit_mm is not None
    wear_frac = x.wear_mm / spec.tool_wear_limit_mm
    residual = abs(r.power_actual_kw - r.power_commanded_kw) / r.power_commanded_kw
    return (
        W_WEAR * max(0.0, wear_frac - WEAR_FRACTION_ONSET)
        + W_LOAD * max(0.0, r.spindle_load_pct - LOAD_THRESHOLD_PCT)
        + W_TEMP * max(0.0, r.temperature_c - TEMP_THRESHOLD_C)
        + W_POWER * max(0.0, residual - POWER_RESIDUAL_FREE)
        + (W_PARAM if out_of_window else 0.0)
    )


def hardness_defect_logit(hardness_hb: float) -> float:
    return W_HARD * max(0.0, abs(hardness_hb - NOMINAL_HARDNESS_HB) - HARDNESS_FREE_HB)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))
