"""Statistical test templates: one per taxonomy cause (``test_template`` in causes.yaml).

Each template runs fixed, parameterised SQL on the read-only process store (never
free-form SQL), applies a pre-registered test and returns a ``TemplateResult``:
one-sided p-value (two-sided for the gauge trend), effect size, group sizes, the SQL and
its parameters, and a plotly-compatible evidence figure.

Periods (same as the symptom profile): **reference** = first ``baseline_hours`` of
production; **affected** = from the alarm-window start to the end of the data.

Templates that scan several targets (machines, batches, shifts) report
``p = min(1, m * min_i p_i)`` (Bonferroni over the m targets with enough data), so each
template's p-value is valid on its own; the runner then corrects across hypotheses with
Benjamini-Hochberg. A template without enough data returns ``p_value=None``
(inconclusive) and says why.

Specificity choices (validity is unaffected; they only change what else can trigger a
template): the batch template compares a batch with its neighbouring batches, so faults
that simply start mid-run do not look like a bad batch; the cooling template uses
temperature minus ambient, so a hot hall does not look like a cooling loss.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.scientist.plots import bar_figure, box_figure, trend_figure

MIN_GROUP: Final = 30  # parts per group for rate comparisons and for "passed" samples
MIN_REJECTS: Final = 5  # rejected parts needed to compare a signal
MIN_TOTAL_REJECTS: Final = 3  # rejects needed in a 2x2 table
MIN_CHECKS: Final = 8  # gauge checks for a trend
WEAR_HIGH: Final = 0.8  # fraction of the wear limit (the quality model's onset of wear defects)


@dataclass(frozen=True)
class Window:
    reference_end: pd.Timestamp
    affected_start: pd.Timestamp


class TemplateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    template: str
    test: str
    target: str | None = None  # machine / batch / shift with the smallest p
    p_value: float | None  # None = inconclusive
    p_min_raw: float | None = None
    n_targets: int = 0
    statistic: float | None = None
    effect: dict[str, float] = Field(default_factory=dict)
    groups: dict[str, int] = Field(default_factory=dict)
    sql: str
    params: list[str] = Field(default_factory=list)
    reason: str | None = None  # why inconclusive
    plot: dict[str, Any] | None = None


Template = Callable[[ProcessStore, Window], TemplateResult]

# ----------------------------------------------------------------------------- SQL

SQL_CUTS: Final = """
SELECT r.machine_id, r.ts, r.tool_wear_mm, r.spindle_load_pct, r.temperature_c,
       r.ambient_temperature_c, r.power_actual_kw, r.power_commanded_kw,
       r.parameter_set_id, r.setpoint_speed_rpm,
       p.part_id, p.completed_ts, (p.final_quality = 'reject') AS is_reject, p.reject_reason
FROM read_parquet(?) AS r JOIN read_parquet(?) AS p USING (part_id)
WHERE p.completed_ts >= ?
"""
SQL_PARTS: Final = """
SELECT p.part_id, p.batch_id, p.completed_ts, p.s1_machine_id, p.s2_machine_id,
       p.s3_machine_id, p.measured_dimension_mm, (p.final_quality = 'reject') AS is_reject,
       p.reject_reason, s.shift_code
FROM read_parquet(?) AS p JOIN read_parquet(?) AS s USING (shift_id)
"""
SQL_MACHINES: Final = """
SELECT machine_id, tool_wear_limit_mm, speed_window_low_rpm, speed_window_high_rpm
FROM read_parquet(?) WHERE kind = 'cutting'
"""
SQL_REFERENCE_SETS: Final = """
SELECT DISTINCT r.machine_id, r.parameter_set_id
FROM read_parquet(?) AS r JOIN read_parquet(?) AS p USING (part_id)
WHERE p.completed_ts < ?
"""
SQL_GAUGE: Final = """
SELECT ts, value FROM read_parquet(?) WHERE event_type = 'gauge_check' ORDER BY ts
"""


def _path(store: ProcessStore, table: str) -> str:
    return str(store.observed_dir / f"{table}.parquet")


RUN_START: Final = pd.Timestamp("1970-01-01", tz="UTC")


def _cuts(
    store: ProcessStore, w: Window, *, since: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Cuts of parts completed from ``since`` (default: the affected-period start)."""
    start = w.affected_start if since is None else since
    params = [_path(store, "sensor_readings"), _path(store, "parts"), str(start)]
    df = store.query(SQL_CUTS, [params[0], params[1], start.to_pydatetime()])
    return df, params


def _parts(store: ProcessStore) -> tuple[pd.DataFrame, list[str]]:
    params = [_path(store, "parts"), _path(store, "shifts")]
    return store.query(SQL_PARTS, list(params)), params


def _inconclusive(
    template: str, test: str, sql: str, params: list[str], reason: str
) -> TemplateResult:
    return TemplateResult(
        template=template, test=test, p_value=None, sql=sql.strip(), params=params, reason=reason
    )


def _bonferroni(ps: dict[str, float]) -> tuple[float, str, float]:
    target = min(ps, key=lambda k: (ps[k], k))
    return min(1.0, len(ps) * ps[target]), target, ps[target]


def _fisher_greater(a_rej: int, a_n: int, b_rej: int, b_n: int) -> float:
    table = [[a_rej, a_n - a_rej], [b_rej, b_n - b_rej]]
    return float(stats.fisher_exact(table, alternative="greater").pvalue)


def _mwu_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r = stats.mannwhitneyu(x, y, alternative="greater")
    return float(r.pvalue), float(r.statistic)


# ----------------------------------------------------- per-part signal: rejected vs passed


def _signal_template(
    name: str,
    signal: Callable[[pd.DataFrame], pd.Series],
    label: str,
    *,
    by_machine: bool = True,
) -> Template:
    """Rejected vs passed parts (affected period): signal per part (mean over its cuts)."""

    def run(store: ProcessStore, w: Window) -> TemplateResult:
        test = "mann_whitney_u_greater"
        cuts, params = _cuts(store, w)
        if cuts.empty:
            return _inconclusive(name, test, SQL_CUTS, params, "no cuts in the affected period")
        cuts = cuts.assign(signal=signal(cuts))
        groups = cuts.groupby("machine_id") if by_machine else [("HALL", cuts)]
        ps: dict[str, float] = {}
        detail: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for target, g in groups:
            per_part = g.groupby("part_id").agg(signal=("signal", "mean"), rej=("is_reject", "max"))
            x = per_part.loc[per_part["rej"], "signal"].to_numpy(dtype=float)
            y = per_part.loc[~per_part["rej"], "signal"].to_numpy(dtype=float)
            if len(x) < MIN_REJECTS or len(y) < MIN_GROUP:
                continue
            p, u = _mwu_greater(x, y)
            ps[str(target)], detail[str(target)] = p, (x, y, u)
        if not ps:
            return _inconclusive(
                name,
                test,
                SQL_CUTS,
                params,
                f"fewer than {MIN_REJECTS} rejected or {MIN_GROUP} passed parts per target",
            )
        p_adj, target, p_raw = _bonferroni(ps)
        x, y, u = detail[target]
        return TemplateResult(
            template=name,
            test=test,
            target=target,
            p_value=p_adj,
            p_min_raw=p_raw,
            n_targets=len(ps),
            statistic=u,
            effect={
                "median_rejected": float(np.median(x)),
                "median_passed": float(np.median(y)),
                "median_difference": float(np.median(x) - np.median(y)),
                "rank_biserial": float(2 * u / (len(x) * len(y)) - 1),
            },
            groups={"rejected": len(x), "passed": len(y)},
            sql=SQL_CUTS.strip(),
            params=params,
            plot=box_figure(f"{label} at {target}: rejected vs passed parts", label, x, y),
        )

    return run


spindle_load_defective_vs_good = _signal_template(
    "spindle_load_defective_vs_good", lambda c: c["spindle_load_pct"], "spindle load (%)"
)
temperature_delta_defective_vs_good = _signal_template(
    "temperature_delta_defective_vs_good",
    lambda c: c["temperature_c"] - c["ambient_temperature_c"],
    "machine temperature minus ambient (°C)",
)
# ----------------------------------------------------------- rate comparisons (Fisher)


def _rate_result(
    name: str,
    sql: str,
    params: list[str],
    tables: dict[str, tuple[int, int, int, int]],
    group_labels: tuple[str, str],
    title: str,
) -> TemplateResult:
    """``tables``: target -> (rejects in group, parts in group, rejects in rest, parts in rest)."""
    test = "fisher_exact_greater"
    ps = {
        t: _fisher_greater(*v)
        for t, v in tables.items()
        if v[1] >= MIN_GROUP and v[3] >= MIN_GROUP and v[0] + v[2] >= MIN_TOTAL_REJECTS
    }
    if not ps:
        return _inconclusive(
            name,
            test,
            sql,
            params,
            f"no target with {MIN_GROUP}+ parts in both groups and {MIN_TOTAL_REJECTS}+ rejects",
        )
    p_adj, target, p_raw = _bonferroni(ps)
    a, n1, b, n2 = tables[target]
    r1, r2 = a / n1, b / n2
    return TemplateResult(
        template=name,
        test=test,
        target=target,
        p_value=p_adj,
        p_min_raw=p_raw,
        n_targets=len(ps),
        effect={"rate_group": r1, "rate_rest": r2, "rate_difference": r1 - r2},
        groups={"group_parts": n1, "group_rejects": a, "rest_parts": n2, "rest_rejects": b},
        sql=sql.strip(),
        params=params,
        plot=bar_figure(f"{title} ({target})", group_labels, (r1, r2), (n1, n2)),
    )


def defect_rate_by_tool_wear_level(store: ProcessStore, w: Window) -> TemplateResult:
    name = "defect_rate_by_tool_wear_level"
    cuts, params = _cuts(store, w)
    machines = store.query(SQL_MACHINES, [_path(store, "machines")]).set_index("machine_id")
    params = [*params, _path(store, "machines")]
    sql = SQL_CUTS + ";\n" + SQL_MACHINES
    tables: dict[str, tuple[int, int, int, int]] = {}
    for m, g in cuts.groupby("machine_id"):
        limit = float(machines.loc[m, "tool_wear_limit_mm"])
        high = g["tool_wear_mm"] >= WEAR_HIGH * limit
        hi, lo = g[high], g[~high]
        tables[str(m)] = (int(hi["is_reject"].sum()), len(hi), int(lo["is_reject"].sum()), len(lo))
    return _rate_result(
        name,
        sql,
        params,
        tables,
        (f"wear >= {WEAR_HIGH:.0%} of limit", "below"),
        "Reject rate by tool wear level",
    )


def defect_rate_by_parameter_set(store: ProcessStore, w: Window) -> TemplateResult:
    name = "defect_rate_by_parameter_set"
    cuts, params = _cuts(store, w, since=RUN_START)  # whole run: both groups exist
    ref_params = [_path(store, "sensor_readings"), _path(store, "parts")]
    ref = store.query(SQL_REFERENCE_SETS, [*ref_params, w.reference_end.to_pydatetime()])
    machines = store.query(SQL_MACHINES, [_path(store, "machines")]).set_index("machine_id")
    sql = SQL_CUTS + ";\n" + SQL_REFERENCE_SETS + ";\n" + SQL_MACHINES
    params = [*params, str(w.reference_end), _path(store, "machines")]
    known = set(zip(ref["machine_id"], ref["parameter_set_id"], strict=True))
    tables: dict[str, tuple[int, int, int, int]] = {}
    for m, g in cuts.groupby("machine_id"):
        lo_rpm = float(machines.loc[m, "speed_window_low_rpm"])
        hi_rpm = float(machines.loc[m, "speed_window_high_rpm"])
        changed = ~pd.Series([(m, s) in known for s in g["parameter_set_id"]], index=g.index) | ~g[
            "setpoint_speed_rpm"
        ].between(lo_rpm, hi_rpm)
        a, b = g[changed], g[~changed]
        tables[str(m)] = (int(a["is_reject"].sum()), len(a), int(b["is_reject"].sum()), len(b))
    return _rate_result(
        name,
        sql,
        params,
        tables,
        ("changed parameters", "standard"),
        "Reject rate by parameter set",
    )


def defect_rate_by_batch(store: ProcessStore, w: Window) -> TemplateResult:
    """Each batch vs its previous and its next batch (whole run).

    Intersection-union test: a batch counts only if it is worse than *both* neighbours,
    ``p_b = max(p_vs_previous, p_vs_next)`` (valid at level alpha). A fault that starts
    mid-run makes a batch worse than the previous one but not than the next one.
    """
    name = "defect_rate_by_batch"
    test = "fisher_exact_greater_intersection_union"
    parts, params = _parts(store)
    order = sorted(parts["batch_id"].unique())
    by = parts.groupby("batch_id")["is_reject"].agg(["sum", "size"])
    ps: dict[str, float] = {}
    for i in range(1, len(order) - 1):
        b, prev, nxt = order[i], order[i - 1], order[i + 1]
        nb, kb = int(by.loc[b, "size"]), int(by.loc[b, "sum"])
        pair = []
        for other in (prev, nxt):
            no, ko = int(by.loc[other, "size"]), int(by.loc[other, "sum"])
            if nb < MIN_GROUP or no < MIN_GROUP or kb + ko < MIN_TOTAL_REJECTS:
                break
            pair.append(_fisher_greater(kb, nb, ko, no))
        if len(pair) == 2:
            ps[str(b)] = max(pair)
    if not ps:
        return _inconclusive(name, test, SQL_PARTS, params, "no batch with two usable neighbours")
    p_adj, target, p_raw = _bonferroni(ps)
    i = order.index(target)
    labels = (order[i - 1], target, order[i + 1])
    rates = [float(by.loc[x, "sum"] / by.loc[x, "size"]) for x in labels]
    sizes = [int(by.loc[x, "size"]) for x in labels]
    return TemplateResult(
        template=name,
        test=test,
        target=target,
        p_value=p_adj,
        p_min_raw=p_raw,
        n_targets=len(ps),
        effect={
            "rate_batch": rates[1],
            "rate_previous": rates[0],
            "rate_next": rates[2],
        },
        groups={"batch_parts": sizes[1], "previous_parts": sizes[0], "next_parts": sizes[2]},
        sql=SQL_PARTS.strip(),
        params=params,
        plot=bar_figure(f"Reject rate of batch {target} and its neighbours", labels, rates, sizes),
    )


def power_command_residual_defective_vs_good(store: ProcessStore, w: Window) -> TemplateResult:
    """Per machine: reject rate of cuts whose |actual/commanded - 1| exceeds the reference
    period's 99th percentile, vs all other cuts (whole run).

    A drive fault raises the residual of every cut on its machine, so comparing rejected
    with passed parts *within* the affected period sees no difference; comparing high-
    with normal-residual cuts does.
    """
    name = "power_command_residual_defective_vs_good"
    cuts, params = _cuts(store, w, since=RUN_START)
    cuts = cuts.assign(res=(cuts["power_actual_kw"] / cuts["power_commanded_kw"] - 1.0).abs())
    ref = cuts[cuts["completed_ts"] < w.reference_end]
    tables: dict[str, tuple[int, int, int, int]] = {}
    for m, g in cuts.groupby("machine_id"):
        r = ref.loc[ref["machine_id"] == m, "res"]
        if len(r) < MIN_GROUP:
            continue
        high = g["res"] > float(r.quantile(0.99))
        hi, lo = g[high], g[~high]
        tables[str(m)] = (int(hi["is_reject"].sum()), len(hi), int(lo["is_reject"].sum()), len(lo))
    return _rate_result(
        name,
        SQL_CUTS,
        [*params, str(w.reference_end)],
        tables,
        ("residual above reference p99", "normal residual"),
        "Reject rate by power residual",
    )


def ambient_temperature_defective_vs_good(store: ProcessStore, w: Window) -> TemplateResult:
    """Reject rate of parts cut while the hall was outside its specified range vs inside
    (whole run; spec range from the observable run.json)."""
    name = "ambient_temperature_defective_vs_good"
    cuts, params = _cuts(store, w, since=RUN_START)
    spec = store.run_info()["spec"]
    assert isinstance(spec, dict)
    lo_c, hi_c = (float(x) for x in spec["ambient_temperature_range_c"])
    per_part = cuts.groupby("part_id").agg(
        amb_max=("ambient_temperature_c", "max"),
        amb_min=("ambient_temperature_c", "min"),
        rej=("is_reject", "max"),
    )
    out = (per_part["amb_max"] > hi_c) | (per_part["amb_min"] < lo_c)
    a, b = per_part[out], per_part[~out]
    tables = {"HALL": (int(a["rej"].sum()), len(a), int(b["rej"].sum()), len(b))}
    return _rate_result(
        name,
        SQL_CUTS,
        [*params, "observed/run.json"],
        tables,
        (f"ambient outside {lo_c:g}-{hi_c:g} °C", "inside"),
        "Reject rate by hall temperature",
    )


def defect_rate_by_shift(store: ProcessStore, w: Window) -> TemplateResult:
    """Dimension rejects of one shift code vs the other shifts (affected period)."""
    name = "defect_rate_by_shift"
    parts, params = _parts(store)
    aff = parts[parts["completed_ts"] >= w.affected_start]
    dim_rej = aff["is_reject"] & aff["reject_reason"].fillna("").str.contains("dimension")
    aff = aff.assign(dim_rej=dim_rej)
    tables: dict[str, tuple[int, int, int, int]] = {}
    for code, g in aff.groupby("shift_code"):
        rest = aff[aff["shift_code"] != code]
        tables[str(code)] = (int(g["dim_rej"].sum()), len(g), int(rest["dim_rej"].sum()), len(rest))
    return _rate_result(
        name,
        SQL_PARTS,
        [*params, str(w.affected_start)],
        tables,
        ("shift", "other shifts"),
        "Dimension-reject rate by shift",
    )


# --------------------------------------------------------------- before / after, trend


def station_defect_rate_change_point(store: ProcessStore, w: Window) -> TemplateResult:
    """Per machine: |dimension - nominal| after the alarm-window start vs the reference period.

    The split point is the window start given by monitoring (fixed, not searched), so the
    p-value needs no change-point correction.
    """
    name = "station_defect_rate_change_point"
    test = "mann_whitney_u_greater"
    parts, params = _parts(store)
    params = [*params, str(w.reference_end), str(w.affected_start)]
    dev = (parts["measured_dimension_mm"] - parts["measured_dimension_mm"].median()).abs()
    parts = parts.assign(dev=dev)
    ref = parts[parts["completed_ts"] < w.reference_end]
    aff = parts[parts["completed_ts"] >= w.affected_start]
    ps: dict[str, float] = {}
    detail: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for col in ("s1_machine_id", "s2_machine_id", "s3_machine_id"):
        for m in sorted(parts[col].unique()):
            x = aff.loc[aff[col] == m, "dev"].to_numpy(dtype=float)
            y = ref.loc[ref[col] == m, "dev"].to_numpy(dtype=float)
            if len(x) < MIN_GROUP or len(y) < MIN_GROUP:
                continue
            p, u = _mwu_greater(x, y)
            ps[str(m)], detail[str(m)] = p, (x, y, u)
    if not ps:
        return _inconclusive(name, test, SQL_PARTS, params, "too few parts per machine and period")
    p_adj, target, p_raw = _bonferroni(ps)
    x, y, u = detail[target]
    return TemplateResult(
        template=name,
        test=test,
        target=target,
        p_value=p_adj,
        p_min_raw=p_raw,
        n_targets=len(ps),
        statistic=u,
        effect={
            "median_abs_deviation_after_mm": float(np.median(x)),
            "median_abs_deviation_before_mm": float(np.median(y)),
            "rank_biserial": float(2 * u / (len(x) * len(y)) - 1),
        },
        groups={"after": len(x), "before": len(y)},
        sql=SQL_PARTS.strip(),
        params=params,
        plot=box_figure(
            f"|dimension - median| through {target}: after vs before",
            "|deviation| (mm)",
            x,
            y,
            labels=("after window start", "reference period"),
        ),
    )


def sensor_reference_residual_trend(store: ProcessStore, w: Window) -> TemplateResult:
    """Master-part gauge checks: OLS slope over time, two-sided t-test."""
    name = "sensor_reference_residual_trend"
    test = "ols_slope_t_two_sided"
    params = [_path(store, "station_events")]
    checks = store.query(SQL_GAUGE, list(params))
    if len(checks) < MIN_CHECKS:
        return _inconclusive(name, test, SQL_GAUGE, params, f"fewer than {MIN_CHECKS} gauge checks")
    hours = (checks["ts"] - checks["ts"].min()).dt.total_seconds().to_numpy() / 3600.0
    values = checks["value"].to_numpy(dtype=float)
    fit = stats.linregress(hours, values)
    return TemplateResult(
        template=name,
        test=test,
        target="Q1",
        p_value=float(fit.pvalue),
        p_min_raw=float(fit.pvalue),
        n_targets=1,
        statistic=float(fit.slope / fit.stderr) if fit.stderr > 0 else None,
        effect={"slope_um_per_hour": float(fit.slope) * 1000.0, "r": float(fit.rvalue)},
        groups={"checks": len(values)},
        sql=SQL_GAUGE.strip(),
        params=params,
        plot=trend_figure(
            "Master-part gauge check (reading - certified size)",
            hours,
            values * 1000.0,
            float(fit.intercept) * 1000.0,
            float(fit.slope) * 1000.0,
        ),
    )


TEMPLATES: Final[dict[str, Template]] = {
    "defect_rate_by_tool_wear_level": defect_rate_by_tool_wear_level,
    "spindle_load_defective_vs_good": spindle_load_defective_vs_good,
    "temperature_delta_defective_vs_good": temperature_delta_defective_vs_good,
    "power_command_residual_defective_vs_good": power_command_residual_defective_vs_good,
    "station_defect_rate_change_point": station_defect_rate_change_point,
    "defect_rate_by_batch": defect_rate_by_batch,
    "defect_rate_by_parameter_set": defect_rate_by_parameter_set,
    "sensor_reference_residual_trend": sensor_reference_residual_trend,
    "ambient_temperature_defective_vs_good": ambient_temperature_defective_vs_good,
    "defect_rate_by_shift": defect_rate_by_shift,
}
