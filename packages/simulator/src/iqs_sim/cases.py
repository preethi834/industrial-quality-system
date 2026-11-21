"""Case generator: turn an observed run into a diagnosis case, like a quality engineer would.

It reads **only** the ``observed/`` directory (never ``ground_truth/``), so a case
cannot contain the injected fault, its target or its onset. What it reports is
what monitoring on the real line would see:

- an SPC-style alarm: the first time the rolling reject rate exceeds the baseline
  rate by more than ``z`` binomial standard errors (the alarm time is *not* the
  fault onset, which stays unknown);
- the reject rate before and after the alarm;
- the mix of reject reasons after the alarm.

Where the rejects concentrate (machine, batch, shift, ...) is deliberately *not*
reported: finding that is the diagnosis agent's job.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from iqs_sim import __version__
from iqs_sim.export import write_json

CASE_FILE = "case.json"


@dataclass(frozen=True)
class AlarmConfig:
    """Defaults measured in step 5a: 0/30 false alarms on fault-free 72 h runs,
    55/60 alarms on a sampled single-fault family (see docs/notes/step-5a.md)."""

    baseline_hours: float = 12.0
    window_hours: float = 6.0
    step_hours: float = 1.0
    z: float = 3.5
    # Floor on the baseline rate: the line's known long-run reject rate (about 2 %).
    # Without it a lucky 12 h baseline makes later normal hours look like an alarm.
    min_baseline_rate: float = 0.02
    min_relative_increase: float = 1.5


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _iso(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def find_alarm(parts: pd.DataFrame, cfg: AlarmConfig) -> pd.Timestamp | None:
    """End of the first rolling window whose reject rate is significantly above baseline."""
    done = parts["completed_ts"]
    t0 = done.min().floor("h")
    baseline_end = t0 + pd.Timedelta(hours=cfg.baseline_hours)
    base = parts[done < baseline_end]
    if base.empty:
        return None
    p0 = max(cfg.min_baseline_rate, float((base["final_quality"] == "reject").mean()))
    window = pd.Timedelta(hours=cfg.window_hours)
    end = baseline_end + window
    last = done.max()
    while end <= last:
        sel = parts[(done >= end - window) & (done < end)]
        n = len(sel)
        if n:
            k = int((sel["final_quality"] == "reject").sum())
            limit = n * p0 + cfg.z * math.sqrt(n * p0 * (1 - p0))
            if k > limit and k / n >= cfg.min_relative_increase * p0:
                return end
        end += pd.Timedelta(hours=cfg.step_hours)
    return None


def generate_case(observed_dir: Path, cfg: AlarmConfig | None = None) -> dict[str, Any] | None:
    """Build a case from observed data, or ``None`` if monitoring raises no alarm."""
    cfg = cfg or AlarmConfig()
    run = json.loads((observed_dir / "run.json").read_text(encoding="utf-8"))
    parts = pd.read_parquet(
        observed_dir / "parts.parquet",
        columns=["part_id", "completed_ts", "final_quality", "reject_reason"],
    )
    alarm = find_alarm(parts, cfg)
    if alarm is None:
        return None

    done = parts["completed_ts"]
    before = parts[done < alarm - pd.Timedelta(hours=cfg.window_hours)]
    after = parts[done >= alarm - pd.Timedelta(hours=cfg.window_hours)]
    rate_before = float((before["final_quality"] == "reject").mean())
    rate_after = float((after["final_quality"] == "reject").mean())
    rejected_after = after[after["final_quality"] == "reject"]
    mix = {
        str(reason): round(float(share), 4)
        for reason, share in rejected_after["reject_reason"]
        .value_counts(normalize=True)
        .sort_index()
        .items()
    }
    top_reason = max(mix, key=lambda r: (mix[r], r))
    case_id = (
        "SIM-"
        + hashlib.sha256(
            (observed_dir / "run.json").read_bytes() + str(cfg).encode("utf-8")
        ).hexdigest()[:12]
    )
    line = run["line_id"]
    return {
        "case_id": case_id,
        "source": "simulator",
        "generator": f"iqs-sim {__version__}",
        "title": f"Reject rate increase on line {line}",
        "description": (
            f"Reject-rate alarm on line {line} at {_iso(alarm)}. The reject rate rose from "
            f"{_pct(rate_before)} before the alarm window to {_pct(rate_after)} since "
            f"{_iso(alarm - pd.Timedelta(hours=cfg.window_hours))}. Most rejects are "
            f"'{top_reason}' ({_pct(mix[top_reason])})."
        ),
        "line_id": line,
        "alarm_ts": _iso(alarm),
        "symptoms": [
            {
                "code": "reject_rate_increase",
                "rate_before": round(rate_before, 4),
                "rate_after": round(rate_after, 4),
                "parts_before": len(before),
                "parts_after": len(after),
                "window_start": _iso(alarm - pd.Timedelta(hours=cfg.window_hours)),
            },
            {"code": "reject_reason_mix", "shares_after": mix},
        ],
        "alarm_rule": {
            "baseline_hours": cfg.baseline_hours,
            "window_hours": cfg.window_hours,
            "z": cfg.z,
            "min_relative_increase": cfg.min_relative_increase,
        },
        "data": {
            "observed_dir": "../observed",
            "observed_run_sha256": hashlib.sha256(
                (observed_dir / "run.json").read_bytes()
            ).hexdigest(),
            "tables": sorted(run["tables"]),
        },
    }


def write_case(case: dict[str, Any], cases_dir: Path) -> Path:
    cases_dir.mkdir(parents=True, exist_ok=False)
    path = cases_dir / CASE_FILE
    write_json(path, case)
    return path
