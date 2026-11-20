#!/usr/bin/env python
"""Step 5 gate 2 (null check) and template power per effect size.

Usage::

    nice -n 19 uv run python scripts/check_scientist_null.py --workers 2

- **Null:** ``--null`` fault-free 72 h simulations (default 1,000; seeds 900000+i), each
  with a pseudo alarm window (reference = first 12 h, affected from hour 36). Every
  template runs on every simulation; false-positive rate per template must be
  <= 0.05 + 3 SE.
- **Power:** per fault, ``--levels`` magnitudes evenly spaced from the fault's validity
  floor to the top of its standard sampling range, ``--power-runs`` runs each. Power is
  measured on runs that raised a case (the scientist only runs for cases), with the
  case's own window; the alarm rate is reported alongside.

Writes ``reports/diagnosis/scientist_null_power.json``. Simulations run in temporary
directories and are deleted. Simulator power says nothing about real plants.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.scientist.checks import null_summary, template_pvalues
from iqs_diagnosis.scientist.runner import window_for_case
from iqs_diagnosis.scientist.templates import TEMPLATES
from iqs_sim.faults import FAULT_SPECS, FaultType
from iqs_sim.pipeline import run_scenario
from iqs_sim.rng import stream
from iqs_sim.scenarios import LineConfig, Scenario, sample_fault

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = REPO_ROOT / "configs/taxonomy/causes.yaml"
START = datetime(2026, 1, 5, 6, tzinfo=UTC)
HOURS = 72.0


def _null_one(i: int) -> dict[str, float | None]:
    sc = Scenario(name=f"null_{i:04d}", seed=900_000 + i, start=START, duration_hours=HOURS)
    with tempfile.TemporaryDirectory() as d:
        out = run_scenario(sc, Path(d) / "run", load_taxonomy(TAXONOMY))
        return template_pvalues(out.observed_dir)


def _power_one(args: tuple[str, int, int, float]) -> dict[str, Any]:
    fault, level, i, magnitude = args
    ft = FaultType(fault)
    rng = stream(424_243, f"power:{fault}:{level}:{i}")
    f = sample_fault(ft, rng, duration_hours=HOURS, line=LineConfig())
    sign = -1.0 if f.magnitude < 0 else 1.0
    f = f.model_copy(update={"magnitude": round(sign * magnitude, 6)})
    sc = Scenario(
        name=f"power_{i:04d}",
        seed=int(rng.integers(0, 2**31 - 1)),
        start=START,
        duration_hours=HOURS,
        faults=(f,),
    )
    with tempfile.TemporaryDirectory() as d:
        out = run_scenario(sc, Path(d) / "run", load_taxonomy(TAXONOMY))
        case = load_case(out.run_dir)
        if case is None:
            return {"fault": fault, "level": level, "magnitude": magnitude, "case": False}
        with ProcessStore(out.observed_dir) as store:
            w = window_for_case(case, store)
            ps = {n: t(store, w).p_value for n, t in TEMPLATES.items()}
        return {"fault": fault, "level": level, "magnitude": magnitude, "case": True, "p": ps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--null", type=int, default=1000)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--power-runs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "reports/diagnosis/scientist_null_power.json"
    )
    args = ap.parse_args()

    own = {c.simulator_fault: c.test_template for c in load_taxonomy(TAXONOMY).causes}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        null_p = list(ex.map(_null_one, range(args.null), chunksize=8))
    t_null = time.perf_counter() - t0
    summary = null_summary(null_p)
    print(f"null: {args.null} fault-free runs in {t_null:.0f}s")
    for s in summary:
        r, c = s.false_positive_rate, s.rate_among_conclusive
        print(
            f"  {s.template:42s} FP {r.value:.4f} [{r.low:.4f}, {r.high:.4f}] ({s.positives}/{s.n})"
            f"  inconclusive {s.inconclusive}  among conclusive {c.value:.4f} (n={c.n})"
            f"  limit {s.limit:.4f}  {'PASS' if s.passed else 'FAIL'}"
        )

    jobs = []
    for ft in FaultType:
        spec = FAULT_SPECS[ft]
        mags = np.linspace(spec.min_abs_magnitude, spec.sample_range[1], args.levels)
        for level, m in enumerate(mags):
            jobs += [(ft.value, level, i, float(m)) for i in range(args.power_runs)]
    t1 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        power_rows = list(ex.map(_power_one, jobs, chunksize=4))
    t_power = time.perf_counter() - t1
    power: dict[str, list[dict[str, Any]]] = {}
    for ft in FaultType:
        tmpl = own[ft.value]
        levels = []
        for level in range(args.levels):
            rows = [r for r in power_rows if r["fault"] == ft.value and r["level"] == level]
            cases = [r for r in rows if r["case"]]
            hits = sum(1 for r in cases if r["p"][tmpl] is not None and r["p"][tmpl] <= 0.05)
            levels.append(
                {
                    "magnitude": rows[0]["magnitude"],
                    "runs": len(rows),
                    "cases": len(cases),
                    "alarm_rate": len(cases) / len(rows),
                    "power_given_case": (hits / len(cases)) if cases else None,
                    "supported": hits,
                }
            )
        power[ft.value] = levels
    print(f"power: {len(jobs)} runs in {t_power:.0f}s")
    for ft, levels in power.items():
        cells = "  ".join(
            f"{lv['magnitude']:.4g}: alarm {lv['alarm_rate']:.2f}, power "
            + ("n/a" if lv["power_given_case"] is None else f"{lv['power_given_case']:.2f}")
            for lv in levels
        )
        print(f"  {ft:30s} {cells}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "note": "Simulator null and power checks; not real-plant numbers.",
                "null": {
                    "runs": args.null,
                    "seconds": round(t_null, 1),
                    "window": "reference first 12 h, affected from hour 36",
                    "templates": [s.model_dump(mode="json") for s in summary],
                    "gate2_passed": all(s.passed for s in summary),
                },
                "power": {
                    "runs": len(jobs),
                    "seconds": round(t_power, 1),
                    "own_template": own,
                    "by_fault": power,
                },
                "workers": args.workers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"gate 2 {'PASSED' if all(s.passed for s in summary) else 'FAILED'}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
