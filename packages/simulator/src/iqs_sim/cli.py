"""Command-line interface (``scripts/simulate.py`` is a thin wrapper around ``main``).

Examples (from the repository root, prefixed with ``uv run python scripts/``)::

simulate.py run --scenario configs/simulator/scenarios/tool_wear_m1.yaml --out ~/iqs-data/sim/tw1
simulate.py validate configs/simulator/scenarios/*.yaml
simulate.py family --name eval_v1 --seed 2026 --n-per-fault 50 --out families/eval_v1
simulate.py run-family --family-dir families/eval_v1 --out runs/eval_v1 --workers 2
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from iqs_common.taxonomy import load_taxonomy
from iqs_sim.faults import FaultType
from iqs_sim.pipeline import run_family, run_scenario
from iqs_sim.scenarios import generate_family, load_scenario, write_family


def _parser(default_taxonomy: Path | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simulate.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate one scenario into a new run directory")
    run.add_argument("--scenario", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True, help="run directory (must be empty)")
    run.add_argument("--seed", type=int, help="override the scenario seed")
    run.add_argument("--duration-hours", type=float, help="override the scenario duration")
    run.add_argument(
        "--taxonomy",
        type=Path,
        default=default_taxonomy,
        required=default_taxonomy is None,
        help="taxonomy YAML used to label ground truth",
    )

    val = sub.add_parser("validate", help="validate scenario files without running them")
    val.add_argument("scenarios", type=Path, nargs="+")

    fam = sub.add_parser("family", help="write a reproducible family of single-fault scenarios")
    fam.add_argument("--name", required=True, help="family name (prefix of scenario names)")
    fam.add_argument("--seed", type=int, required=True)
    fam.add_argument("--n-per-fault", type=int, default=50)
    fam.add_argument("--duration-hours", type=float, default=72.0)
    fam.add_argument("--start", default="2026-01-05T06:00:00Z")
    fam.add_argument("--out", type=Path, required=True)
    fam.add_argument(
        "--fault", action="append", choices=[f.value for f in FaultType], help="limit fault types"
    )

    rf = sub.add_parser("run-family", help="run every scenario YAML of a family directory")
    rf.add_argument("--family-dir", type=Path, required=True, help="output of `family`")
    rf.add_argument("--out", type=Path, required=True, help="runs directory (must be empty)")
    rf.add_argument("--workers", type=int, default=1)
    rf.add_argument(
        "--taxonomy",
        type=Path,
        default=default_taxonomy,
        required=default_taxonomy is None,
        help="taxonomy YAML used to label ground truth",
    )
    return p


def _require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"error: {path} is not empty; choose a new directory")


def main(argv: Sequence[str] | None = None, *, default_taxonomy: Path | None = None) -> int:
    args = _parser(default_taxonomy).parse_args(argv)
    if args.command == "run":
        _require_empty(args.out)
        scenario = load_scenario(args.scenario).with_overrides(
            seed=args.seed, duration_hours=args.duration_hours
        )
        out = run_scenario(scenario, args.out, load_taxonomy(args.taxonomy))
        print(f"parts:        {out.n_parts}")
        print(f"observed:     {out.observed_dir}")
        print(f"ground truth: {out.ground_truth_dir}  (evaluation only, never give to the agent)")
        print(f"case:         {out.case_path or 'none (no reject-rate alarm)'}")
        return 0
    if args.command == "validate":
        failed = 0
        for path in args.scenarios:
            try:
                sc = load_scenario(path)
            except Exception as exc:  # report every file, then fail
                failed += 1
                print(f"FAIL {path}: {exc}", file=sys.stderr)
            else:
                print(f"ok   {path} ({sc.name}, {len(sc.faults)} fault(s))")
        return 1 if failed else 0
    if args.command == "run-family":
        _require_empty(args.out)
        paths = sorted(args.family_dir.glob("*.yaml"))
        if not paths:
            raise SystemExit(f"error: no scenario YAMLs in {args.family_dir}")
        index = run_family(paths, args.out, args.taxonomy, workers=args.workers)
        print(f"runs: {index['n_runs']}, cases: {index['n_cases']} -> {args.out}")
        return 0
    # family
    _require_empty(args.out)
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    types = tuple(FaultType(f) for f in args.fault) if args.fault else tuple(FaultType)
    scenarios = generate_family(
        family=args.name,
        seed=args.seed,
        n_per_fault=args.n_per_fault,
        start=start,
        duration_hours=args.duration_hours,
        fault_types=types,
    )
    manifest = write_family(scenarios, args.out)
    print(f"{manifest['n_scenarios']} scenarios, family_sha256={manifest['family_sha256']}")
    return 0
