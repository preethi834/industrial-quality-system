"""One call from scenario to files: simulate, export, generate the case, label it."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from iqs_common.taxonomy import Taxonomy, load_taxonomy
from iqs_sim.cases import AlarmConfig, generate_case, write_case
from iqs_sim.export import CASES_DIR, write_json, write_run
from iqs_sim.faults import cause_code_for
from iqs_sim.line import simulate
from iqs_sim.scenarios import Scenario, load_scenario

CASE_LABELS_FILE = "case_labels.json"
RUNS_INDEX_FILE = "runs_index.json"


@dataclass(frozen=True)
class RunOutput:
    run_dir: Path
    observed_dir: Path
    ground_truth_dir: Path
    case_path: Path | None
    n_parts: int


def run_scenario(
    scenario: Scenario,
    run_dir: Path,
    taxonomy: Taxonomy,
    *,
    alarm: AlarmConfig | None = None,
) -> RunOutput:
    """Simulate ``scenario`` into ``run_dir`` (created; must not contain earlier output)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result = simulate(scenario)
    dirs = write_run(result, run_dir, taxonomy)
    case = generate_case(dirs["observed"], alarm)
    case_path = write_case(case, run_dir / CASES_DIR) if case is not None else None
    write_json(
        dirs["ground_truth"] / CASE_LABELS_FILE,
        {
            "case_id": None if case is None else case["case_id"],
            "alarm_ts": None if case is None else case["alarm_ts"],
            "faults": [
                {
                    "fault_id": f"F{i + 1:02d}",
                    "simulator_fault": f.type.value,
                    "cause_code": cause_code_for(taxonomy, f.type.value),
                }
                for i, f in enumerate(scenario.faults)
            ],
        },
    )
    return RunOutput(
        run_dir=run_dir,
        observed_dir=dirs["observed"],
        ground_truth_dir=dirs["ground_truth"],
        case_path=case_path,
        n_parts=len(result.observed["parts"]),
    )


def run_dir_name(scenario_name: str) -> str:
    """Opaque run directory name: scenario names contain the fault type, paths must not."""
    return "run-" + hashlib.sha256(scenario_name.encode("utf-8")).hexdigest()[:12]


def _run_one(args: tuple[Path, Path, Path]) -> dict[str, object]:
    scenario_path, out_dir, taxonomy_path = args
    scenario = load_scenario(scenario_path)
    run_dir = out_dir / run_dir_name(scenario.name)
    out = run_scenario(scenario, run_dir, load_taxonomy(taxonomy_path))
    return {
        "scenario": scenario.name,
        "run_dir": run_dir.name,
        "case": out.case_path is not None,
        "n_parts": out.n_parts,
    }


def run_family(
    scenario_paths: Sequence[Path],
    out_dir: Path,
    taxonomy_path: Path,
    *,
    workers: int = 1,
) -> dict[str, object]:
    """Run every scenario into ``out_dir/run-<hash>/`` and write ``runs_index.json``.

    The index maps run directories to scenario names (which name the fault), so it is
    evaluation-side data like ``ground_truth/``: never hand it to the agent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [run_dir_name(load_scenario(p).name) for p in scenario_paths]
    if len(set(names)) != len(names):
        raise ValueError("duplicate scenario names in family")
    jobs = [(p, out_dir, taxonomy_path) for p in sorted(scenario_paths)]
    if workers <= 1:
        rows = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, jobs))
    index = {
        "n_runs": len(rows),
        "n_cases": sum(1 for r in rows if r["case"]),
        "runs": sorted(rows, key=lambda r: str(r["run_dir"])),
    }
    write_json(out_dir / RUNS_INDEX_FILE, index)
    return index
