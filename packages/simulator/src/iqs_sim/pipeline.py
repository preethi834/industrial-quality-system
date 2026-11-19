"""One call from scenario to files: simulate, export, generate the case, label it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iqs_common.taxonomy import Taxonomy
from iqs_sim.cases import AlarmConfig, generate_case, write_case
from iqs_sim.export import CASES_DIR, write_json, write_run
from iqs_sim.faults import cause_code_for
from iqs_sim.line import simulate
from iqs_sim.scenarios import Scenario

CASE_LABELS_FILE = "case_labels.json"


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
