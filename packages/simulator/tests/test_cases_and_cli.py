"""Case generator (observed symptoms only), the run pipeline and the CLI."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from iqs_common.taxonomy import Taxonomy
from iqs_sim.cases import AlarmConfig, find_alarm, generate_case
from iqs_sim.cli import main
from iqs_sim.export import CASES_DIR, GROUND_TRUTH_DIR, OBSERVED_DIR, write_run
from iqs_sim.faults import FaultType
from iqs_sim.line import SimulationResult
from iqs_sim.pipeline import CASE_LABELS_FILE, run_scenario
from iqs_sim.scenarios import Scenario, scenario_to_yaml

from .conftest import SCENARIO_DIR, TAXONOMY_PATH

Run = Callable[[Scenario], SimulationResult]


def _parts(rates: list[float], per_hour: int = 30) -> pd.DataFrame:
    rows = []
    t0 = datetime(2026, 1, 5, tzinfo=UTC)
    for h, rate in enumerate(rates):
        n_bad = round(rate * per_hour)
        for i in range(per_hour):
            rows.append(
                {
                    "completed_ts": pd.Timestamp(t0) + pd.Timedelta(hours=h, minutes=i * 2),
                    "final_quality": "reject" if i < n_bad else "pass",
                }
            )
    return pd.DataFrame(rows)


def test_alarm_rule_on_synthetic_data() -> None:
    cfg = AlarmConfig()
    assert find_alarm(_parts([0.02] * 48), cfg) is None
    alarm = find_alarm(_parts([0.02] * 24 + [0.3] * 24), cfg)
    assert alarm is not None
    assert pd.Timestamp("2026-01-06T00:00Z") < alarm <= pd.Timestamp("2026-01-06T06:00Z")


def test_case_from_fault_has_symptoms_but_no_ground_truth(
    reference_scenarios: dict[str, Scenario],
    run_sim: Run,
    taxonomy: Taxonomy,
    tmp_path: Path,
) -> None:
    sc = reference_scenarios["fixture_offset_m2b"]
    run = tmp_path / "r"
    write_run(run_sim(sc), run, taxonomy)
    # Remove ground truth first: the generator must not need it.
    shutil.rmtree(run / GROUND_TRUTH_DIR)
    case = generate_case(run / OBSERVED_DIR)
    assert case is not None
    assert case["source"] == "simulator"
    assert {s["code"] for s in case["symptoms"]} == {"reject_rate_increase", "reject_reason_mix"}
    rise = case["symptoms"][0]
    assert rise["rate_after"] > 2 * rise["rate_before"]
    alarm = pd.Timestamp(case["alarm_ts"])
    assert alarm >= sc.start + pd.Timedelta(hours=sc.faults[0].onset_hours)

    text = json.dumps(case).lower()
    forbidden = {f.value for f in FaultType} | set(taxonomy.leaf_codes)
    forbidden |= {sc.name, "fault", "m2b", "onset", "magnitude", "fixture"}
    for token in forbidden:
        assert token.lower() not in text, token


def test_no_case_without_alarm(
    reference_scenarios: dict[str, Scenario], run_sim: Run, taxonomy: Taxonomy, tmp_path: Path
) -> None:
    write_run(run_sim(reference_scenarios["baseline"]), tmp_path / "r", taxonomy)
    assert generate_case(tmp_path / "r" / OBSERVED_DIR) is None


def test_pipeline_writes_case_and_label(
    reference_scenarios: dict[str, Scenario], taxonomy: Taxonomy, tmp_path: Path
) -> None:
    sc = reference_scenarios["tool_wear_m1"]
    out = run_scenario(sc, tmp_path / "r", taxonomy)
    assert out.case_path == tmp_path / "r" / CASES_DIR / "case.json"
    case = json.loads(out.case_path.read_text(encoding="utf-8"))
    label = json.loads((out.ground_truth_dir / CASE_LABELS_FILE).read_text(encoding="utf-8"))
    assert label["case_id"] == case["case_id"]
    assert label["faults"] == [
        {
            "fault_id": "F01",
            "simulator_fault": "accelerated_tool_wear",
            "cause_code": "MACHINE.TOOL_WEAR",
        }
    ]
    # Deterministic case id and content.
    again = run_scenario(sc, tmp_path / "r2", taxonomy)
    assert again.case_path is not None
    assert again.case_path.read_bytes() == out.case_path.read_bytes()


def test_cli_validate_run_and_family(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = sorted(str(p) for p in SCENARIO_DIR.glob("*.yaml"))
    assert main(["validate", *files]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nseed: -1\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1

    out = tmp_path / "run"
    code = main(
        [
            "run",
            "--scenario",
            str(SCENARIO_DIR / "drive_fault_m1.yaml"),
            "--out",
            str(out),
            "--duration-hours",
            "30",
            "--taxonomy",
            str(TAXONOMY_PATH),
        ]
    )
    assert code == 0
    assert (out / OBSERVED_DIR / "parts.parquet").is_file()
    assert (out / GROUND_TRUTH_DIR / "faults.parquet").is_file()
    with pytest.raises(SystemExit, match="not empty"):
        main(["run", "--scenario", files[0], "--out", str(out), "--taxonomy", str(TAXONOMY_PATH)])

    fam = tmp_path / "fam"
    assert (
        main(["family", "--name", "t", "--seed", "1", "--n-per-fault", "1", "--out", str(fam)]) == 0
    )
    assert len(list(fam.glob("*.yaml"))) == len(FaultType)
    assert "family_sha256" in capsys.readouterr().out


def test_cli_requires_taxonomy_without_default(tmp_path: Path) -> None:
    sc = tmp_path / "s.yaml"
    sc.write_text(
        scenario_to_yaml(
            Scenario(name="s", seed=1, start=datetime(2026, 1, 5, tzinfo=UTC), duration_hours=1)
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(["run", "--scenario", str(sc), "--out", str(tmp_path / "o")])
