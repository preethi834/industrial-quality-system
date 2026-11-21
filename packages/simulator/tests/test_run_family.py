"""run_family / `simulate.py run-family`: opaque run directories plus an index."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iqs_sim.cli import main
from iqs_sim.faults import FaultType
from iqs_sim.pipeline import RUNS_INDEX_FILE, run_dir_name, run_family
from iqs_sim.scenarios import generate_family, write_family

from .conftest import TAXONOMY_PATH


def _family(tmp_path: Path) -> list[Path]:
    scenarios = generate_family(
        family="rf",
        seed=5,
        n_per_fault=1,
        start=datetime(2026, 1, 5, 6, tzinfo=UTC),
        duration_hours=24.0,
        fault_types=(FaultType.FIXTURE_OFFSET, FaultType.COOLANT_FLOW_LOSS),
    )
    write_family(scenarios, tmp_path / "fam")
    return sorted((tmp_path / "fam").glob("*.yaml"))


def test_run_dir_name_is_opaque() -> None:
    name = run_dir_name("eval_v1_fixture_offset_003")
    assert name.startswith("run-") and len(name) == 16
    assert "fixture" not in name and name == run_dir_name("eval_v1_fixture_offset_003")


def test_run_family_writes_runs_and_index(tmp_path: Path) -> None:
    paths = _family(tmp_path)
    index = run_family(paths, tmp_path / "runs", TAXONOMY_PATH, workers=1)
    assert index["n_runs"] == 2
    dirs = sorted(p.name for p in (tmp_path / "runs").iterdir() if p.is_dir())
    assert all(d.startswith("run-") for d in dirs) and len(dirs) == 2
    assert (tmp_path / "runs" / RUNS_INDEX_FILE).is_file()
    for d in dirs:
        assert (tmp_path / "runs" / d / "observed" / "parts.parquet").is_file()


def test_cli_run_family_requires_empty_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _family(tmp_path)
    out = tmp_path / "runs"
    argv = ["run-family", "--family-dir", str(tmp_path / "fam"), "--out", str(out)]
    assert main(argv, default_taxonomy=TAXONOMY_PATH) == 0
    assert "runs: 2" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(argv, default_taxonomy=TAXONOMY_PATH)
