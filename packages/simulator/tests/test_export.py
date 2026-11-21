"""Parquet export: fixed schemas, byte-level determinism, ground truth kept apart."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from iqs_common.taxonomy import Taxonomy
from iqs_sim.export import (
    GROUND_TRUTH_DIR,
    GROUND_TRUTH_SCHEMAS,
    OBSERVED_DIR,
    OBSERVED_SCHEMAS,
    sha256_file,
    write_run,
)
from iqs_sim.faults import FaultType
from iqs_sim.line import SimulationResult
from iqs_sim.scenarios import Scenario

Run = Callable[[Scenario], SimulationResult]
OBSERVED_FILES = {f"{name}.parquet" for name in OBSERVED_SCHEMAS} | {"run.json"}


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): sha256_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _write(result: SimulationResult, path: Path, taxonomy: Taxonomy) -> Path:
    write_run(result, path, taxonomy)
    return path


def test_same_seed_gives_identical_files(
    reference_scenarios: dict[str, Scenario], taxonomy: Taxonomy, tmp_path: Path
) -> None:
    from iqs_sim.line import simulate  # fresh runs, not the session cache

    sc = reference_scenarios["tool_wear_m1"].with_overrides(duration_hours=30)
    a = _hashes(_write(simulate(sc), tmp_path / "a", taxonomy))
    b = _hashes(_write(simulate(sc), tmp_path / "b", taxonomy))
    assert a == b
    assert len(a) == len(OBSERVED_FILES) + len(GROUND_TRUTH_SCHEMAS) + 2

    other = _hashes(_write(simulate(sc.with_overrides(seed=sc.seed + 1)), tmp_path / "c", taxonomy))
    # Tables that depend on random draws must change with the seed (shifts, machines
    # and the fault list are fixed by the calendar, the layout and the scenario).
    random_tables = ["parts", "sensor_readings", "station_events", "batches"]
    for name in random_tables:
        key = f"{OBSERVED_DIR}/{name}.parquet"
        assert a[key] != other[key], name


def test_observed_files_have_exact_schemas(
    reference_scenarios: dict[str, Scenario], run_sim: Run, taxonomy: Taxonomy, tmp_path: Path
) -> None:
    run = _write(run_sim(reference_scenarios["baseline"]), tmp_path / "r", taxonomy)
    observed, truth = run / OBSERVED_DIR, run / GROUND_TRUTH_DIR
    assert {p.name for p in observed.iterdir()} == OBSERVED_FILES
    for name, schema in OBSERVED_SCHEMAS.items():
        table = pq.read_table(observed / f"{name}.parquet")
        assert table.schema.remove_metadata() == schema, name
        assert table.num_rows > 0
    for name, schema in GROUND_TRUTH_SCHEMAS.items():
        assert pq.read_schema(truth / f"{name}.parquet").remove_metadata() == schema


def test_timestamps_and_keys_are_consistent(
    reference_scenarios: dict[str, Scenario], run_sim: Run, taxonomy: Taxonomy, tmp_path: Path
) -> None:
    sc = reference_scenarios["baseline"]
    run = _write(run_sim(sc), tmp_path / "r", taxonomy) / OBSERVED_DIR
    parts = pq.read_table(run / "parts.parquet").to_pandas()
    readings = pq.read_table(run / "sensor_readings.parquet").to_pandas()
    batches = pq.read_table(run / "batches.parquet").to_pandas()
    shifts = pq.read_table(run / "shifts.parquet").to_pandas()
    machines = pq.read_table(run / "machines.parquet").to_pandas()
    assert parts["part_id"].is_unique and readings["reading_id"].is_unique
    assert set(readings["part_id"]) == set(parts["part_id"])
    assert set(parts["batch_id"]) == set(batches["batch_id"])
    assert set(parts["shift_id"]) <= set(shifts["shift_id"])
    assert set(readings["machine_id"]) <= set(machines["machine_id"])
    assert batches["quantity"].sum() == len(parts)
    assert parts["started_ts"].min() >= sc.start
    assert str(parts["started_ts"].dt.tz) == "UTC"
    run_json = json.loads((run / "run.json").read_text(encoding="utf-8"))
    for name, info in run_json["tables"].items():
        assert info["sha256"] == sha256_file(run / f"{name}.parquet")


def test_write_refuses_existing_output(
    reference_scenarios: dict[str, Scenario], run_sim: Run, taxonomy: Taxonomy, tmp_path: Path
) -> None:
    res = run_sim(reference_scenarios["baseline"])
    write_run(res, tmp_path / "r", taxonomy)
    with pytest.raises(FileExistsError):
        write_run(res, tmp_path / "r", taxonomy)


# ------------------------------------------------------------------- no leakage


def _forbidden_tokens(sc: Scenario, taxonomy: Taxonomy) -> set[str]:
    tokens = {f.value for f in FaultType} | set(taxonomy.leaf_codes)
    tokens |= {sc.name, "fault", "truth", "scenario", "seed", "magnitude", "onset"}
    if sc.description:
        tokens.add(sc.description)
    return tokens


def _observed_strings(observed: Path) -> list[str]:
    strings: list[str] = []
    for path in observed.iterdir():
        if path.suffix == ".parquet":
            table = pq.read_table(path)
            strings += table.column_names
            meta = table.schema.metadata or {}
            strings += [k.decode() + v.decode() for k, v in meta.items()]
            for col in table.columns:
                if col.type == "string":
                    strings += [v for v in col.to_pylist() if v is not None]
        else:
            strings.append(path.read_text(encoding="utf-8"))
    return strings


@pytest.mark.parametrize("name", ["tool_wear_m1", "bad_batch_b0008", "gauge_drift_q1"])
def test_observed_output_contains_no_ground_truth(
    name: str,
    reference_scenarios: dict[str, Scenario],
    run_sim: Run,
    taxonomy: Taxonomy,
    tmp_path: Path,
) -> None:
    sc = reference_scenarios[name]
    run = _write(run_sim(sc), tmp_path / "r", taxonomy)
    blob = "\n".join(_observed_strings(run / OBSERVED_DIR)).lower()
    for token in _forbidden_tokens(sc, taxonomy):
        assert token.lower() not in blob, token
    truth = json.loads((run / GROUND_TRUTH_DIR / "run.json").read_text(encoding="utf-8"))
    assert truth["scenario"] == sc.name and truth["seed"] == sc.seed


def test_observed_output_does_not_depend_on_scenario_metadata(
    reference_scenarios: dict[str, Scenario], run_sim: Run, taxonomy: Taxonomy, tmp_path: Path
) -> None:
    """Renaming or re-describing a scenario must not change one observable byte."""
    sc = reference_scenarios["baseline"]
    renamed = sc.model_copy(update={"name": "something_else", "description": "secret"})
    a = _hashes(_write(run_sim(sc), tmp_path / "a", taxonomy) / OBSERVED_DIR)
    b = _hashes(_write(run_sim(renamed), tmp_path / "b", taxonomy) / OBSERVED_DIR)
    assert a == b


def test_agent_style_duckdb_connection_cannot_read_ground_truth(
    reference_scenarios: dict[str, Scenario], run_sim: Run, taxonomy: Taxonomy, tmp_path: Path
) -> None:
    """Reference configuration for the scientist agent's read-only connection (step 5b).

    DuckDB is a dependency of the diagnosis package, not of iqs_sim, so this test is
    skipped when it is not installed.
    """
    duckdb = pytest.importorskip("duckdb")
    run = _write(run_sim(reference_scenarios["baseline"]), tmp_path / "r", taxonomy)
    observed = (run / OBSERVED_DIR).resolve()
    con = duckdb.connect()
    # Order matters: the allow-list must be set before external access is switched off.
    con.execute("SET allowed_directories = ?", [[str(observed) + "/"]])
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{observed}/parts.parquet')").fetchone()
    assert n is not None and n[0] > 0
    truth = (run / GROUND_TRUTH_DIR / "faults.parquet").resolve()
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT * FROM read_parquet('{truth}')").fetchall()
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT * FROM read_parquet('{observed}/../ground_truth/faults.parquet')")
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")
