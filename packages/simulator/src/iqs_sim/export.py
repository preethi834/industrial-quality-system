"""Write a simulation run to disk: observable tables and ground truth, separately.

Layout of one run directory::

    <run>/observed/       parts, sensor_readings, station_events, batches, shifts,
                          machines (.parquet) + run.json   <- the agent may read this
    <run>/ground_truth/   faults, part_truth, batch_truth (.parquet), scenario.json,
                          run.json, case_labels.json        <- evaluation only
    <run>/cases/          case.json (observed symptoms only; written by iqs_sim.cases)

Rules:

- Schemas are fixed and identical for every scenario, so the presence of a column
  can never reveal a fault. The observable side holds no scenario name, seed,
  description or fault field (a seed would let a reader re-run the fault-free twin).
- Files are byte-for-byte deterministic: explicit Arrow schemas, no pandas
  metadata, no write timestamps, sorted JSON keys.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from iqs_common.taxonomy import Taxonomy
from iqs_sim import __version__
from iqs_sim.faults import cause_code_for
from iqs_sim.line import Row, SimulationResult
from iqs_sim.machines import (
    AMBIENT_SPEC_C,
    DIMENSION_TOLERANCE_MM,
    LINE_ID,
    MATERIAL,
    NOMINAL_DIMENSION_MM,
    NOMINAL_HARDNESS_HB,
)

SCHEMA_VERSION = "1"
OBSERVED_DIR = "observed"
GROUND_TRUTH_DIR = "ground_truth"
CASES_DIR = "cases"

_TS = pa.timestamp("us", tz="UTC")
_STR = pa.string()
_F64 = pa.float64()


def _f(name: str, type_: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, type_, nullable=nullable)


OBSERVED_SCHEMAS: dict[str, pa.Schema] = {
    "parts": pa.schema(
        [
            _f("part_id", _STR),
            _f("batch_id", _STR),
            _f("shift_id", _STR),
            _f("started_ts", _TS),
            _f("completed_ts", _TS),
            _f("s1_machine_id", _STR),
            _f("s2_machine_id", _STR),
            _f("s3_machine_id", _STR),
            _f("station_path", _STR),
            _f("measured_dimension_mm", _F64),
            _f("surface_defect_detected", pa.bool_()),
            _f("final_quality", _STR),
            _f("reject_reason", _STR, nullable=True),
        ]
    ),
    "sensor_readings": pa.schema(
        [
            _f("reading_id", pa.int64()),
            _f("ts", _TS),
            _f("part_id", _STR),
            _f("station_id", _STR),
            _f("machine_id", _STR),
            _f("tool_id", _STR),
            _f("parameter_set_id", _STR),
            _f("setpoint_speed_rpm", _F64),
            _f("setpoint_feed_mm_rev", _F64),
            _f("power_commanded_kw", _F64),
            _f("power_actual_kw", _F64),
            _f("spindle_load_pct", _F64),
            _f("temperature_c", _F64),
            _f("ambient_temperature_c", _F64),
            _f("vibration_mm_s", _F64),
            _f("tool_wear_mm", _F64),
        ]
    ),
    "station_events": pa.schema(
        [
            _f("event_id", pa.int64()),
            _f("ts", _TS),
            _f("station_id", _STR),
            _f("machine_id", _STR),
            _f("event_type", _STR),
            _f("part_id", _STR, nullable=True),
            _f("value", _F64, nullable=True),
        ]
    ),
    "batches": pa.schema(
        [
            _f("batch_id", _STR),
            _f("supplier_id", _STR),
            _f("material", _STR),
            _f("hardness_cert_hb", _F64),
            _f("quantity", pa.int64()),
            _f("first_use_ts", _TS),
        ]
    ),
    "shifts": pa.schema(
        [
            _f("shift_id", _STR),
            _f("shift_code", _STR),
            _f("crew_id", _STR),
            _f("start_ts", _TS),
            _f("end_ts", _TS),
        ]
    ),
    "machines": pa.schema(
        [
            _f("machine_id", _STR),
            _f("station_id", _STR),
            _f("kind", _STR),
            _f("process", _STR),
            _f("nominal_cycle_minutes", _F64),
            _f("nominal_speed_rpm", _F64, nullable=True),
            _f("nominal_feed_mm_rev", _F64, nullable=True),
            _f("speed_window_low_rpm", _F64, nullable=True),
            _f("speed_window_high_rpm", _F64, nullable=True),
            _f("tool_life_parts", pa.int64(), nullable=True),
            _f("tool_wear_limit_mm", _F64, nullable=True),
        ]
    ),
}

GROUND_TRUTH_SCHEMAS: dict[str, pa.Schema] = {
    "faults": pa.schema(
        [
            _f("fault_id", _STR),
            _f("simulator_fault", _STR),
            _f("cause_code", _STR),
            _f("target", _STR),
            _f("magnitude", _F64),
            _f("magnitude_unit", _STR),
            _f("onset_ts", _TS, nullable=True),
            _f("end_ts", _TS, nullable=True),
        ]
    ),
    "part_truth": pa.schema(
        [
            _f("part_id", _STR),
            _f("true_dimension_mm", _F64),
            _f("true_surface_defect", pa.bool_()),
            _f("truly_defective", pa.bool_()),
            _f("defect_probability", _F64),
            _f("fault_ids", pa.list_(pa.field("item", _STR, nullable=False))),
        ]
    ),
    "batch_truth": pa.schema([_f("batch_id", _STR), _f("hardness_true_hb", _F64)]),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _to_table(rows: list[Row], schema: pa.Schema, start_us: int) -> pa.Table:
    """Rows hold times as float minutes since the scenario start; convert to UTC microseconds."""
    columns: dict[str, list[Any]] = {}
    for fld in schema:
        values = [row[fld.name] for row in rows]
        if pa.types.is_timestamp(fld.type):
            values = [None if v is None else start_us + round(v * 60_000_000) for v in values]
        columns[fld.name] = values
    return pa.Table.from_pydict(columns, schema=schema)


def _write_tables(
    tables: dict[str, list[Row]], schemas: dict[str, pa.Schema], out: Path, start_us: int
) -> dict[str, dict[str, Any]]:
    if set(tables) != set(schemas):
        raise ValueError(f"tables {sorted(tables)} do not match schemas {sorted(schemas)}")
    out.mkdir(parents=True, exist_ok=False)
    info: dict[str, dict[str, Any]] = {}
    for name in sorted(schemas):
        path = out / f"{name}.parquet"
        table = _to_table(tables[name], schemas[name], start_us)
        pq.write_table(table, path, compression="zstd")
        info[name] = {"rows": table.num_rows, "sha256": sha256_file(path)}
    return info


def write_run(result: SimulationResult, run_dir: Path, taxonomy: Taxonomy) -> dict[str, Path]:
    """Write ``observed/`` and ``ground_truth/`` under ``run_dir`` (both must not exist yet)."""
    sc = result.scenario
    start_us = int(sc.start.timestamp()) * 1_000_000 + sc.start.microsecond
    observed_dir = run_dir / OBSERVED_DIR
    truth_dir = run_dir / GROUND_TRUTH_DIR

    obs_info = _write_tables(result.observed, OBSERVED_SCHEMAS, observed_dir, start_us)
    end = sc.start.timestamp() + result.end_minutes * 60
    write_json(
        observed_dir / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generator": f"iqs-sim {__version__}",
            "line_id": LINE_ID,
            "start": sc.start.isoformat().replace("+00:00", "Z"),
            "end": _iso(end),
            "spec": {
                "nominal_dimension_mm": NOMINAL_DIMENSION_MM,
                "dimension_tolerance_mm": DIMENSION_TOLERANCE_MM,
                "ambient_temperature_range_c": list(AMBIENT_SPEC_C),
                "material": MATERIAL,
                "nominal_hardness_hb": NOMINAL_HARDNESS_HB,
            },
            "tables": obs_info,
        },
    )

    truth_tables = dict(result.ground_truth)
    truth_tables["faults"] = [
        {**row, "cause_code": cause_code_for(taxonomy, row["simulator_fault"])}
        for row in truth_tables["faults"]
    ]
    truth_info = _write_tables(truth_tables, GROUND_TRUTH_SCHEMAS, truth_dir, start_us)
    write_json(truth_dir / "scenario.json", sc.model_dump(mode="json"))
    write_json(
        truth_dir / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generator": f"iqs-sim {__version__}",
            "scenario": sc.name,
            "seed": sc.seed,
            "taxonomy_version": taxonomy.version,
            "taxonomy_fingerprint": taxonomy.fingerprint,
            "observed_run_sha256": sha256_file(observed_dir / "run.json"),
            "tables": truth_info,
        },
    )
    return {"observed": observed_dir, "ground_truth": truth_dir}


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(round(epoch_s, 6), UTC).isoformat().replace("+00:00", "Z")
