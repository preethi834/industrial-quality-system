"""Read-only access to a simulator run's observable tables (DuckDB over Parquet).

The connection may read files under ``observed/`` only: the allow-list is set before
external access is switched off, and the configuration is then locked, so later SQL
cannot widen it (ADR from step 5a, decision 5). ``ground_truth/`` is a sibling
directory and therefore unreadable through this connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import duckdb
import pandas as pd

OBSERVED_TABLES: Final = (
    "parts",
    "sensor_readings",
    "station_events",
    "batches",
    "shifts",
    "machines",
)


class ProcessStore:
    """One locked, read-only DuckDB connection per observed directory."""

    def __init__(self, observed_dir: Path) -> None:
        self.observed_dir = observed_dir.resolve()
        if not self.observed_dir.is_dir():
            raise FileNotFoundError(f"observed directory not found: {self.observed_dir}")
        con = duckdb.connect(":memory:")
        # Order matters: allow-list first, then disable external access, then lock.
        con.execute("SET allowed_directories = ?", [[str(self.observed_dir) + "/"]])
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        self._con = con

    def table(self, name: str) -> pd.DataFrame:
        """Whole table as a DataFrame. ``name`` must be one of ``OBSERVED_TABLES``."""
        if name not in OBSERVED_TABLES:
            raise ValueError(f"unknown observed table {name!r}")
        path = self.observed_dir / f"{name}.parquet"
        return self.query("SELECT * FROM read_parquet(?)", [str(path)])

    def run_info(self) -> dict[str, object]:
        """The observable run metadata (``observed/run.json``: spec limits, table hashes)."""
        import json

        data: dict[str, object] = json.loads((self.observed_dir / "run.json").read_text("utf-8"))
        return data

    def query(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        """Run a fixed SQL template (never free-form model output) on this connection."""
        rel = self._con.sql(sql, params=params or None)
        if rel is None:  # statement without a result set
            return pd.DataFrame()
        return rel.to_arrow_table().to_pandas()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ProcessStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
