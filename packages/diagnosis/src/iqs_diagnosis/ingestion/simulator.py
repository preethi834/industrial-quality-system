"""Load simulator cases (``<run>/cases/case.json`` + ``<run>/observed/``).

Only the case file and the observed directory are read. The run layout is fixed by
``iqs_sim.export`` (a test checks that these names match); this module does not
import the simulator, so the diagnosis service has no runtime dependency on it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from iqs_diagnosis.domain.case import DiagnosisCase

CASES_DIR: Final = "cases"
CASE_FILE: Final = "case.json"
OBSERVED_DIR: Final = "observed"


def _ts(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_case(run_dir: Path) -> DiagnosisCase | None:
    """The case of one run, or ``None`` if monitoring raised no alarm (no case file)."""
    path = run_dir / CASES_DIR / CASE_FILE
    if not path.is_file():
        return None
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    observed = (path.parent / raw["data"]["observed_dir"]).resolve()
    rate = next((s for s in raw["symptoms"] if s["code"] == "reject_rate_increase"), None)
    return DiagnosisCase(
        case_id=raw["case_id"],
        source=raw["source"],
        title=raw["title"],
        description=raw["description"],
        line_id=raw.get("line_id"),
        alarm_ts=_ts(raw.get("alarm_ts")),
        window_start=_ts(rate["window_start"]) if rate else None,
        baseline_hours=float(raw["alarm_rule"]["baseline_hours"]),
        observed_dir=observed,
        reported_symptoms=tuple(raw["symptoms"]),
    )


def iter_run_dirs(root: Path) -> Iterator[Path]:
    """Every run directory below ``root`` (a directory that has an ``observed/`` child)."""
    for observed in sorted(root.rglob(OBSERVED_DIR)):
        if observed.is_dir():
            yield observed.parent
