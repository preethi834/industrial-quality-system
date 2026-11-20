"""Case labels from simulator ground truth: evaluation only.

This is the only module in ``iqs_diagnosis`` (besides the evaluation package that
calls it) that reads a run's ground-truth directory; a test enforces that. The agent
path (ingestion, symptoms, retrieval, scoring) never sees these labels for a query
case; the case library gets them only as *confirmed* causes of past cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

GROUND_TRUTH_DIR: Final = "ground_truth"
CASE_LABELS_FILE: Final = "case_labels.json"


class CaseLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    cause_code: str
    simulator_fault: str


def load_case_label(run_dir: Path) -> CaseLabel | None:
    """Label of the run's case; ``None`` if no case was raised.

    Single-fault runs only (one primary cause per case, labelling guide); a run with
    several faults is rejected rather than silently labelled with the first one.
    """
    raw = json.loads((run_dir / GROUND_TRUTH_DIR / CASE_LABELS_FILE).read_text(encoding="utf-8"))
    if raw["case_id"] is None:
        return None
    faults = raw["faults"]
    if len(faults) != 1:
        raise ValueError(f"{run_dir}: expected exactly one fault, found {len(faults)}")
    return CaseLabel(
        case_id=raw["case_id"],
        cause_code=faults[0]["cause_code"],
        simulator_fault=faults[0]["simulator_fault"],
    )
