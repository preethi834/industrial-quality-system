"""Case labels from simulator ground truth: evaluation only.

This is the only module in ``iqs_diagnosis`` (besides the evaluation package that
calls it) that reads a run's ground-truth directory; a test enforces that. The agent
path (ingestion, symptoms, retrieval, scoring) never sees these labels for a query
case; the case library gets them only as *confirmed* causes of past cases.

**Primary-cause rule for runs with several faults** (labelling guide, simulator
section): the primary cause is the fault with the most *excess truly defective parts*,

    excess(f) = #(truly defective parts touched by f) - p0 * #(parts touched by f)

where ``p0`` is the defect rate of parts touched by no fault. Ties go to the fault
listed first. The other faults are recorded as secondary causes. This is a Pareto rule
on the defects each fault adds, computed from ground truth (``part_truth``) only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict

GROUND_TRUTH_DIR: Final = "ground_truth"
CASE_LABELS_FILE: Final = "case_labels.json"
PART_TRUTH_FILE: Final = "part_truth.parquet"


class CaseLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    cause_code: str  # primary cause
    simulator_fault: str
    secondary_causes: tuple[str, ...] = ()
    secondary_faults: tuple[str, ...] = ()
    rule: str = "single_fault"  # or "excess_defects"
    excess_defects: dict[str, float] = {}


def excess_defects(part_truth: pd.DataFrame, fault_ids: list[str]) -> dict[str, float]:
    touched = part_truth["fault_ids"].map(lambda ids: set(ids))
    clean = part_truth[touched.map(len) == 0]
    p0 = float(clean["truly_defective"].mean()) if len(clean) else 0.0
    out: dict[str, float] = {}
    for fid in fault_ids:
        mask = touched.map(lambda ids, fid=fid: fid in ids)
        n = int(mask.sum())
        k = int(part_truth.loc[mask, "truly_defective"].sum())
        out[fid] = round(k - p0 * n, 3)
    return out


def load_case_label(run_dir: Path) -> CaseLabel | None:
    """Label of the run's case; ``None`` if no case was raised."""
    truth = run_dir / GROUND_TRUTH_DIR
    raw = json.loads((truth / CASE_LABELS_FILE).read_text(encoding="utf-8"))
    if raw["case_id"] is None:
        return None
    faults = raw["faults"]
    if not faults:
        raise ValueError(f"{run_dir}: a case without an injected fault")
    if len(faults) == 1:
        return CaseLabel(
            case_id=raw["case_id"],
            cause_code=faults[0]["cause_code"],
            simulator_fault=faults[0]["simulator_fault"],
        )
    ids = [f["fault_id"] for f in faults]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{run_dir}: duplicate fault ids {ids}")
    part_truth = pd.read_parquet(truth / PART_TRUTH_FILE, columns=["fault_ids", "truly_defective"])
    excess = excess_defects(part_truth, ids)
    order = sorted(range(len(faults)), key=lambda i: (-excess[ids[i]], i))
    primary, rest = faults[order[0]], [faults[i] for i in order[1:]]
    return CaseLabel(
        case_id=raw["case_id"],
        cause_code=primary["cause_code"],
        simulator_fault=primary["simulator_fault"],
        secondary_causes=tuple(f["cause_code"] for f in rest),
        secondary_faults=tuple(f["simulator_fault"] for f in rest),
        rule="excess_defects",
        excess_defects={f["simulator_fault"]: excess[f["fault_id"]] for f in faults},
    )
