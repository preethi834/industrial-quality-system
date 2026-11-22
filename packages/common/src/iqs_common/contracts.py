"""Event-contract checks shared by both services' test suites (step 7).

The pinned schemas live in ``tests/contracts/`` at the repository root. Each service calls
:func:`check_contracts` from its own tests, so a contract change fails in both.
"""

from __future__ import annotations

import json
from pathlib import Path

from iqs_common.events import CaseConfirmedEvent, NewDefectClusterEvent, parse_event

CONTRACT_MODELS = {
    "new_defect_cluster": NewDefectClusterEvent,
    "case_confirmed": CaseConfirmedEvent,
}


def contracts_dir(start: Path) -> Path:
    """``tests/contracts`` of the repository that contains ``start``."""
    for parent in [start, *start.parents]:
        candidate = parent / "tests" / "contracts"
        if (candidate / "README.md").is_file():
            return candidate
    raise FileNotFoundError("tests/contracts not found above " + str(start))


def check_contracts(directory: Path) -> list[str]:
    """Problems found (empty = contracts hold): schema drift or an example that fails."""
    problems: list[str] = []
    for name, model in CONTRACT_MODELS.items():
        pinned = json.loads((directory / f"{name}.schema.json").read_text(encoding="utf-8"))
        if pinned != model.model_json_schema():
            problems.append(f"{name}: live schema differs from the pinned schema")
        example = json.loads((directory / f"{name}.example.json").read_text(encoding="utf-8"))
        try:
            event = parse_event(example)
        except ValueError as exc:
            problems.append(f"{name}: example does not parse ({exc})")
            continue
        if event.type.value != name:
            problems.append(f"{name}: example parsed as {event.type.value}")
    return problems
