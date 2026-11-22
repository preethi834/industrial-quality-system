"""Write the pinned JSON Schemas of the cross-service events to tests/contracts/."""

from __future__ import annotations

import json
from pathlib import Path

from iqs_common.events import CaseConfirmedEvent, NewDefectClusterEvent

OUT = Path(__file__).resolve().parents[1] / "tests" / "contracts"
MODELS = {"new_defect_cluster": NewDefectClusterEvent, "case_confirmed": CaseConfirmedEvent}


def schema_text(name: str) -> str:
    return json.dumps(MODELS[name].model_json_schema(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    for name in MODELS:
        (OUT / f"{name}.schema.json").write_text(schema_text(name), encoding="utf-8")
        print(f"wrote {name}.schema.json")


if __name__ == "__main__":
    main()
