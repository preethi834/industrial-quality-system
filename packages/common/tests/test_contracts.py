from __future__ import annotations

import json
import shutil
from pathlib import Path

from iqs_common.contracts import check_contracts, contracts_dir

HERE = Path(__file__).resolve()


def test_pinned_contracts_hold() -> None:
    assert check_contracts(contracts_dir(HERE)) == []


def test_schema_drift_is_reported(tmp_path: Path) -> None:
    src = contracts_dir(HERE)
    for f in src.iterdir():
        shutil.copy(f, tmp_path / f.name)
    schema = json.loads((tmp_path / "case_confirmed.schema.json").read_text())
    schema["title"] = "Changed"
    (tmp_path / "case_confirmed.schema.json").write_text(json.dumps(schema))
    problems = check_contracts(tmp_path)
    assert problems == ["case_confirmed: live schema differs from the pinned schema"]
