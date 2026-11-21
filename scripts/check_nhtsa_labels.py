#!/usr/bin/env python
"""Check the NHTSA labelling sheet and show progress (run as often as you like).

Usage::

    uv run python scripts/check_nhtsa_labels.py [eval_sets/diagnosis/nhtsa/recalls_v1.csv]

Exit code 1 if any row breaks a rule in LABELLING_GUIDE.md (every problem is listed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.evaluation.nhtsa_labels import check_sheet

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    csv_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO_ROOT / "eval_sets/diagnosis/nhtsa/recalls_v1.csv"
    )
    res = check_sheet(
        csv_path,
        csv_path.with_suffix(".manifest.json"),
        load_taxonomy(REPO_ROOT / "configs/taxonomy/causes.yaml"),
    )
    print(json.dumps(res.summary(), indent=2, sort_keys=True))
    for e in res.errors:
        print("ERROR", e)
    return 1 if res.errors else 0


if __name__ == "__main__":
    sys.exit(main())
