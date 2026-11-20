"""Case library file: past cases with their confirmed causes (JSONL).

The agent reads this file, never a labels file. For the simulator corpus the library
is written from the train split by ``iqs_diagnosis.evaluation.corpus.write_library``;
in production it grows from engineer confirmations (feedback loop, step 8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from iqs_diagnosis.domain.case import SymptomProfile
from iqs_diagnosis.retrieval.base import LibraryCase


class LibraryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: SymptomProfile
    cause_code: str
    source: str  # simulator / engineer
    confirmed_by: str  # simulator_ground_truth / <engineer id>


def write_library_file(path: Path, records: Iterable[LibraryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.model_dump(mode="json"), sort_keys=True) for r in records]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def load_library(path: Path) -> list[LibraryCase]:
    with path.open(encoding="utf-8") as fh:
        records = [LibraryRecord.model_validate_json(line) for line in fh if line.strip()]
    return [LibraryCase(profile=r.profile, cause_code=r.cause_code) for r in records]
