"""Simulator case corpus: symptom profiles (agent side) and labels (evaluation side).

Layout under ``<corpus dir>``::

    families/<split>/   scenario YAMLs + manifest.json (family_sha256)
    runs/<split>/       run-<hash>/{observed,ground_truth,cases} + runs_index.json
    profiles/<split>.jsonl   one SymptomProfile per case, sorted by case_id
    labels/<split>.jsonl     one CaseLabel per case, sorted by case_id (evaluation only)

Profiles and labels are separate files so that the agent can be handed profiles
without labels; they are joined only by ``case_id``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iqs_diagnosis.domain.case import SymptomProfile
from iqs_diagnosis.evaluation.labels import CaseLabel, load_case_label
from iqs_diagnosis.ingestion.simulator import iter_run_dirs, load_case
from iqs_diagnosis.symptoms import extract_profile
from iqs_diagnosis.symptoms.vocabulary import Vocabulary

SPLITS: Final = ("train", "dev", "test")
M = TypeVar("M", bound=BaseModel)


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str = Field(min_length=1)
    seed: int
    n_per_fault: int = Field(gt=0)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    start: datetime
    duration_hours: float = Field(gt=0)
    novel_faults: tuple[str, ...] = ()
    splits: dict[str, SplitConfig]

    @model_validator(mode="after")
    def _check_splits(self) -> CorpusConfig:
        if set(self.splits) != set(SPLITS):
            raise ValueError(f"splits must be exactly {SPLITS}, got {sorted(self.splits)}")
        families = [s.family for s in self.splits.values()]
        seeds = [s.seed for s in self.splits.values()]
        if len(set(families)) != len(families) or len(set(seeds)) != len(seeds):
            raise ValueError("every split needs its own family name and seed")
        return self


def load_corpus_config(path: Path) -> CorpusConfig:
    with path.open(encoding="utf-8") as fh:
        return CorpusConfig.model_validate(yaml.safe_load(fh))


def build_split(
    runs_dir: Path, vocabulary: Vocabulary
) -> tuple[list[SymptomProfile], list[CaseLabel]]:
    """Profiles and labels for every run below ``runs_dir`` that raised a case."""
    profiles: list[SymptomProfile] = []
    labels: list[CaseLabel] = []
    for run_dir in iter_run_dirs(runs_dir):
        case = load_case(run_dir)
        if case is None:
            continue
        label = load_case_label(run_dir)
        if label is None or label.case_id != case.case_id:
            raise ValueError(f"{run_dir}: case file and ground-truth label disagree")
        profiles.append(extract_profile(case, vocabulary))
        labels.append(label)
    order = sorted(range(len(profiles)), key=lambda i: profiles[i].case_id)
    return [profiles[i] for i in order], [labels[i] for i in order]


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(m.model_dump(mode="json"), sort_keys=True) for m in items]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    with path.open(encoding="utf-8") as fh:
        return [model.model_validate_json(line) for line in fh if line.strip()]


def load_split(corpus_dir: Path, split: str) -> tuple[list[SymptomProfile], list[CaseLabel]]:
    profiles = read_jsonl(corpus_dir / "profiles" / f"{split}.jsonl", SymptomProfile)
    labels = read_jsonl(corpus_dir / "labels" / f"{split}.jsonl", CaseLabel)
    if [p.case_id for p in profiles] != [lab.case_id for lab in labels]:
        raise ValueError(f"{split}: profiles and labels are not aligned by case_id")
    return profiles, labels


def counts_by_fault(labels: Sequence[CaseLabel]) -> dict[str, int]:
    out: dict[str, int] = {}
    for lab in labels:
        out[lab.simulator_fault] = out.get(lab.simulator_fault, 0) + 1
    return dict(sorted(out.items()))
