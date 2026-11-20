"""Case loading, labels and the corpus builder (profiles and labels kept apart)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from iqs_diagnosis.evaluation import labels as labels_mod
from iqs_diagnosis.evaluation.corpus import (
    CorpusConfig,
    build_split,
    load_corpus_config,
    load_split,
    write_jsonl,
)
from iqs_diagnosis.evaluation.labels import load_case_label
from iqs_diagnosis.ingestion import simulator as ingest
from iqs_diagnosis.ingestion.simulator import iter_run_dirs, load_case
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary

from .conftest import CORPUS_CONFIG_PATH, VOCABULARY_PATH


def test_layout_names_match_simulator() -> None:
    sim_export = pytest.importorskip("iqs_sim.export")
    from iqs_sim.cases import CASE_FILE
    from iqs_sim.pipeline import CASE_LABELS_FILE

    assert ingest.CASES_DIR == sim_export.CASES_DIR
    assert ingest.OBSERVED_DIR == sim_export.OBSERVED_DIR
    assert ingest.CASE_FILE == CASE_FILE
    assert labels_mod.GROUND_TRUTH_DIR == sim_export.GROUND_TRUTH_DIR
    assert labels_mod.CASE_LABELS_FILE == CASE_LABELS_FILE


def test_load_case_fields(reference_runs: dict[str, Path]) -> None:
    case = load_case(reference_runs["coolant_loss_m3"])
    assert case is not None
    assert case.source == "simulator" and case.case_id.startswith("SIM-")
    assert case.observed_dir == (reference_runs["coolant_loss_m3"] / "observed").resolve()
    assert case.window_start is not None and case.alarm_ts is not None
    assert case.window_start < case.alarm_ts
    assert case.baseline_hours == 12.0


def test_label_matches_case(reference_runs: dict[str, Path]) -> None:
    run = reference_runs["bad_batch_b0008"]
    label = load_case_label(run)
    case = load_case(run)
    assert label is not None and case is not None
    assert label.case_id == case.case_id
    assert label.cause_code == "MATERIAL.BAD_BATCH"
    assert load_case_label(reference_runs["baseline"]) is None


def test_multi_fault_label_rejected(reference_runs: dict[str, Path], tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(reference_runs["tool_wear_m1"] / "ground_truth", run / "ground_truth")
    path = run / "ground_truth" / "case_labels.json"
    raw = json.loads(path.read_text())
    raw["faults"] = raw["faults"] * 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exactly one fault"):
        load_case_label(run)


def test_build_split_aligned_and_sorted(reference_runs: dict[str, Path], tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for name in ("tool_wear_m1", "drive_fault_m1", "baseline"):
        shutil.copytree(reference_runs[name], runs / name)
    vocab = load_vocabulary(VOCABULARY_PATH)
    profiles, labels = build_split(runs, vocab)
    assert len(profiles) == len(labels) == 2  # baseline raised no case
    ids = [p.case_id for p in profiles]
    assert ids == sorted(ids) == [lab.case_id for lab in labels]

    corpus = tmp_path / "corpus"
    write_jsonl(corpus / "profiles" / "dev.jsonl", profiles)
    write_jsonl(corpus / "labels" / "dev.jsonl", labels)
    assert load_split(corpus, "dev") == (profiles, labels)
    # Profiles file carries no labels.
    text = (corpus / "profiles" / "dev.jsonl").read_text()
    assert "cause_code" not in text and "MACHINE." not in text

    write_jsonl(corpus / "labels" / "dev.jsonl", list(reversed(labels)))
    with pytest.raises(ValueError, match="aligned"):
        load_split(corpus, "dev")


def test_iter_run_dirs(reference_runs: dict[str, Path]) -> None:
    root = next(iter(reference_runs.values())).parent
    assert {p.name for p in iter_run_dirs(root)} == set(reference_runs)


def test_corpus_config_is_valid_and_separated() -> None:
    cfg = load_corpus_config(CORPUS_CONFIG_PATH)
    assert set(cfg.splits) == {"train", "dev", "test"}
    taxonomy_faults = pytest.importorskip("iqs_sim.faults").FaultType
    assert set(cfg.novel_faults) <= {f.value for f in taxonomy_faults}


def test_corpus_config_rejects_shared_seed() -> None:
    raw = {
        "name": "x",
        "start": "2026-01-05T06:00:00Z",
        "duration_hours": 72,
        "splits": {
            "train": {"family": "a", "seed": 1, "n_per_fault": 1},
            "dev": {"family": "b", "seed": 1, "n_per_fault": 1},
            "test": {"family": "c", "seed": 3, "n_per_fault": 1},
        },
    }
    with pytest.raises(ValidationError, match="own family name and seed"):
        CorpusConfig.model_validate(raw)
