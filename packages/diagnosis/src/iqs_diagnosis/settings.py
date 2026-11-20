"""Diagnosis service settings (``IQS_DIAGNOSIS__*`` on top of the common settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from iqs_common.settings import CommonSettings


class DiagnosisSection(BaseModel):
    # Directory that holds simulator run directories (run-<hash>/observed, cases/).
    # POST /v1/cases registers runs from here only.
    runs_root: Path | None = None
    # Case library (past cases with confirmed causes); without it the agent is off.
    library_path: Path | None = None
    ranker_params_path: Path = Path("configs/diagnosis/ranker_params.json")
    vocabulary_path: Path = Path("configs/diagnosis/symptoms.yaml")
    # Frozen calibration run (step 4). When set, its scorer parameters replace
    # ranker_params_path so the served scores are the calibrated ones.
    calibration_run_path: Path | None = None
    # APS sets are shown; the LAC-empty flag marks "uncertain / possible new problem"
    # (user decision 2026-11-20, see docs/notes/step-4.md).
    cause_set_variant: Literal["lac", "aps", "lac_pac", "aps_pac"] = "aps"
    # memory: cases, checkpoints and tickets live in the process (demo, tests).
    # postgres: diagnosis.* tables, LangGraph Postgres checkpointer, qms.* tickets.
    store: Literal["memory", "postgres"] = "memory"


class DiagnosisSettings(CommonSettings):
    diagnosis: DiagnosisSection = DiagnosisSection()


@lru_cache(maxsize=1)
def get_diagnosis_settings() -> DiagnosisSettings:
    return DiagnosisSettings()
