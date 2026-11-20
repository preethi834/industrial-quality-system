"""Test isolation: no IQS_* variables, no .env, logging restored after each test."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from iqs_common.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("IQS_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    get_settings.cache_clear()
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    structlog.reset_defaults()


# --------------------------------------------------------------------------------------
# Simulator runs for symptom / corpus tests (step 3a). The simulator is a workspace
# package but not a dependency of iqs_diagnosis, so these fixtures skip without it.
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "configs" / "simulator" / "scenarios"
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy" / "causes.yaml"
VOCABULARY_PATH = REPO_ROOT / "configs" / "diagnosis" / "symptoms.yaml"
CORPUS_CONFIG_PATH = REPO_ROOT / "configs" / "diagnosis" / "sim_corpus.yaml"


@pytest.fixture(scope="session")
def reference_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """All reference scenarios (full 72 h: slow faults alarm late), simulated once."""
    pytest.importorskip("iqs_sim")
    from iqs_common.taxonomy import load_taxonomy
    from iqs_sim.pipeline import run_scenario
    from iqs_sim.scenarios import load_scenario

    root = tmp_path_factory.mktemp("reference_runs")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    runs: dict[str, Path] = {}
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        run_scenario(load_scenario(path), root / path.stem, taxonomy)
        runs[path.stem] = root / path.stem
    return runs
