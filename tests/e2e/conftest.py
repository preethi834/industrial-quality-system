"""End-to-end stack for the step-7 quality loop (``make test-e2e``; marker ``e2e``).

The session fixture

1. prepares the **world** on the host (``.e2e-work/``, git-ignored): simulator runs of
   ``tool_wear_m1`` (one per repeat, different seeds), SYNTHETIC image-to-part linkage
   (``iqs_sim.linker``, toy stripe images), a tiny PatchCore per repeat (random-init
   ResNet-18, no weight download), and a case library from the reference scenarios;
2. copies only each run's ``observed/`` into the directory mounted into diagnosis
   (ground truth and the linker manifest stay outside every container);
3. starts Postgres + Neo4j under the compose project ``iqs-e2e`` (throwaway credentials,
   own ports and volumes), applies the migrations from the host, then the two APIs and the
   two workers (base + loop + e2e compose files);
4. removes the stack and its volumes afterwards (``IQS_E2E_KEEP=1`` keeps it for debugging).

Images are not built here: run ``make build`` first (the test checks they exist).
The harness may read ground truth (it plays the world and the oracle engineer); the
services never can.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("IQS_E2E_WORK", REPO_ROOT / ".e2e-work"))
PROJECT = "iqs-e2e"
COMPOSE_FILES = (
    "deploy/compose/docker-compose.yml",
    "deploy/compose/docker-compose.loop.yml",
    "deploy/compose/docker-compose.e2e.yml",
)
SCENARIO = REPO_ROOT / "configs/simulator/scenarios/tool_wear_m1.yaml"
SCENARIOS = REPO_ROOT / "configs/simulator/scenarios"
TAXONOMY = REPO_ROOT / "configs/taxonomy/causes.yaml"
VOCABULARY = REPO_ROOT / "configs/diagnosis/symptoms.yaml"
LINKAGE = REPO_ROOT / "configs/simulator/image_linkage.yaml"
CALIBRATION = "configs/diagnosis/calibration/cal_dx_corpus_v2_dev_a010.json"
# Kept out of the library so the agent can meet them as new problems (as agent_fixtures).
NOVEL_SCENARIOS = ("drive_fault_m1", "handling_error_shift_b")
PORTS = {"pg": 15433, "bolt": 17687, "neo4j_http": 17474, "vision": 18001, "diagnosis": 18002}


@dataclass
class WorldRun:
    """One repeat's world: what only the harness may know is kept here."""

    repeat: int
    category: str
    run_id: str
    seed: int
    world_dir: Path  # full run incl. ground_truth/ and linkage/ (never mounted)
    pool: Path  # image pool for this category
    manifest: pd.DataFrame
    true_cause: str
    onset: pd.Timestamp


@dataclass
class Stack:
    env: dict[str, str]
    api_key: str
    vision: httpx.Client
    diagnosis: httpx.Client
    db_url: str
    neo4j_auth: tuple[str, str]
    bolt_uri: str
    runs: list[WorldRun] = field(default_factory=list)

    def compose(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return _compose(self.env, *args, check=check)


def _compose(
    env: dict[str, str], *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "--project-directory", str(REPO_ROOT), "-p", PROJECT]
    for f in COMPOSE_FILES:
        cmd += ["-f", str(REPO_ROOT / f)]
    return subprocess.run(
        [*cmd, *args], env=env, cwd=REPO_ROOT, check=check, capture_output=True, text=True
    )


def _repeats() -> int:
    return int(os.environ.get("IQS_E2E_REPEATS", "5"))


# ------------------------------------------------------------------ world (host side)


def _library(path: Path, root: Path) -> int:
    """Case library from the reference scenarios (simulator ground-truth causes)."""
    from iqs_common.taxonomy import load_taxonomy
    from iqs_diagnosis.evaluation.labels import load_case_label
    from iqs_diagnosis.ingestion.simulator import load_case
    from iqs_diagnosis.retrieval.library import LibraryRecord, write_library_file
    from iqs_diagnosis.symptoms import extract_profile
    from iqs_diagnosis.symptoms.vocabulary import load_vocabulary
    from iqs_sim.pipeline import run_scenario
    from iqs_sim.scenarios import load_scenario

    taxonomy = load_taxonomy(TAXONOMY)
    vocab = load_vocabulary(VOCABULARY)
    records = []
    for sc in sorted(SCENARIOS.glob("*.yaml")):
        if sc.stem in NOVEL_SCENARIOS:
            continue
        run = root / sc.stem
        run_scenario(load_scenario(sc), run, taxonomy)
        case, label = load_case(run), load_case_label(run)
        if case is None or label is None:
            continue
        records.append(
            LibraryRecord(
                profile=extract_profile(case, vocab),
                cause_code=label.cause_code,
                source="simulator",
                confirmed_by="simulator_ground_truth",
            )
        )
    write_library_file(path, records)
    return len(records)


def _world(work: Path, repeats: int) -> tuple[list[WorldRun], dict[str, Any]]:
    from iqs_common.taxonomy import load_taxonomy
    from iqs_sim.linker import LinkageConfig, read_manifest, write_linkage
    from iqs_sim.pipeline import run_scenario
    from iqs_sim.scenarios import load_scenario
    from iqs_vision.testing import make_tiny_model

    taxonomy = load_taxonomy(TAXONOMY)
    base = LinkageConfig.load(LINKAGE)
    toy = base.categories["toy"]
    t0 = time.perf_counter()
    n_library = _library(work / "loop/library/library.jsonl", work / "library_runs")
    runs: list[WorldRun] = []
    scenario = load_scenario(SCENARIO)
    for r in range(repeats):
        category = "toy" if r == 0 else f"toy{r + 1}"
        run_id = f"run-e2e{r:09x}"
        seed = scenario.seed + r
        world = work / "world" / run_id
        run_scenario(scenario.with_overrides(seed=seed), world, taxonomy)
        pool = work / "images"
        make_tiny_model(work / "loop/models/patchcore", pool, category)
        cfg = base.model_copy(update={"categories": {category: toy}})
        write_linkage(world, pool, category, cfg)
        shutil.copytree(world / "observed", work / "loop/runs" / run_id / "observed")
        faults = pd.read_parquet(world / "ground_truth" / "faults.parquet")
        runs.append(
            WorldRun(
                repeat=r,
                category=category,
                run_id=run_id,
                seed=seed,
                world_dir=world,
                pool=pool,
                manifest=read_manifest(world),
                true_cause=str(faults["cause_code"].iloc[0]),
                onset=pd.Timestamp(faults["onset_ts"].iloc[0]),
            )
        )
    info = {"library_cases": n_library, "prepare_s": round(time.perf_counter() - t0, 1)}
    return runs, info


# ------------------------------------------------------------------ stack


def _wait_ready(client: httpx.Client, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    last: object = None
    while time.monotonic() < deadline:
        try:
            r = client.get("/ready")
            if r.status_code == 200:
                return
            last = r.status_code
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(1)
    raise TimeoutError(f"{client.base_url} not ready: {last}")


def _migrate(env: dict[str, str]) -> None:
    menv = {k: v for k, v in env.items() if not k.startswith("IQS_")}
    menv.update(
        IQS_DATABASE__HOST="127.0.0.1",
        IQS_DATABASE__PORT=env["IQS_DATABASE__PORT"],
        IQS_DATABASE__NAME=env["IQS_DATABASE__NAME"],
        IQS_DATABASE__USER=env["IQS_DATABASE__USER"],
        IQS_DATABASE__PASSWORD=env["IQS_DATABASE__PASSWORD"],
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "alembic",
            "-c",
            "infrastructure/database/alembic.ini",
            "upgrade",
            "head",
        ],
        env=menv,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _check_images() -> None:
    for image in ("iqs-vision:dev", "iqs-diagnosis:dev"):
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
        if r.returncode != 0:
            pytest.fail(f"image {image} missing: run `make build` first")


@pytest.fixture(scope="session")
def stack() -> Iterator[Stack]:
    _check_images()
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "loop").mkdir(parents=True)
    runs, info = _world(WORK, _repeats())
    # Containers run as uid 10001: the mounted tree must be world-readable.
    for p in [WORK, *WORK.joinpath("loop").rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)

    api_key = secrets.token_hex(16)
    pg_pw, neo_pw = secrets.token_hex(12), secrets.token_hex(12)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IQS_", "COMPOSE_"))}
    env.update(
        IQS_DATABASE__NAME="iqs_e2e",
        IQS_DATABASE__USER="iqs_e2e",
        IQS_DATABASE__PASSWORD=pg_pw,
        IQS_DATABASE__PORT=str(PORTS["pg"]),
        IQS_NEO4J__USER="neo4j",
        IQS_NEO4J__PASSWORD=neo_pw,
        IQS_API__KEYS=api_key,
        IQS_LOOP_DIR=str(WORK / "loop"),
        IQS_LOOP_MODELS_DIR=str(WORK / "loop/models"),
        IQS_LOOP_CALIBRATION_RUN=CALIBRATION,
        IQS_NEO4J_BOLT_HOST_PORT=str(PORTS["bolt"]),
        IQS_NEO4J_HTTP_HOST_PORT=str(PORTS["neo4j_http"]),
        IQS_VISION_HOST_PORT=str(PORTS["vision"]),
        IQS_DIAGNOSIS_HOST_PORT=str(PORTS["diagnosis"]),
        COMPOSE_PROFILES="apps",
    )
    _compose(env, "down", "-v", "--remove-orphans", check=False)  # leftovers of a crashed run
    t0 = time.perf_counter()
    headers = {"X-API-Key": api_key}
    vision = httpx.Client(
        base_url=f"http://127.0.0.1:{PORTS['vision']}", headers=headers, timeout=60
    )
    diagnosis = httpx.Client(
        base_url=f"http://127.0.0.1:{PORTS['diagnosis']}", headers=headers, timeout=120
    )
    try:
        _compose(env, "up", "-d", "--wait", "postgres", "neo4j")
        _migrate(env)
        _compose(env, "up", "-d", "--wait", "vision", "diagnosis")
        _compose(env, "up", "-d", "--wait", "vision-worker", "diagnosis-worker")
        _wait_ready(vision)
        _wait_ready(diagnosis)
        info["stack_up_s"] = round(time.perf_counter() - t0, 1)
        (WORK / "stack_info.json").write_text(json.dumps(info))
        yield Stack(
            env=env,
            api_key=api_key,
            vision=vision,
            diagnosis=diagnosis,
            db_url=(f"postgresql+psycopg://iqs_e2e:{pg_pw}@127.0.0.1:{PORTS['pg']}/iqs_e2e"),
            neo4j_auth=("neo4j", neo_pw),
            bolt_uri=f"bolt://127.0.0.1:{PORTS['bolt']}",
            runs=runs,
        )
    except subprocess.CalledProcessError as exc:
        logs = _compose(env, "logs", "--tail", "80", check=False).stdout
        pytest.fail(f"stack failed: {exc.cmd}\n{exc.stderr}\n{logs}")
    finally:
        vision.close()
        diagnosis.close()
        if os.environ.get("IQS_E2E_KEEP") != "1":
            (WORK / "compose_logs.txt").write_text(
                _compose(env, "logs", "--no-color", check=False).stdout
            )
            _compose(env, "down", "-v", "--remove-orphans", check=False)
