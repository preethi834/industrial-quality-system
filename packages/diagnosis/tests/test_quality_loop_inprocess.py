"""Step 7 core loop, in one process, no Docker (the e2e test against compose is 7g).

    simulator (tool_wear_m1) -> linker (SYNTHETIC image-to-part linkage, toy images)
    -> tiny real PatchCore (random-init ResNet-18) -> discovery trigger -> outbox
    -> diagnosis consumer -> exactly one case -> agent run -> engineer confirms
    -> feedback (counts, candidate pool, graph, case_confirmed) -> cluster named
    -> later images of the same defect are a known class: no second case.

What this shows: the plumbing and the idempotency rules work together on simulator data.
What it does not show: detection quality (random-init features on synthetic stripes).
The "engineer" confirms the injected cause read from ground truth (an oracle engineer); only
this test harness reads ground truth, never a service.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from iqs_common.events import EventType
from iqs_common.outbox_worker import InMemoryOutbox
from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.conformal.calibration import CalibrationRun, ConformalPredictor
from iqs_diagnosis.feedback.store import InMemoryFeedbackStore
from iqs_diagnosis.feedback.update import FeedbackService
from iqs_diagnosis.graph.repository import InMemoryGraph
from iqs_diagnosis.worker.consumer import VisionCaseConsumer

from .agent_fixtures import make_agent
from .conftest import REPO_ROOT, TAXONOMY_PATH

pytest.importorskip("iqs_vision")
pytest.importorskip("iqs_sim")

from iqs_sim.linker import LinkageConfig, read_manifest, write_linkage  # noqa: E402
from iqs_vision.discovery.consumer import make_case_confirmed_handler  # noqa: E402
from iqs_vision.discovery.service import (  # noqa: E402
    DiscoveryService,
    PartContext,
    PatchCoreScorer,
)
from iqs_vision.discovery.store import ClusterStatus, InMemoryDiscoveryStore, Routing  # noqa: E402
from iqs_vision.discovery.trigger import TriggerParams  # noqa: E402
from iqs_vision.inference.registry import ModelRegistry  # noqa: E402
from iqs_vision.testing import make_tiny_model  # noqa: E402

RUN = "run-0123456789ab"
CALIBRATION = REPO_ROOT / "configs" / "diagnosis" / "calibration" / "cal_dx_corpus_v2_dev_a010.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_core_quality_loop_in_process(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any], tmp_path: Path
) -> None:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    # World: simulated run + synthetic image linkage (the world may read ground truth).
    runs_root = tmp_path / "runs"
    run_dir = runs_root / RUN
    shutil.copytree(reference_runs["tool_wear_m1"], run_dir)
    pool = tmp_path / "images"
    models = tmp_path / "models"
    make_tiny_model(models, pool, "toy")
    write_linkage(
        run_dir, pool, "toy", LinkageConfig.load(REPO_ROOT / "configs/simulator/image_linkage.yaml")
    )
    manifest = read_manifest(run_dir)
    faults = pd.read_parquet(run_dir / "ground_truth" / "faults.parquet")
    true_cause = str(faults["cause_code"].iloc[0])
    onset = pd.Timestamp(faults["onset_ts"].iloc[0])

    # One shared outbox (platform.events) for both services.
    outbox = InMemoryOutbox()
    vstore = InMemoryDiscoveryStore(outbox)
    vision = DiscoveryService(
        PatchCoreScorer(ModelRegistry(models, "deploy"), 10),
        vstore,
        TriggerParams.load(REPO_ROOT / "configs" / "vision" / "discovery.yaml"),
    )
    graph = InMemoryGraph()
    feedback = FeedbackService(InMemoryFeedbackStore(outbox), taxonomy, graph)
    calibration_sha = _sha(CALIBRATION)
    conformal = ConformalPredictor(CalibrationRun.load(CALIBRATION), "aps")
    agent, repo, tickets = make_agent(reference_profiles, conformal=conformal, feedback=feedback)
    consumer = VisionCaseConsumer(agent, repo, runs_root, 12.0)
    name_cluster = make_case_confirmed_handler(vstore)

    def diagnosis_worker_tick() -> None:
        outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer)

    def vision_worker_tick() -> None:
        outbox.process_pending({EventType.CASE_CONFIRMED}, name_cluster)

    first_anomalous: tuple[int, float] | None = None
    alert_at: int | None = None
    case_opened_wall: float | None = None
    confirmed = False
    routings: list[tuple[int, str, str]] = []  # (seq, shown_as, routing)
    t_start = time.perf_counter()
    for row in manifest.itertuples():
        img = Image.open(pool / row.image).convert("RGB")
        ctx = PartContext(row.part_id, RUN, "L1", pd.Timestamp(row.inspected_ts).to_pydatetime())
        r = vision.route_observation(img, "toy", ctx)
        routing = r.observation.routing.value
        routings.append((row.seq, row.shown_as, routing))
        if r.observation.decision == "anomalous" and first_anomalous is None:
            first_anomalous = (row.seq, time.perf_counter())
        if r.alert_event_id is not None:
            assert alert_at is None, "a second cluster alert"
            alert_at = row.seq
        diagnosis_worker_tick()
        cases = repo.list_cases(source="vision")
        if cases and case_opened_wall is None:
            case_opened_wall = time.perf_counter()
        if cases and not confirmed:
            # Engineer confirms (oracle: the injected cause) once the case waits.
            ((case, _),) = cases
            assert agent.phase(case.case_id) == "awaiting_confirmation"
            state = agent.state(case.case_id)
            assert state["cause_set"]["guarantee_applies"] is False  # vision != simulator
            out = asyncio.run(agent.confirm(case.case_id, true_cause, "engineer-1", "scratch"))
            assert out["feedback"]["graph"]["status"] == "written"
            confirmed = True
            vision_worker_tick()
        # Enough evidence that the class is known: 20 defect images after naming.
        if confirmed and sum(1 for _, sh, rt in routings if sh == "defect" and rt == "known") >= 20:
            break
    t_total = time.perf_counter() - t_start

    # Exactly one alert, one case, confirmed; nothing failed in the outbox.
    assert alert_at is not None and confirmed
    assert len(outbox.rows(EventType.NEW_DEFECT_CLUSTER)) == 1
    assert len(outbox.rows(EventType.CASE_CONFIRMED)) == 1
    assert {r.status for r in outbox.rows()} == {"processed"}
    [(case, status)] = repo.list_cases(source="vision")
    assert status.value == "confirmed" and agent.phase(case.case_id) == "done"
    ticket = tickets.get(agent.state(case.case_id)["ticket"]["ticket_id"])
    assert ticket.status == "cause_confirmed"

    # Feedback landed: graph edge, count 0 -> 1, candidate pool; calibration untouched.
    assert graph.edge("Case", case.case_id, "CONFIRMED_CAUSE", "Cause", true_cause)
    assert graph.edge("DefectType", "toy/scratch", "INDICATES", "Cause", true_cause) == {"count": 1}
    assert feedback.store.count("defect_type:toy/scratch", true_cause) == 1
    assert [c["case_id"] for c in feedback.store.candidates()] == [case.case_id]
    assert _sha(CALIBRATION) == calibration_sha

    # Vision learned the class: the cluster is named, later scratches are known.
    [cluster] = vstore.clusters("toy")
    assert cluster.status == ClusterStatus.NAMED and cluster.defect_class_name == "scratch"
    assert cluster.cause_code == true_cause
    named_seq = next(seq for seq, _, rt in routings if rt == Routing.KNOWN.value)
    later_defects = [rt for seq, shown, rt in routings if seq >= named_seq and shown == "defect"]
    assert len(later_defects) >= 20
    assert later_defects.count(Routing.KNOWN.value) / len(later_defects) >= 0.9
    # Good parts were never anomalous (training goods, see image_linkage.yaml).
    assert all(rt == Routing.NORMAL.value for _, shown, rt in routings if shown == "good")

    # Recorded (simulator + synthetic linkage + random-init detector; not a quality result).
    after_onset = manifest[manifest["inspected_ts"] >= onset]
    assert first_anomalous is not None and case_opened_wall is not None
    n_anomalous = sum(1 for s, _, rt in routings if s <= alert_at and rt != "normal")
    print(
        f"\n[step-7 in-process loop] parts_streamed={len(routings)}/{len(manifest)} "
        f"images_after_onset_to_alert={int((after_onset['seq'] <= alert_at).sum())} "
        f"anomalous_before_alert={n_anomalous} "
        f"latency_first_anomalous_to_case_s={case_opened_wall - first_anomalous[1]:.2f} "
        f"total_s={t_total:.1f}"
    )
