"""Step 7 quality loop against real Postgres 16 and Neo4j 5.26 (testcontainers).

- migration 0005: upgrade / downgrade to 0004 / upgrade, and the 0005 tables are in the
  model metadata, so the drift check (``alembic check``) really covers them;
- outbox: two workers with ``FOR UPDATE SKIP LOCKED`` claim disjoint rows; a failing handler
  increments ``attempts`` and parks the row as ``failed`` at the maximum;
- diagnosis consumer: duplicate delivery and a second event for the same cluster open one
  case; ``source_cluster_id`` UNIQUE is enforced by the database;
- feedback on Postgres + Neo4j: applied twice -> one count, one edge, one ``case_confirmed``
  event, ``INDICATES.count`` equals the Postgres count;
- vision: cluster row + outbox row in one transaction; ``case_confirmed`` names the cluster
  once (second delivery is a no-op).

Events here are hand-made test payloads (no images, no simulator run beyond an empty
``observed/`` directory); they test persistence and idempotency, nothing about detection.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from neo4j import GraphDatabase
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from iqs_common.db import Base, create_db_engine, create_session_factory
from iqs_common.events import (
    CaseConfirmedEvent,
    CaseConfirmedPayload,
    Event,
    EventType,
    NewDefectClusterEvent,
    NewDefectClusterPayload,
)
from iqs_common.outbox import OutboxEvent, OutboxStatus
from iqs_common.outbox_worker import SqlOutbox
from iqs_common.settings import DatabaseSettings, Neo4jSettings
from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.feedback.store import SqlFeedbackStore
from iqs_diagnosis.feedback.update import FeedbackService
from iqs_diagnosis.graph.repository import Neo4jGraphWriter
from iqs_diagnosis.ingestion.vision import case_from_cluster_event, vision_case_id
from iqs_diagnosis.persistence.repository import SqlCaseRepository
from iqs_diagnosis.worker.consumer import VisionCaseConsumer
from iqs_vision.discovery.consumer import make_case_confirmed_handler
from iqs_vision.discovery.store import (
    ClusterRecord,
    ClusterStatus,
    Observation,
    Routing,
    SqlDiscoveryStore,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = REPO_ROOT / "configs/taxonomy/causes.yaml"
RUN = "run-0123456789ab"
T0 = datetime(2026, 1, 6, 7, 26, tzinfo=UTC)

TABLES_0005 = {
    ("vision", "clusters"),
    ("vision", "discovery_observations"),
    ("vision", "processed_events"),
    ("diagnosis", "processed_events"),
    ("diagnosis", "cause_counts"),
    ("diagnosis", "calibration_candidates"),
    ("diagnosis", "feedback_applied"),
}


def _tables(engine: Engine) -> set[tuple[str, str]]:
    insp = inspect(engine)
    return {(s, t) for s in ("vision", "diagnosis") for t in insp.get_table_names(schema=s)}


@pytest.fixture
def factory(migrated: Config, pg_settings: DatabaseSettings) -> Iterator[sessionmaker[Session]]:
    engine = create_db_engine(pg_settings)
    yield create_session_factory(engine)
    engine.dispose()


def _cluster_event(cluster_id: uuid.UUID, size: int = 5) -> NewDefectClusterEvent:
    ids = tuple(uuid.uuid4() for _ in range(size))
    return NewDefectClusterEvent(
        payload=NewDefectClusterPayload(
            cluster_id=cluster_id,
            category="toy",
            size=size,
            first_seen=T0,
            last_seen=T0 + timedelta(hours=2),
            inspection_ids=ids,
            part_ids=tuple(f"P{900 + i:06d}" for i in range(size)),
            model_version="patchcore-toy-deploy-test",
            line_id="L1",
            simulator_run_id=RUN,
            linkage="synthetic_simulator",
            representative_inspection_id=ids[0],
            score_max=3.0,
            score_median=2.0,
        )
    )


# ------------------------------------------------------------------ migration 0005


def test_0005_upgrade_downgrade_upgrade(alembic_cfg: Config, pg_engine: Engine) -> None:
    command.upgrade(alembic_cfg, "head")
    assert TABLES_0005 <= _tables(pg_engine)
    cols = {c["name"] for c in inspect(pg_engine).get_columns("cases", schema="diagnosis")}
    assert "source_cluster_id" in cols
    command.downgrade(alembic_cfg, "0004")
    assert not (TABLES_0005 & _tables(pg_engine))
    cols = {c["name"] for c in inspect(pg_engine).get_columns("cases", schema="diagnosis")}
    assert "source_cluster_id" not in cols
    command.upgrade(alembic_cfg, "head")
    assert TABLES_0005 <= _tables(pg_engine)
    command.downgrade(alembic_cfg, "base")


def test_0006_image_sha256_upgrade_downgrade(alembic_cfg: Config, pg_engine: Engine) -> None:
    def cols() -> set[str]:
        return {
            c["name"]
            for c in inspect(pg_engine).get_columns("discovery_observations", schema="vision")
        }

    command.upgrade(alembic_cfg, "head")
    assert "image_sha256" in cols()
    command.downgrade(alembic_cfg, "0005")
    assert "image_sha256" not in cols() and "embedding" in cols()
    command.upgrade(alembic_cfg, "head")
    assert "image_sha256" in cols()
    command.downgrade(alembic_cfg, "base")


def test_0005_models_are_in_the_metadata_the_drift_check_uses(migrated: Config) -> None:
    """``test_models_and_migrations_have_no_drift`` compares Base.metadata with the
    database; it only covers 0005 if env.py registered the step-7 models."""
    env_src = (REPO_ROOT / "infrastructure/database/migrations/env.py").read_text("utf-8")
    assert "iqs_vision.discovery.store" in env_src
    assert "iqs_diagnosis.feedback.store" in env_src
    in_metadata = {(t.schema, t.name) for t in Base.metadata.tables.values()}
    assert TABLES_0005 <= in_metadata
    command.check(migrated)  # no drift between the 0005 models and the 0005 migration


def test_source_cluster_id_unique_in_the_database(
    factory: sessionmaker[Session], pg_engine: Engine
) -> None:
    sql = text(
        "INSERT INTO diagnosis.cases (id, source, title, description, symptoms, status, "
        "case_json, source_cluster_id) VALUES (:id, 'vision', 't', 'd', '[]', 'registered', "
        "'{}', 'c-1')"
    )
    with pg_engine.begin() as conn:
        conn.execute(sql, {"id": "case-a"})
    with pytest.raises(IntegrityError, match="uq_cases_source_cluster_id"):
        with pg_engine.begin() as conn:
            conn.execute(sql, {"id": "case-b"})


# ------------------------------------------------------------------ outbox worker


def test_two_workers_claim_disjoint_rows(factory: sessionmaker[Session]) -> None:
    outbox = SqlOutbox(factory)
    events = [_cluster_event(uuid.uuid4()) for _ in range(6)]
    for e in events:
        outbox.append(e)
    first_holds = threading.Event()
    release = threading.Event()
    seen: dict[str, list[uuid.UUID]] = {"a": [], "b": []}

    def slow(event: Event) -> None:
        seen["a"].append(event.event_id)
        first_holds.set()
        assert release.wait(30)

    def fast(event: Event) -> None:
        seen["b"].append(event.event_id)

    t = threading.Thread(
        target=lambda: outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, slow, limit=3)
    )
    t.start()
    assert first_holds.wait(30)  # worker A holds its 3 row locks
    rb = outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, fast, limit=10)
    release.set()
    t.join(30)
    assert len(rb.processed) == 3  # B skipped A's locked rows
    assert len(seen["a"]) == 3 and not set(seen["a"]) & set(seen["b"])
    with factory() as s:
        statuses = s.scalars(select(OutboxEvent.status)).all()
    assert statuses == [OutboxStatus.PROCESSED.value] * 6


def test_failing_handler_increments_attempts_then_fails(factory: sessionmaker[Session]) -> None:
    outbox = SqlOutbox(factory)
    event = _cluster_event(uuid.uuid4())
    outbox.append(event)

    def boom(_: Event) -> None:
        raise RuntimeError("handler down")

    for expected in (1, 2):
        r = outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, boom, max_attempts=3)
        assert r.retried == [event.event_id]
        with factory() as s:
            row = s.get(OutboxEvent, event.event_id)
            assert row is not None and (row.attempts, row.status) == (expected, "pending")
    r = outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, boom, max_attempts=3)
    assert r.failed == [event.event_id]
    with factory() as s:
        row = s.get(OutboxEvent, event.event_id)
        assert row is not None and (row.attempts, row.status) == (3, "failed")
    # A failed row is not claimed again.
    assert outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, boom).total == 0


# ------------------------------------------------------------------ diagnosis consumer


class _FakeAgent:
    """Counts agent runs; ``phase`` is ``not_started`` until run (as the real agent)."""

    def __init__(self) -> None:
        self.runs: list[str] = []

    def phase(self, case_id: str) -> str:
        return "awaiting_confirmation" if case_id in self.runs else "not_started"

    async def run(self, case: Any) -> dict[str, Any]:
        self.runs.append(case.case_id)
        return {"novel": False, "top3": []}


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    (tmp_path / "runs" / RUN / "observed").mkdir(parents=True)
    return tmp_path / "runs"


def test_consumer_on_postgres_opens_one_case(
    factory: sessionmaker[Session], runs_root: Path
) -> None:
    repo = SqlCaseRepository(factory)
    agent = _FakeAgent()
    consumer = VisionCaseConsumer(agent, repo, runs_root, 12.0)  # type: ignore[arg-type]
    outbox = SqlOutbox(factory)
    cluster_id = uuid.uuid4()
    event = _cluster_event(cluster_id)
    outbox.append(event)
    assert outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer).processed
    # Redelivery of the same event (lost acknowledgement) and a new event for the cluster.
    with factory.begin() as s:
        s.execute(
            text(
                "UPDATE platform.events SET status = 'pending', processed_at = NULL WHERE id = :id"
            ),
            {"id": event.event_id},
        )
    outbox.append(_cluster_event(cluster_id))
    r = outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer)
    assert len(r.processed) == 2 and not r.failed
    cases = repo.list_cases(source="vision")
    assert [c.case_id for c, _ in cases] == [vision_case_id(cluster_id)]
    assert agent.runs == [vision_case_id(cluster_id)]  # the agent ran once
    with factory() as s:
        n_processed = s.execute(text("SELECT count(*) FROM diagnosis.processed_events")).scalar()
        n_cases = s.execute(text("SELECT count(*) FROM diagnosis.cases")).scalar()
    assert (n_processed, n_cases) == (2, 1)  # both events recorded, one case


def test_event_for_missing_run_is_parked_as_failed(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    repo = SqlCaseRepository(factory)
    consumer = VisionCaseConsumer(_FakeAgent(), repo, tmp_path, 12.0)  # type: ignore[arg-type]
    outbox = SqlOutbox(factory)
    event = _cluster_event(uuid.uuid4())
    outbox.append(event)
    for _ in range(2):
        outbox.process_pending({EventType.NEW_DEFECT_CLUSTER}, consumer, max_attempts=2)
    with factory() as s:
        row = s.get(OutboxEvent, event.event_id)
        assert row is not None and row.status == "failed"
    assert repo.list_cases(source="vision") == []


# ------------------------------------------------------------------ feedback + Neo4j


@pytest.fixture
def neo4j_driver(neo4j_settings: Neo4jSettings) -> Iterator[Any]:
    auth = (neo4j_settings.user, neo4j_settings.password.get_secret_value())
    with GraphDatabase.driver(neo4j_settings.uri, auth=auth) as driver:
        driver.execute_query("MATCH (n) DETACH DELETE n")
        yield driver


def test_feedback_on_postgres_and_neo4j_is_applied_once(
    factory: sessionmaker[Session], runs_root: Path, neo4j_driver: Any
) -> None:
    taxonomy = load_taxonomy(TAXONOMY)
    cause = taxonomy.causes[0].code
    writer = Neo4jGraphWriter(neo4j_driver)
    writer.ensure_schema()
    feedback = FeedbackService(SqlFeedbackStore(factory), taxonomy, writer)
    case = case_from_cluster_event(_cluster_event(uuid.uuid4()).payload, runs_root, 12.0)

    first = feedback.apply(case, cause, "engineer-1", "scratch")
    again = feedback.apply(case, cause, "engineer-1", "scratch")
    assert first["first_application"] is True and again["first_application"] is False
    assert (first["count_before"], first["count_after"]) == (0, 1)
    assert first["graph"]["status"] == "written" and again["graph"]["status"] == "written"
    store = feedback.store
    assert store.count("defect_type:toy/scratch", cause) == 1
    assert [c["case_id"] for c in store.candidates()] == [case.case_id]
    with factory() as s:
        rows = s.scalars(
            select(OutboxEvent).where(OutboxEvent.type == EventType.CASE_CONFIRMED.value)
        ).all()
    assert len(rows) == 1 and rows[0].payload["payload"]["cause_code"] == cause

    records, _, _ = neo4j_driver.execute_query(
        "MATCH (c:Case {id: $id})-[r:CONFIRMED_CAUSE]->(k:Cause {code: $code}) "
        "RETURN count(r) AS n",
        id=case.case_id,
        code=cause,
    )
    assert records[0]["n"] == 1
    # Re-running the MERGE (e.g. a retry after a crash) still leaves one edge.
    writer.write_confirmation(_write_from(feedback, case.case_id, taxonomy))
    records, _, _ = neo4j_driver.execute_query(
        "MATCH (d:DefectType {key: 'toy/scratch'})-[i:INDICATES]->(k:Cause {code: $code}) "
        "RETURN count(i) AS n, collect(i.count) AS counts",
        code=cause,
    )
    assert records[0]["n"] == 1
    assert records[0]["counts"] == [store.count("defect_type:toy/scratch", cause)]
    records, _, _ = neo4j_driver.execute_query(
        "MATCH ()-[r:CONFIRMED_CAUSE]->() RETURN count(r) AS n"
    )
    assert records[0]["n"] == 1


def _write_from(feedback: FeedbackService, case_id: str, taxonomy: Any) -> Any:
    from iqs_diagnosis.graph.repository import ConfirmationWrite, DefectTypeRef

    rec = feedback.store.get(case_id)
    assert rec is not None and rec.cause_code is not None
    assert rec.category is not None and rec.defect_class_name is not None
    return ConfirmationWrite(
        case_id=rec.case_id,
        source=rec.source,
        cause_code=rec.cause_code,
        cause_name=rec.cause_code,
        engineer=rec.engineer,
        confirmed_at=rec.confirmed_at,
        defect_type=DefectTypeRef(rec.category, rec.defect_class_name),
        indicates_count=rec.count_after,
    )


def test_graph_write_retried_after_neo4j_was_down(
    factory: sessionmaker[Session], runs_root: Path, neo4j_driver: Any
) -> None:
    """Postgres feedback commits while Neo4j is unreachable; ``flush_graph`` catches up."""
    taxonomy = load_taxonomy(TAXONOMY)
    cause = taxonomy.causes[0].code
    dead = GraphDatabase.driver("bolt://127.0.0.1:1", auth=("neo4j", "x"))
    try:
        down = FeedbackService(SqlFeedbackStore(factory), taxonomy, Neo4jGraphWriter(dead))
        case = case_from_cluster_event(_cluster_event(uuid.uuid4()).payload, runs_root, 12.0)
        out = down.apply(case, cause, "engineer-1", "scratch")
    finally:
        dead.close()
    assert out["graph"]["status"] == "pending" and out["count_after"] == 1
    up = FeedbackService(SqlFeedbackStore(factory), taxonomy, Neo4jGraphWriter(neo4j_driver))
    assert up.flush_graph() == 1
    assert up.flush_graph() == 0
    records, _, _ = neo4j_driver.execute_query(
        "MATCH (:Case {id: $id})-[r:CONFIRMED_CAUSE]->() RETURN count(r) AS n", id=case.case_id
    )
    assert records[0]["n"] == 1


# ------------------------------------------------------------------ vision store + consumer


def _alert(store: SqlDiscoveryStore, n: int = 5) -> tuple[ClusterRecord, NewDefectClusterEvent]:
    rng = np.random.default_rng(0)
    ids = [uuid.uuid4() for _ in range(n)]
    for i, oid in enumerate(ids):
        store.add_observation(
            Observation(
                id=oid,
                category="toy",
                observed_at=T0 + timedelta(minutes=i),
                score=2.0,
                threshold=1.0,
                decision="anomalous",
                routing=Routing.UNKNOWN_BUFFERED,
                model_version="m",
                embedding=rng.normal(size=8).astype(np.float32),
                part_id=f"P{i:06d}",
                run_id=RUN,
            )
        )
    cluster_id = uuid.uuid4()
    event = _cluster_event(cluster_id, n)
    cluster = ClusterRecord(
        id=cluster_id,
        category="toy",
        status=ClusterStatus.ALERTED,
        member_count=n,
        centroid=np.zeros(8, dtype=np.float32),
        radius=0.1,
        first_seen=T0,
        last_seen=T0 + timedelta(minutes=n),
        representative_id=ids[0],
        alerted_event_id=event.event_id,
    )
    store.create_alerted_cluster(cluster, ids, event)
    return cluster, event


def test_alerted_cluster_and_outbox_row_commit_together(factory: sessionmaker[Session]) -> None:
    store = SqlDiscoveryStore(factory)
    cluster, event = _alert(store)
    assert store.get_cluster(cluster.id) is not None
    assert len(store.members(cluster.id)) == 5
    with factory() as s:
        row = s.get(OutboxEvent, event.event_id)
        assert row is not None and row.type == EventType.NEW_DEFECT_CLUSTER.value
    # Same alert again (e.g. a duplicate event id): the whole transaction rolls back.
    with pytest.raises(IntegrityError):
        store.create_alerted_cluster(dataclasses.replace(cluster, id=uuid.uuid4()), [], event)
    assert len(store.clusters("toy")) == 1


def test_case_confirmed_names_the_cluster_once(factory: sessionmaker[Session]) -> None:
    store = SqlDiscoveryStore(factory)
    cluster, _ = _alert(store)
    outbox = SqlOutbox(factory)
    confirmed = CaseConfirmedEvent(
        payload=CaseConfirmedPayload(
            case_id=uuid.UUID(vision_case_id(cluster.id)),
            cause_code="MACHINE.TOOL_WEAR",
            taxonomy_version="t",
            confirmed_by="engineer-1",
            confirmed_at=T0 + timedelta(hours=3),
            source_cluster_id=cluster.id,
            defect_class_name="scratch",
        )
    )
    outbox.append(confirmed)
    handler = make_case_confirmed_handler(store)
    assert outbox.process_pending({EventType.CASE_CONFIRMED}, handler).processed
    named = store.get_cluster(cluster.id)
    assert named is not None and named.status == ClusterStatus.NAMED
    assert (named.defect_class_name, named.cause_code) == ("scratch", "MACHINE.TOOL_WEAR")
    first_named_at = named.named_at
    # Second delivery: processed_events stops it; nothing changes.
    with factory.begin() as s:
        s.execute(
            text("UPDATE platform.events SET status = 'pending' WHERE id = :id"),
            {"id": confirmed.event_id},
        )
    assert outbox.process_pending({EventType.CASE_CONFIRMED}, handler).processed
    again = store.get_cluster(cluster.id)
    assert again is not None and again.named_at == first_named_at
    with factory() as s:
        n = s.execute(text("SELECT count(*) FROM vision.processed_events")).scalar()
    assert n == 1
    # Named clusters' members leave the unknown buffer.
    assert store.buffer("toy") == []
