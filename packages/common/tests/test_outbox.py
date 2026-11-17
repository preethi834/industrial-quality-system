"""Unit tests for the outbox model (no database). Postgres behaviour is covered by
tests/integration/test_migrations.py."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, DateTime, Index
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from iqs_common.db import Base
from iqs_common.events import (
    CaseConfirmedEvent,
    CaseConfirmedPayload,
    NewDefectClusterEvent,
    NewDefectClusterPayload,
    parse_event,
)
from iqs_common.outbox import OUTBOX_SCHEMA, OutboxEvent, OutboxStatus

TABLE = Base.metadata.tables["platform.events"]


def _cluster_event() -> NewDefectClusterEvent:
    now = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
    return NewDefectClusterEvent(
        payload=NewDefectClusterPayload(
            cluster_id=uuid.uuid4(),
            category="bottle",
            size=3,
            first_seen=now,
            last_seen=now,
            inspection_ids=(uuid.uuid4(), uuid.uuid4()),
            model_version="patchcore-v1",
        )
    )


def test_table_is_registered_in_platform_schema_on_shared_metadata() -> None:
    assert OUTBOX_SCHEMA == "platform"
    assert TABLE.schema == "platform"
    assert TABLE.fullname == "platform.events"
    assert OutboxEvent.__table__ is TABLE


def test_columns_types_and_nullability() -> None:
    cols = TABLE.c
    assert [c.name for c in cols] == [
        "id",
        "type",
        "payload",
        "status",
        "attempts",
        "created_at",
        "processed_at",
    ]
    assert [c.name for c in TABLE.primary_key.columns] == ["id"]
    nullable = {c.name for c in cols if c.nullable}
    assert nullable == {"processed_at"}
    assert isinstance(cols.payload.type, postgresql.JSONB)
    for name in ("created_at", "processed_at"):
        col_type = cols[name].type
        assert isinstance(col_type, DateTime)
        assert col_type.timezone is True
    assert cols.status.server_default is not None
    assert cols.attempts.server_default is not None
    assert cols.created_at.server_default is not None


def test_status_check_constraint_lists_exactly_the_enum_values() -> None:
    checks = {c.name: str(c.sqltext) for c in TABLE.constraints if isinstance(c, CheckConstraint)}
    assert checks["ck_events_status"] == "status IN ('pending', 'processed', 'failed')"
    assert checks["ck_events_attempts_non_negative"] == "attempts >= 0"
    assert [s.value for s in OutboxStatus] == ["pending", "processed", "failed"]


def test_worker_index_on_status_then_created_at() -> None:
    (index,) = [i for i in TABLE.indexes if isinstance(i, Index)]
    assert index.name == "ix_events_status_created_at"
    assert [c.name for c in index.columns] == ["status", "created_at"]


def test_postgres_ddl_contains_constraints() -> None:
    ddl = str(CreateTable(TABLE).compile(dialect=postgresql.dialect()))
    assert "CREATE TABLE platform.events" in ddl
    assert "payload JSONB NOT NULL" in ddl
    assert "CONSTRAINT pk_events PRIMARY KEY (id)" in ddl
    assert "CONSTRAINT ck_events_status CHECK (status IN ('pending', 'processed', 'failed'))" in ddl


def test_from_event_uses_event_id_and_round_trips_through_parse_event() -> None:
    event = _cluster_event()
    row = OutboxEvent.from_event(event)
    assert row.id == event.event_id
    assert row.type == "new_defect_cluster"
    assert row.status == OutboxStatus.PENDING
    assert row.attempts == 0
    assert row.processed_at is None
    assert parse_event(row.payload) == event


def test_from_event_payload_is_plain_json() -> None:
    event = CaseConfirmedEvent(
        payload=CaseConfirmedPayload(
            case_id=uuid.uuid4(),
            cause_code="MACHINE.TOOL_WEAR",
            taxonomy_version="0.1.0",
            confirmed_by="engineer",
            confirmed_at=datetime(2026, 11, 17, tzinfo=UTC),
        )
    )
    row = OutboxEvent.from_event(event)
    assert row.type == "case_confirmed"
    assert isinstance(row.payload["event_id"], str)
    assert isinstance(row.payload["occurred_at"], str)
    assert parse_event(row.payload) == event
