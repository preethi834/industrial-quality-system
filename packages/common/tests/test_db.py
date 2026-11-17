"""Unit tests for db plumbing. Real-Postgres integration tests arrive in step 0d."""

from __future__ import annotations

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from iqs_common.db import (
    NAMING_CONVENTION,
    Base,
    create_db_engine,
    create_session_factory,
    session_scope,
)
from iqs_common.settings import DatabaseSettings


def test_engine_uses_psycopg3_and_pool_size_without_connecting() -> None:
    engine = create_db_engine(DatabaseSettings(password=SecretStr("pw"), pool_size=3))
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "psycopg"
        assert isinstance(engine.pool, QueuePool)
        assert engine.pool.size() == 3
        assert engine.pool.checkedout() == 0
        assert engine.pool.checkedin() == 0  # no connection has ever been opened
    finally:
        engine.dispose()


def test_session_scope_commits_on_success_and_rolls_back_on_error() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
    factory = create_session_factory(engine)

    with session_scope(factory) as session:
        session.execute(text("INSERT INTO t VALUES (1)"))

    with pytest.raises(RuntimeError), session_scope(factory) as session:
        session.execute(text("INSERT INTO t VALUES (2)"))
        raise RuntimeError("fail inside the transaction")

    with engine.connect() as conn:
        assert conn.execute(text("SELECT x FROM t ORDER BY x")).scalars().all() == [1]


def test_session_factory_keeps_objects_usable_after_commit() -> None:
    factory = create_session_factory(create_engine("sqlite://"))
    assert factory.kw["expire_on_commit"] is False


def test_base_metadata_has_naming_convention() -> None:
    assert dict(Base.metadata.naming_convention) == NAMING_CONVENTION
