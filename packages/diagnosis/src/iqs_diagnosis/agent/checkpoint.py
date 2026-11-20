"""Checkpointers: in memory (tests, evaluation) or Postgres (service, cases can pause)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from iqs_common.settings import DatabaseSettings


def memory_checkpointer() -> BaseCheckpointSaver[Any]:
    return InMemorySaver()


def postgres_checkpointer(settings: DatabaseSettings) -> BaseCheckpointSaver[Any]:
    """PostgresSaver on its own autocommit connection; creates its tables if missing.

    The checkpoint tables are managed by langgraph-checkpoint-postgres (blueprint §2.4),
    not by our Alembic migrations.
    """
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    # SQLAlchemy URL -> libpq URL; str(URL) would hide the password.
    url = settings.sqlalchemy_url().set(drivername="postgresql")
    conninfo = url.render_as_string(hide_password=False)
    conn = psycopg.connect(conninfo, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver
