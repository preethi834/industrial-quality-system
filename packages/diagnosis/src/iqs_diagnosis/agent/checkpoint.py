"""Checkpointers: in memory (tests, evaluation) or Postgres (service, cases can pause)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

from iqs_common.settings import DatabaseSettings


def memory_checkpointer() -> BaseCheckpointSaver[Any]:
    return InMemorySaver()


if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver


# Names of the tables langgraph-checkpoint-postgres creates (outside Alembic).
CHECKPOINT_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)


def _threaded_saver_class() -> type[PostgresSaver]:
    from langgraph.checkpoint.postgres import PostgresSaver

    class ThreadedPostgresSaver(PostgresSaver):
        """The sync PostgresSaver with its async methods run in a worker thread.

        The agent runs with ``ainvoke`` while ``phase()``/``state()`` read synchronously;
        the sync saver implements no async methods (it raises NotImplementedError).
        One sync connection (guarded by the saver's own lock) serves both and does not
        depend on an event loop. ``AsyncPostgresSaver`` would bind the connection to one
        loop and make every state read async.
        """

        async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            return await asyncio.to_thread(self.get_tuple, config)

        async def alist(
            self,
            config: RunnableConfig | None,
            *,
            filter: dict[str, Any] | None = None,
            before: RunnableConfig | None = None,
            limit: int | None = None,
        ) -> AsyncIterator[CheckpointTuple]:
            items = await asyncio.to_thread(
                lambda: list(self.list(config, filter=filter, before=before, limit=limit))
            )
            for item in items:
                yield item

        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

        async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

        async def adelete_thread(self, thread_id: str) -> None:
            await asyncio.to_thread(self.delete_thread, thread_id)

    return ThreadedPostgresSaver


def postgres_checkpointer(settings: DatabaseSettings) -> BaseCheckpointSaver[Any]:
    """Postgres checkpointer on its own autocommit connection; creates its tables if missing.

    Usable from sync and async graph calls (see ``ThreadedPostgresSaver``).

    The checkpoint tables are managed by langgraph-checkpoint-postgres (blueprint §2.4),
    not by our Alembic migrations.
    """
    import psycopg
    from psycopg.rows import dict_row

    # SQLAlchemy URL -> libpq URL; str(URL) would hide the password.
    url = settings.sqlalchemy_url().set(drivername="postgresql")
    conninfo = url.render_as_string(hide_password=False)
    conn = psycopg.connect(conninfo, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    saver = _threaded_saver_class()(conn)
    saver.setup()
    return saver
