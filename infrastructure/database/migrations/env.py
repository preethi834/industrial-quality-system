"""Alembic environment for the iqs database.

- URL comes from iqs_common settings (IQS_DATABASE__*), never from alembic.ini.
- Every module that defines tables must be imported here so its tables are on
  Base.metadata; otherwise autogenerate would propose dropping them.
- Autogenerate only looks at our schemas. The alembic_version table stays in
  public (it is created before the first migration creates our schemas).
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any, Literal

from alembic import context
from sqlalchemy import create_engine, pool

import iqs_common.outbox  # noqa: F401  (registers platform.events)
import iqs_diagnosis.persistence.repository  # noqa: F401  (registers diagnosis.* tables)
import iqs_mcp.store  # noqa: F401  (registers qms.* tables)
from iqs_common.db import Base
from iqs_common.settings import CommonSettings

MANAGED_SCHEMAS = frozenset({"diagnosis", "vision", "qms", "platform"})

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

NameType = Literal[
    "schema", "table", "column", "index", "unique_constraint", "foreign_key_constraint"
]


def include_name(name: str | None, type_: NameType, parent_names: Any) -> bool:
    """Reflection filter: only reflect our schemas (skips public, incl. alembic_version)."""
    if type_ == "schema":
        return name in MANAGED_SCHEMAS
    return True


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Model-side filter: a table outside our schemas is never migrated."""
    if type_ == "table":
        return obj.schema in MANAGED_SCHEMAS
    return True


COMMON_OPTS: dict[str, Any] = {
    "target_metadata": target_metadata,
    "include_schemas": True,
    "include_name": include_name,
    "include_object": include_object,
    "compare_type": True,
    "compare_server_default": True,
}


def run_migrations_offline() -> None:
    url = CommonSettings().database.sqlalchemy_url()
    context.configure(
        url=url.render_as_string(hide_password=True),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **COMMON_OPTS,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = CommonSettings().database.sqlalchemy_url()
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, **COMMON_OPTS)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
