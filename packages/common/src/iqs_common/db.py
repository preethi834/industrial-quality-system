"""Database plumbing: engine, sessions, transactions, declarative base.

Tables themselves are defined by each service (and migrated by Alembic, step 0d);
they all share Base.metadata so Alembic sees one consistent model.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from iqs_common.settings import DatabaseSettings

# Deterministic constraint names, so Alembic migrations are stable and reviewable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def create_db_engine(
    settings: DatabaseSettings, *, echo: bool = False, connect_timeout_s: int | None = None
) -> Engine:
    """Create (but do not connect) an engine with a small connection pool."""
    connect_args = {} if connect_timeout_s is None else {"connect_timeout": connect_timeout_s}
    return create_engine(
        settings.sqlalchemy_url(),
        pool_size=settings.pool_size,
        max_overflow=settings.pool_size,
        pool_pre_ping=True,
        echo=echo,
        connect_args=connect_args,
    )


def check_database(engine: Engine) -> None:
    """Readiness probe: one round trip. Raises on any failure."""
    with engine.connect() as conn:
        conn.exec_driver_sql("SELECT 1")


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One transaction: commit on success, roll back on any exception."""
    with factory() as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
