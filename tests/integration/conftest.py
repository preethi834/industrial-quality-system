"""Shared fixtures for integration tests against real Postgres / Neo4j (testcontainers).

Images are read from deploy/compose/docker-compose.yml, so tests and compose
always run the same pinned image. Containers are started once per session.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine

from iqs_common.settings import DatabaseSettings, Neo4jSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "deploy/compose/docker-compose.yml"
ALEMBIC_INI = REPO_ROOT / "infrastructure/database/alembic.ini"

PG_USER, PG_PASSWORD, PG_DB = "iqs_test", "iqs_test_pw", "iqs_test"


def compose_image(service: str) -> str:
    data: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    image: str = data["services"][service]["image"]
    return image


NEO4J_PASSWORD = "iqs_test_pw"


@pytest.fixture(scope="session")
def neo4j_image() -> str:
    return compose_image("neo4j")


@pytest.fixture(scope="session")
def neo4j_settings(neo4j_image: str) -> Iterator[Neo4jSettings]:
    """One Neo4j container per session, with a small heap for the test VM."""
    from testcontainers.community.neo4j import Neo4jContainer

    container = Neo4jContainer(neo4j_image, password=NEO4J_PASSWORD)
    container.with_env("NEO4J_server_memory_heap_max__size", "512m")
    container.with_env("NEO4J_server_memory_pagecache_size", "128m")
    with container:
        yield Neo4jSettings(
            uri=container.get_connection_url(),
            user="neo4j",
            password=SecretStr(NEO4J_PASSWORD),
        )


@pytest.fixture(scope="session")
def pg_settings() -> Iterator[DatabaseSettings]:
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(
        compose_image("postgres"),
        username=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
        driver="psycopg",
    ) as pg:
        yield DatabaseSettings(
            host=pg.get_container_host_ip(),
            port=int(pg.get_exposed_port(5432)),
            name=PG_DB,
            user=PG_USER,
            password=SecretStr(PG_PASSWORD),
        )


@pytest.fixture(scope="session")
def pg_engine(pg_settings: DatabaseSettings) -> Iterator[Engine]:
    engine = create_engine(pg_settings.sqlalchemy_url())
    yield engine
    engine.dispose()


@pytest.fixture
def alembic_cfg(
    pg_settings: DatabaseSettings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Config:
    """Alembic config whose env.py reaches the test container through IQS_* variables.

    Runs in an empty working directory and with all other IQS_* variables removed,
    so a developer's real .env can never point a test at the compose database.
    """
    for key in list(os.environ):
        if key.startswith("IQS_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IQS_DATABASE__HOST", pg_settings.host)
    monkeypatch.setenv("IQS_DATABASE__PORT", str(pg_settings.port))
    monkeypatch.setenv("IQS_DATABASE__NAME", pg_settings.name)
    monkeypatch.setenv("IQS_DATABASE__USER", pg_settings.user)
    monkeypatch.setenv("IQS_DATABASE__PASSWORD", pg_settings.password.get_secret_value())
    cfg = Config(str(ALEMBIC_INI))
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture
def migrated(alembic_cfg: Config) -> Iterator[Config]:
    """Database at head for one test, back to an empty database afterwards."""
    command.upgrade(alembic_cfg, "head")
    yield alembic_cfg
    command.downgrade(alembic_cfg, "base")
