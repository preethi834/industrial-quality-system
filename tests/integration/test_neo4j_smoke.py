"""Neo4j smoke test: the pinned compose image starts and answers over Bolt."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_neo4j_answers_cypher(neo4j_image: str) -> None:
    from testcontainers.community.neo4j import Neo4jContainer

    container = Neo4jContainer(neo4j_image, password="iqs_test_pw")
    container.with_env("NEO4J_server_memory_heap_max__size", "512m")
    container.with_env("NEO4J_server_memory_pagecache_size", "128m")
    with container, container.get_driver() as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query("RETURN 1 AS one")
        assert records[0]["one"] == 1
        info = driver.get_server_info()
        assert info.agent.startswith("Neo4j/5.26")
