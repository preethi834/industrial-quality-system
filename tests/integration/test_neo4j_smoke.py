"""Neo4j smoke test: the pinned compose image starts and answers over Bolt."""

from __future__ import annotations

import pytest
from neo4j import GraphDatabase

from iqs_common.settings import Neo4jSettings

pytestmark = pytest.mark.integration


def test_neo4j_answers_cypher(neo4j_settings: Neo4jSettings) -> None:
    auth = (neo4j_settings.user, neo4j_settings.password.get_secret_value())
    with GraphDatabase.driver(neo4j_settings.uri, auth=auth) as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query("RETURN 1 AS one")
        assert records[0]["one"] == 1
        assert driver.get_server_info().agent.startswith("Neo4j/5.26")
