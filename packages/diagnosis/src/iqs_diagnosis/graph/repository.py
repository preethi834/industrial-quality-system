"""Minimal knowledge-graph writer for confirmations (step 7, owner decision D4).

Blueprint §2.4 nodes / relationships used here:

    (:Case {id, source})-[:CONFIRMED_CAUSE {engineer, confirmed_at}]->(:Cause {code})
    (:DefectType {key, category, name})-[:INDICATES {count}]->(:Cause {code})
    (:Case)-[:SHOWS_DEFECT]->(:DefectType)          (vision-originated cases only)

``SHOWS_DEFECT`` is not in the blueprint list; it records which defect type a vision case
was opened for, so the graph can answer "which cases showed this defect type".

Every write is a ``MERGE`` and ``INDICATES.count`` is *set* to the Postgres count (the
source of truth), so a repeated write after a crash or retry changes nothing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class DefectTypeRef:
    category: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.category}/{self.name}"


@dataclass(frozen=True)
class ConfirmationWrite:
    case_id: str
    source: str
    cause_code: str
    cause_name: str
    engineer: str
    confirmed_at: datetime
    defect_type: DefectTypeRef | None = None
    indicates_count: int | None = None  # set on (:DefectType)-[:INDICATES]->(:Cause)


class GraphWriter(Protocol):
    def write_confirmation(self, w: ConfirmationWrite) -> None: ...


class InMemoryGraph:
    """Test double with the same MERGE semantics (sets of nodes and keyed edges)."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.writes = 0

    def _node(self, label: str, key: str, **props: Any) -> None:
        self.nodes.setdefault((label, key), {}).update(props)

    def write_confirmation(self, w: ConfirmationWrite) -> None:
        with self._lock:
            self.writes += 1
            self._node("Case", w.case_id, source=w.source)
            self._node("Cause", w.cause_code, name_en=w.cause_name)
            self.edges[("Case", w.case_id, "CONFIRMED_CAUSE", "Cause", w.cause_code)] = {
                "engineer": w.engineer,
                "confirmed_at": w.confirmed_at.isoformat(),
            }
            if w.defect_type is not None:
                d = w.defect_type
                self._node("DefectType", d.key, category=d.category, name=d.name)
                self.edges[("Case", w.case_id, "SHOWS_DEFECT", "DefectType", d.key)] = {}
                self.edges[("DefectType", d.key, "INDICATES", "Cause", w.cause_code)] = {
                    "count": w.indicates_count
                }

    def edge(
        self, src_label: str, src: str, rel: str, dst_label: str, dst: str
    ) -> dict[str, Any] | None:
        return self.edges.get((src_label, src, rel, dst_label, dst))


_CYPHER_CASE = """
MERGE (c:Case {id: $case_id}) SET c.source = $source
MERGE (k:Cause {code: $cause_code}) SET k.name_en = $cause_name
MERGE (c)-[r:CONFIRMED_CAUSE]->(k)
SET r.engineer = $engineer, r.confirmed_at = $confirmed_at
"""

_CYPHER_DEFECT = """
MATCH (c:Case {id: $case_id}), (k:Cause {code: $cause_code})
MERGE (d:DefectType {key: $key}) SET d.category = $category, d.name = $name
MERGE (c)-[:SHOWS_DEFECT]->(d)
MERGE (d)-[i:INDICATES]->(k) SET i.count = $count
"""

SCHEMA_CYPHER = (
    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT cause_code IF NOT EXISTS FOR (k:Cause) REQUIRE k.code IS UNIQUE",
    "CREATE CONSTRAINT defect_type_key IF NOT EXISTS FOR (d:DefectType) REQUIRE d.key IS UNIQUE",
)


class Neo4jGraphWriter:
    """Neo4j adapter (exercised by the integration and e2e tests, step 7g)."""

    def __init__(self, driver: Any, database: str | None = None) -> None:
        self._driver = driver
        self._database = database

    def ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as s:
            for stmt in SCHEMA_CYPHER:
                s.run(stmt)

    def write_confirmation(self, w: ConfirmationWrite) -> None:
        def tx(t: Any) -> None:
            t.run(
                _CYPHER_CASE,
                case_id=w.case_id,
                source=w.source,
                cause_code=w.cause_code,
                cause_name=w.cause_name,
                engineer=w.engineer,
                confirmed_at=w.confirmed_at.isoformat(),
            )
            if w.defect_type is not None:
                t.run(
                    _CYPHER_DEFECT,
                    case_id=w.case_id,
                    cause_code=w.cause_code,
                    key=w.defect_type.key,
                    category=w.defect_type.category,
                    name=w.defect_type.name,
                    count=w.indicates_count,
                )

        with self._driver.session(database=self._database) as s:
            s.execute_write(tx)
