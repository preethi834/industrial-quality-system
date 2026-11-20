"""Build a small agent over the reference scenarios (no Docker, in-process MCP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.agent.checkpoint import memory_checkpointer
from iqs_diagnosis.agent.graph import AgentDeps, DiagnosisAgent
from iqs_diagnosis.agent.tickets import McpTicketGateway, TicketGateway
from iqs_diagnosis.persistence.repository import InMemoryCaseRepository
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.scoring.ranker import CauseRanker, RankerParams
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary

from .conftest import TAXONOMY_PATH, VOCABULARY_PATH

# Left out of the library so the agent meets them as new problems.
NOVEL_SCENARIOS = ("drive_fault_m1", "handling_error_shift_b")
PARAMS = RankerParams(
    n_neighbours=3,
    alpha=0.01,
    novelty_threshold=0.5,
    novelty_alpha=0.1,
    n_novelty_calibration=0,
)


def make_agent(
    reference_profiles: dict[str, tuple[Any, Any]], gateway: TicketGateway | None = None
) -> tuple[DiagnosisAgent, InMemoryCaseRepository, Any]:
    from iqs_mcp.server import build_server
    from iqs_mcp.store import InMemoryTicketStore

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    library = [
        LibraryCase(p, lab.cause_code)
        for name, (p, lab) in reference_profiles.items()
        if name not in NOVEL_SCENARIOS
    ]
    ranker = CauseRanker(library, [c.code for c in taxonomy.causes], PARAMS)
    store = InMemoryTicketStore()
    repo = InMemoryCaseRepository()
    deps = AgentDeps(
        vocabulary=load_vocabulary(VOCABULARY_PATH),
        ranker=ranker,
        taxonomy=taxonomy,
        tickets=gateway or McpTicketGateway(build_server(store)),
        repository=repo,
    )
    return DiagnosisAgent(deps, memory_checkpointer()), repo, store


def run_dir(reference_runs: dict[str, Path], name: str) -> Path:
    return reference_runs[name]
