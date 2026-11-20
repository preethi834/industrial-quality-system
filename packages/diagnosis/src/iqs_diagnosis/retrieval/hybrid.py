"""Hybrid retrieval: BM25 on symptom text + vector search on features, fused with RRF.

Step 3a/3b version: both retrievers run in memory over the case library. Dense text
embeddings and the cross-encoder reranker are added in step 3f (they need model
downloads and matter for free-text NHTSA cases, not for simulator profiles).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from iqs_diagnosis.domain.case import SymptomProfile
from iqs_diagnosis.retrieval.base import Hit, LibraryCase
from iqs_diagnosis.retrieval.bm25 import BM25Index
from iqs_diagnosis.retrieval.fusion import RRF_K, rrf
from iqs_diagnosis.retrieval.vector import VectorIndex
from iqs_diagnosis.symptoms.simulator_profile import FEATURE_NAMES


@dataclass(frozen=True)
class RetrievalResult:
    fused: list[Hit]
    bm25: list[Hit]
    vector: list[Hit]
    max_similarity: float  # vector similarity of the closest library case


class HybridRetriever:
    def __init__(
        self,
        library: Sequence[LibraryCase],
        *,
        depth: int = 50,
        rrf_k: int = RRF_K,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        if not library:
            raise ValueError("empty case library")
        self.library = {c.case_id: c for c in library}
        if len(self.library) != len(library):
            raise ValueError("duplicate case ids in library")
        self.depth = depth
        self.rrf_k = rrf_k
        self.bm25 = BM25Index([(c.case_id, c.profile.text()) for c in library])
        self.vector = VectorIndex(feature_names, [(c.case_id, c.profile.features) for c in library])

    def search(self, profile: SymptomProfile) -> RetrievalResult:
        b = self.bm25.search(profile.text(), self.depth)
        v = self.vector.search(profile.features, self.depth)
        return RetrievalResult(
            fused=rrf([b, v], k=self.rrf_k, top=self.depth),
            bm25=b,
            vector=v,
            max_similarity=self.vector.max_similarity(profile.features),
        )

    def cause_of(self, case_id: str) -> str:
        return self.library[case_id].cause_code
