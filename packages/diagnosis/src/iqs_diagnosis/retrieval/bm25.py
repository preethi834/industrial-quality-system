"""Keyword retrieval (BM25 over the rendered symptom text)."""

from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from iqs_diagnosis.retrieval.base import Hit, ranked


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, documents: Sequence[tuple[str, str]]) -> None:
        if not documents:
            raise ValueError("BM25 index needs at least one document")
        self.ids = [doc_id for doc_id, _ in documents]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate document ids")
        self._bm25 = BM25Okapi([tokenize(text) for _, text in documents])

    def search(self, text: str, k: int) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(text))
        return ranked(list(zip(self.ids, (float(s) for s in scores), strict=True)), k)
