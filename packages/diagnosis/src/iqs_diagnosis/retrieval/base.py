"""Shared retrieval types (pure data)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from iqs_diagnosis.domain.case import SymptomProfile


@dataclass(frozen=True)
class LibraryCase:
    """A past case whose cause an engineer confirmed (the retrieval corpus)."""

    profile: SymptomProfile
    cause_code: str

    @property
    def case_id(self) -> str:
        return self.profile.case_id


@dataclass(frozen=True)
class Hit:
    case_id: str
    score: float  # higher is closer; scale depends on the retriever
    rank: int  # 1-based


def ranked(scores: Sequence[tuple[str, float]], k: int) -> list[Hit]:
    """Top ``k`` by score, ties broken by case_id so results are deterministic."""
    order = sorted(scores, key=lambda s: (-s[1], s[0]))[:k]
    return [Hit(case_id=cid, score=float(sc), rank=i + 1) for i, (cid, sc) in enumerate(order)]
