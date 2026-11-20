"""Cause ranking for one case: retrieve -> vote -> top 3, plus the novelty flag.

Probabilities come only from retrieval votes over confirmed past cases (blueprint
§2.3: the LLM never produces the probabilities). They are not yet calibrated;
conformal cause sets are step 4.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from iqs_diagnosis.domain.case import SymptomProfile
from iqs_diagnosis.novelty.threshold import (
    NoveltyRates,
    NoveltyThreshold,
    fit_novelty_threshold,
    novelty_rates,
)
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.retrieval.fusion import RRF_K
from iqs_diagnosis.retrieval.hybrid import HybridRetriever
from iqs_diagnosis.scoring.votes import VoteParams, fit_vote_params, top_k, vote_probabilities


class RankerParams(BaseModel):
    """Fitted parameters, stored as JSON next to the corpus they were fitted on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_neighbours: int = Field(ge=1)
    alpha: float = Field(gt=0)
    depth: int = Field(ge=1, default=50)
    rrf_k: int = Field(ge=0, default=RRF_K)
    novelty_threshold: float | None  # None = never flag
    novelty_alpha: float = Field(gt=0, lt=1)
    n_novelty_calibration: int = Field(ge=0)
    fitted_on: dict[str, str] = Field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RankerParams:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Neighbour:
    case_id: str
    cause_code: str
    score: float


@dataclass(frozen=True)
class CauseRanking:
    case_id: str
    probabilities: dict[str, float]
    top3: list[tuple[str, float]]
    novel: bool
    max_similarity: float
    neighbours: list[Neighbour]


class CauseRanker:
    def __init__(
        self, library: Sequence[LibraryCase], causes: Sequence[str], params: RankerParams
    ) -> None:
        self.causes = tuple(causes)
        unknown = sorted({c.cause_code for c in library} - set(self.causes))
        if unknown:
            raise ValueError(f"library causes not in the taxonomy: {unknown}")
        self.params = params
        self.retriever = HybridRetriever(library, depth=params.depth, rrf_k=params.rrf_k)
        self.votes = VoteParams(params.n_neighbours, params.alpha)
        t = params.novelty_threshold
        self.novelty = NoveltyThreshold(
            threshold=float("-inf") if t is None else t,
            alpha=params.novelty_alpha,
            n_calibration=params.n_novelty_calibration,
        )

    def rank(self, profile: SymptomProfile) -> CauseRanking:
        r = self.retriever.search(profile)
        probs = vote_probabilities(r.fused, self.retriever.cause_of, self.causes, self.votes)
        return CauseRanking(
            case_id=profile.case_id,
            probabilities=probs,
            top3=top_k(probs, 3),
            novel=self.novelty.is_novel(r.max_similarity),
            max_similarity=r.max_similarity,
            neighbours=[
                Neighbour(h.case_id, self.retriever.cause_of(h.case_id), h.score)
                for h in r.fused[: self.votes.n_neighbours]
            ],
        )


@dataclass(frozen=True)
class FitReport:
    params: RankerParams
    dev_nll: float
    novelty: NoveltyRates


def fit_ranker(
    library: Sequence[LibraryCase],
    causes: Sequence[str],
    dev_profiles: Sequence[SymptomProfile],
    dev_causes: Sequence[str],
    *,
    novelty_alpha: float = 0.05,
    depth: int = 50,
    rrf_k: int = RRF_K,
    fitted_on: dict[str, str] | None = None,
) -> FitReport:
    """Fit vote parameters and the novelty threshold on labelled dev cases.

    Dev cases whose cause is absent from the library are the "new problems": they are
    left out of the vote fit and the threshold fit and used only to report detection.
    """
    if len(dev_profiles) != len(dev_causes):
        raise ValueError("one cause per dev profile")
    retriever = HybridRetriever(library, depth=depth, rrf_k=rrf_k)
    known_causes = {c.cause_code for c in library}
    results = [retriever.search(p) for p in dev_profiles]
    known = [i for i, c in enumerate(dev_causes) if c in known_causes]
    novel = [i for i, c in enumerate(dev_causes) if c not in known_causes]
    if not known:
        raise ValueError("no dev case has a cause from the library")
    votes, nll = fit_vote_params(
        [results[i].fused for i in known],
        [dev_causes[i] for i in known],
        retriever.cause_of,
        causes,
    )
    known_sims = [results[i].max_similarity for i in known]
    novel_sims = [results[i].max_similarity for i in novel]
    detector = fit_novelty_threshold(known_sims, novelty_alpha)
    params = RankerParams(
        n_neighbours=votes.n_neighbours,
        alpha=votes.alpha,
        depth=depth,
        rrf_k=rrf_k,
        novelty_threshold=None if detector.threshold == float("-inf") else detector.threshold,
        novelty_alpha=novelty_alpha,
        n_novelty_calibration=detector.n_calibration,
        fitted_on=fitted_on or {},
    )
    return FitReport(
        params=params, dev_nll=nll, novelty=novelty_rates(detector, known_sims, novel_sims)
    )
