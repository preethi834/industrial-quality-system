"""Vector retrieval over the numeric symptom features.

Features are first compressed with ``asinh`` (so a z-shift of 12 does not outweigh
everything else) and then standardised with the library's mean and SD. Similarity is
``1 / (1 + d)`` where ``d`` is the root-mean-square difference per feature in that
space: 1.0 for an identical profile, falling towards 0. The novelty check uses the
largest similarity to any library case.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iqs_diagnosis.retrieval.base import Hit, ranked

SD_FLOOR = 1e-3


@dataclass(frozen=True)
class Standardizer:
    names: tuple[str, ...]
    mean: NDArray[np.float64]
    sd: NDArray[np.float64]

    @classmethod
    def fit(cls, names: Sequence[str], rows: Sequence[Mapping[str, float]]) -> Standardizer:
        x = np.arcsinh(_matrix(names, rows))
        return cls(
            names=tuple(names),
            mean=x.mean(axis=0),
            sd=np.maximum(x.std(axis=0, ddof=1) if len(rows) > 1 else 0.0, SD_FLOOR),
        )

    def transform(self, rows: Sequence[Mapping[str, float]]) -> NDArray[np.float64]:
        return (np.arcsinh(_matrix(self.names, rows)) - self.mean) / self.sd


def _matrix(names: Sequence[str], rows: Sequence[Mapping[str, float]]) -> NDArray[np.float64]:
    missing = {n for r in rows for n in names if n not in r}
    if missing:
        raise KeyError(f"features missing: {sorted(missing)}")
    return np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float64).reshape(
        len(rows), len(names)
    )


class VectorIndex:
    def __init__(
        self,
        names: Sequence[str],
        items: Sequence[tuple[str, Mapping[str, float]]],
    ) -> None:
        if not items:
            raise ValueError("vector index needs at least one item")
        self.ids = [cid for cid, _ in items]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate ids")
        rows = [f for _, f in items]
        self.standardizer = Standardizer.fit(names, rows)
        self._x = self.standardizer.transform(rows)

    def similarities(self, features: Mapping[str, float]) -> NDArray[np.float64]:
        q = self.standardizer.transform([features])[0]
        d = np.sqrt(((self._x - q) ** 2).mean(axis=1))
        return np.asarray(1.0 / (1.0 + d), dtype=np.float64)

    def search(self, features: Mapping[str, float], k: int) -> list[Hit]:
        sims = self.similarities(features)
        return ranked(list(zip(self.ids, (float(s) for s in sims), strict=True)), k)

    def max_similarity(self, features: Mapping[str, float]) -> float:
        return float(self.similarities(features).max())
