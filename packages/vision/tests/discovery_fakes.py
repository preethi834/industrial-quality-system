"""Helpers for the discovery tests: controllable embeddings, no detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from PIL import Image

from iqs_vision.discovery.service import Scored
from iqs_vision.discovery.trigger import l2_normalise

T0 = datetime(2026, 1, 5, 6, 0, tzinfo=UTC)
DIM = 64


def unit(rng: np.random.Generator, centre: np.ndarray | None = None, spread: float = 0.02):  # type: ignore[no-untyped-def]
    if centre is None:
        return l2_normalise(rng.normal(size=DIM))
    return l2_normalise(centre + spread * rng.normal(size=DIM))


@dataclass
class FakeScorer:
    """Returns whatever the test queued: (score, embedding) per call; threshold 1.0."""

    queue: list[tuple[float, np.ndarray | None]] = field(default_factory=list)
    threshold: float | None = 1.0

    def score(self, image: Image.Image, category: str) -> Scored:
        s, emb = self.queue.pop(0)
        anomalous = self.threshold is not None and s > self.threshold
        return Scored(s, self.threshold, "fake-v1", emb if anomalous else None)


def ts(i: int) -> datetime:
    return T0 + timedelta(minutes=10 * i)


IMG = Image.new("RGB", (8, 8))


def img(i: int) -> Image.Image:
    """A distinct image per index (the trigger counts distinct images, KNOWN_ISSUES #46)."""
    return Image.new("RGB", (8, 8), (i % 256, (i // 256) % 256, 7))
