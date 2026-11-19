"""Detector interface: every anomaly detector (PatchCore, EfficientAD, ...) looks the same
to training scripts, evaluation and the API."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@dataclass(frozen=True)
class DetectorOutput:
    """Batch output. Higher = more anomalous."""

    scores: np.ndarray  # (B,) image-level anomaly scores
    maps: np.ndarray  # (B, H, W) pixel-level anomaly maps at the input resolution


@dataclass
class FitStats:
    n_images: int
    n_patches: int
    memory_bank_size: int
    seconds_features: float
    seconds_coreset: float
    peak_rss_mb: float
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Detector(Protocol):
    name: str
    image_size: int

    def fit(self, batches: Iterable[torch.Tensor], n_images: int) -> FitStats:
        """Fit on normalised good images, given as (B, 3, S, S) batches."""
        ...

    def predict(self, images: torch.Tensor) -> DetectorOutput:
        """Score a (B, 3, S, S) batch of normalised images."""
        ...

    def save(self, path: Path) -> None: ...


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (Linux reports KiB)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
