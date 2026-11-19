"""Fit and evaluate detectors on manifest splits (used by scripts and tests)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iqs_vision.data.dataset import ManifestDataset
from iqs_vision.data.splits import ManifestEntry, Protocol, SplitManifest
from iqs_vision.detectors.base import Detector, FitStats
from iqs_vision.evaluation.metrics import f1_max, image_auroc, pixel_auroc, rates_at_threshold


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(
    entries: Sequence[ManifestEntry], root: Path, size: int, batch_size: int
) -> DataLoader[dict[str, object]]:
    ds = ManifestDataset(entries, root, size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)  # type: ignore[arg-type]


def fit_detector(
    detector: Detector,
    manifest: SplitManifest,
    root: Path,
    protocol: Protocol,
    batch_size: int,
) -> FitStats:
    entries = manifest.fit_entries(protocol)

    def images() -> Iterator[torch.Tensor]:
        for b in _loader(entries, root, detector.image_size, batch_size):
            yield b["image"]

    return detector.fit(images(), n_images=len(entries))


@dataclass
class Scored:
    paths: list[str]
    labels: np.ndarray
    scores: np.ndarray
    masks: np.ndarray | None
    maps: np.ndarray | None


def score_entries(
    detector: Detector,
    entries: Sequence[ManifestEntry],
    root: Path,
    batch_size: int,
    *,
    keep_maps: bool = True,
) -> Scored:
    paths: list[str] = []
    labels, scores, masks, maps = [], [], [], []
    for b in _loader(entries, root, detector.image_size, batch_size):
        out = detector.predict(b["image"])
        paths += list(b["path"])
        labels.append(np.asarray(b["label"]))
        scores.append(out.scores)
        if keep_maps:
            masks.append(np.asarray(b["mask"], dtype=np.uint8))
            maps.append(out.maps.astype(np.float32))
    return Scored(
        paths=paths,
        labels=np.concatenate(labels),
        scores=np.concatenate(scores),
        masks=np.concatenate(masks) if keep_maps else None,
        maps=np.concatenate(maps) if keep_maps else None,
    )


def evaluate(
    detector: Detector, manifest: SplitManifest, root: Path, protocol: Protocol, batch_size: int
) -> tuple[dict[str, float], Scored]:
    """Metrics on the official test set (+ sup_test subset, + calibration threshold)."""
    test = score_entries(detector, manifest.select("test"), root, batch_size)
    assert test.masks is not None and test.maps is not None
    metrics: dict[str, float] = {
        "image_auroc": image_auroc(test.labels, test.scores),
        "pixel_auroc": pixel_auroc(test.masks, test.maps),
        "n_test": float(len(test.labels)),
    }
    f1, _ = f1_max(test.labels, test.scores)
    metrics["image_f1_max"] = f1
    sup_paths = {e.path for e in manifest.select(sup_split="sup_test")}
    sel = np.array([p in sup_paths for p in test.paths])
    if sel.any() and len(set(test.labels[sel].tolist())) == 2:
        metrics["sup_test_image_auroc"] = image_auroc(test.labels[sel], test.scores[sel])
    if protocol == "deploy":
        calib = score_entries(detector, manifest.select("calib"), root, batch_size, keep_maps=False)
        # Provisional threshold until step 2's conformal one: the largest calibration score.
        threshold = float(calib.scores.max())
        metrics["threshold_max_calib"] = threshold
        metrics["n_calib"] = float(len(calib.scores))
        for k, v in rates_at_threshold(test.labels, test.scores, threshold).items():
            metrics[f"test_{k}"] = v
    return metrics, test
