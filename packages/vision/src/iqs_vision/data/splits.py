"""Split manifests: which image belongs to which split, with a SHA-256 per file.

One manifest per category, committed under eval_sets/vision/<dataset>/<category>.json.
Images stay out of git; the manifest pins exactly which files every run used.

Primary split (`split`) — every image has exactly one:
    fit    train/good images used to fit the detector (PatchCore memory bank)
    calib  held-out train/good images (good only), never seen by the detector; used for
           the provisional threshold now and the conformal threshold in step 2
    test   the official MVTec test set (good + defects)

Supervised split (`sup_split`) — only for official test images; the test set is the
only labelled defect data, so the supervised baseline re-splits it, stratified by
defect type: sup_train / sup_val / sup_test. PatchCore is also scored on sup_test, so
both models are compared on the same images.

Protocols that pick fit images:
    official  fit + calib (the full MVTec train set, for comparison with published numbers)
    deploy    fit only (calibration images stay unseen, blueprint Rev 2 §2.8)
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iqs_common.assets import sha256_file
from iqs_vision.data.mvtec import GOOD, Sample

PrimarySplit = Literal["fit", "calib", "test"]
SupSplit = Literal["sup_train", "sup_val", "sup_test"]
Protocol = Literal["official", "deploy"]
MANIFEST_VERSION = 1


class SplitLeakageError(ValueError):
    """An image (by path or by content) appears in more than one split."""


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    split: PrimarySplit
    sup_split: SupSplit | None = None
    label: int = Field(ge=0, le=1)
    defect_type: str
    mask: str | None = None


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = MANIFEST_VERSION
    dataset: str
    category: str
    seed: int
    calib_fraction: float
    sup_fractions: tuple[float, float, float]
    entries: tuple[ManifestEntry, ...]

    @model_validator(mode="after")
    def _check(self) -> SplitManifest:
        validate_entries(self.entries)
        return self

    def select(
        self, split: PrimarySplit | None = None, sup_split: SupSplit | None = None
    ) -> list[ManifestEntry]:
        return [
            e
            for e in self.entries
            if (split is None or e.split == split)
            and (sup_split is None or e.sup_split == sup_split)
        ]

    def fit_entries(self, protocol: Protocol) -> list[ManifestEntry]:
        splits = ("fit", "calib") if protocol == "official" else ("fit",)
        return [e for e in self.entries if e.split in splits]

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON; logged with every run."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def validate_entries(entries: Iterable[ManifestEntry]) -> None:
    """Raise SplitLeakageError / ValueError if the split assignment is unsafe."""
    by_path: dict[str, ManifestEntry] = {}
    split_by_hash: dict[str, set[str]] = defaultdict(set)
    sup_by_hash: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        if e.path in by_path:
            raise SplitLeakageError(f"{e.path} listed twice")
        by_path[e.path] = e
        split_by_hash[e.sha256].add(e.split)
        if e.sup_split is not None:
            sup_by_hash[e.sha256].add(e.sup_split)
        if (e.label == 0) != (e.defect_type == GOOD):
            raise ValueError(f"{e.path}: label {e.label} contradicts defect type {e.defect_type}")
        if e.label == 1 and e.mask is None:
            raise ValueError(f"{e.path}: defect image without mask")
        if e.split in ("fit", "calib") and e.label != 0:
            raise ValueError(f"{e.path}: {e.split} must contain good images only")
        if (e.sup_split is not None) != (e.split == "test"):
            raise ValueError(f"{e.path}: sup_split is only (and always) set for test images")
    for digest, splits in split_by_hash.items():
        if len(splits) > 1:
            raise SplitLeakageError(f"identical image content {digest[:12]} in splits {splits}")
    for digest, splits in sup_by_hash.items():
        if len(splits) > 1:
            raise SplitLeakageError(f"identical image content {digest[:12]} in splits {splits}")


def _split_counts(n: int, fractions: tuple[float, ...]) -> list[int]:
    """Largest-remainder rounding; every non-empty group gets at least one test item."""
    raw = [n * f for f in fractions]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[
        : n - sum(counts)
    ]:
        counts[i] += 1
    if n >= len(fractions) and counts[-1] == 0:
        counts[-1], counts[0] = 1, counts[0] - 1
    return counts


def build_manifest(
    samples: list[Sample],
    *,
    root: Path,
    dataset: str,
    category: str,
    seed: int = 42,
    calib_fraction: float = 0.2,
    sup_fractions: tuple[float, float, float] = (0.5, 0.2, 0.3),
) -> SplitManifest:
    """Assign splits deterministically (sorted input + numpy Generator(seed))."""
    if not 0 < calib_fraction < 1:
        raise ValueError("calib_fraction must be in (0, 1)")
    if abs(sum(sup_fractions) - 1) > 1e-9:
        raise ValueError("sup_fractions must sum to 1")
    rng = np.random.default_rng(seed)
    ordered = sorted(samples, key=lambda s: s.path)
    entries: list[ManifestEntry] = []

    train = [s for s in ordered if s.source_split == "train"]
    n_calib = max(1, round(len(train) * calib_fraction))
    calib_idx = set(rng.permutation(len(train))[:n_calib].tolist())
    for i, s in enumerate(train):
        split: PrimarySplit = "calib" if i in calib_idx else "fit"
        entries.append(_entry(s, root, split, None))

    by_type: dict[str, list[Sample]] = defaultdict(list)
    for s in ordered:
        if s.source_split == "test":
            by_type[s.defect_type].append(s)
    names: tuple[SupSplit, SupSplit, SupSplit] = ("sup_train", "sup_val", "sup_test")
    for defect_type in sorted(by_type):
        group = by_type[defect_type]
        order = rng.permutation(len(group)).tolist()
        counts = _split_counts(len(group), sup_fractions)
        start = 0
        for name, count in zip(names, counts, strict=True):
            for j in order[start : start + count]:
                entries.append(_entry(group[j], root, "test", name))
            start += count

    entries.sort(key=lambda e: e.path)
    return SplitManifest(
        dataset=dataset,
        category=category,
        seed=seed,
        calib_fraction=calib_fraction,
        sup_fractions=sup_fractions,
        entries=tuple(entries),
    )


def _entry(s: Sample, root: Path, split: PrimarySplit, sup: SupSplit | None) -> ManifestEntry:
    return ManifestEntry(
        path=s.path,
        sha256=sha256_file(root / s.path),
        split=split,
        sup_split=sup,
        label=s.label,
        defect_type=s.defect_type,
        mask=s.mask_path,
    )


def verify_on_disk(manifest: SplitManifest, root: Path) -> list[str]:
    """Paths whose file is missing or whose SHA-256 changed (empty list = all good)."""
    bad = []
    for e in manifest.entries:
        p = root / e.path
        if not p.is_file() or sha256_file(p) != e.sha256:
            bad.append(e.path)
    return bad
