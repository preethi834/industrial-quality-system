"""Read-only adapter for MVTec AD 2, one folder per category (per-category archives).

It never downloads (blueprint Rev 2 §3.9): missing data raises DatasetNotFoundError with the
import command. Anomalib's MVTecAD2 datamodule is not used because it downloads the dataset.

Layout (per-category archive, e.g. vial.tar.gz):
    <category>/train/good/000_regular.png
    <category>/validation/good/000_regular.png              good only -> calib
    <category>/test_public/good/000_<condition>.png
    <category>/test_public/bad/000_<condition>.png
    <category>/test_public/ground_truth/bad/000_<condition>_mask.png
    <category>/test_private*/                               no labels; not used

The lighting condition is the file-name suffix: "regular", "overexposed", "underexposed",
"shift_1", ... "regular" test images form the test split; the others the shift split.
Defects are not typed in test_public, so every defective image has defect_type "bad".
"""

from __future__ import annotations

import re
from pathlib import Path

from iqs_vision.data.mvtec import GOOD, IMAGE_SUFFIXES, DatasetNotFoundError, Sample

CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
BAD = "bad"
REGULAR = "regular"
_NAME = re.compile(r"^\d+_(?P<condition>[a-z]+(?:_\d+)?)$")


def condition_of(stem: str) -> str:
    """'003_shift_2' -> 'shift_2'; raises ValueError for unexpected names."""
    m = _NAME.match(stem)
    if m is None:
        raise ValueError(f"unexpected MVTec AD 2 file name {stem!r}")
    return m.group("condition")


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def scan_category(root: Path, category: str) -> list[Sample]:
    """All labelled samples of one category (train, validation, test_public), sorted by path."""
    base = root / category
    if not (base / "train" / GOOD).is_dir() or not (base / "test_public").is_dir():
        raise DatasetNotFoundError(
            f"MVTec AD 2 category {category!r} not found under {root}. Import the archive with: "
            f"uv run python scripts/download_dataset.py mvtec_ad2_{category} "
            "--archive <path> --approve"
        )
    samples: list[Sample] = []
    for img in _images(base / "train" / GOOD):
        samples.append(Sample(img.relative_to(root).as_posix(), "train", GOOD, None))
    for img in _images(base / "validation" / GOOD):
        samples.append(Sample(img.relative_to(root).as_posix(), "validation", GOOD, None))
    for img in _images(base / "test_public" / GOOD):
        cond = condition_of(img.stem)
        samples.append(Sample(img.relative_to(root).as_posix(), "test", GOOD, None, cond))
    for img in _images(base / "test_public" / BAD):
        cond = condition_of(img.stem)
        mask = base / "test_public" / "ground_truth" / BAD / f"{img.stem}_mask.png"
        if not mask.is_file():
            raise DatasetNotFoundError(f"missing ground-truth mask {mask}")
        samples.append(
            Sample(
                img.relative_to(root).as_posix(),
                "test",
                BAD,
                mask.relative_to(root).as_posix(),
                cond,
            )
        )
    if not any(s.source_split == "train" for s in samples):
        raise DatasetNotFoundError(f"no training images in {base / 'train' / GOOD}")
    return sorted(samples, key=lambda s: s.path)
