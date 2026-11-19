"""Read-only adapter for MVTec AD laid out as <root>/<category>/{train,test,ground_truth}.

It never downloads (blueprint Rev 2 §3.9): if the data is missing it raises
DatasetNotFoundError with the import command. Anomalib's MVTecAD datamodule is not
used because it downloads the full archive when the folder is absent.

Layout (the official archive):
    <category>/train/good/000.png
    <category>/test/<defect_type or good>/000.png
    <category>/ground_truth/<defect_type>/000_mask.png      (defects only)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

OBJECT_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
TEXTURE_CATEGORIES = ("carpet", "grid", "leather", "tile", "wood")
CATEGORIES = tuple(sorted(OBJECT_CATEGORIES + TEXTURE_CATEGORIES))
GOOD = "good"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

SourceSplit = Literal["train", "test"]


class DatasetNotFoundError(FileNotFoundError):
    """The dataset (or a category) is not on disk; the message says how to import it."""


@dataclass(frozen=True, slots=True)
class Sample:
    """One image, with paths relative to the dataset root (POSIX separators)."""

    path: str
    source_split: SourceSplit
    defect_type: str  # "good" or the MVTec defect folder name
    mask_path: str | None

    @property
    def label(self) -> int:
        """0 = good, 1 = defective."""
        return 0 if self.defect_type == GOOD else 1


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def scan_category(root: Path, category: str) -> list[Sample]:
    """All samples of one category, sorted by path. Validates the layout."""
    base = root / category
    if not (base / "train").is_dir() or not (base / "test").is_dir():
        raise DatasetNotFoundError(
            f"MVTec AD category {category!r} not found under {root}. Import the archive with: "
            "uv run python scripts/download_dataset.py mvtec_ad --archive <path> --approve"
        )
    samples: list[Sample] = []
    train_good = base / "train" / GOOD
    for img in _images(train_good):
        samples.append(Sample(img.relative_to(root).as_posix(), "train", GOOD, None))
    for type_dir in sorted(d for d in (base / "test").iterdir() if d.is_dir()):
        defect = type_dir.name
        for img in _images(type_dir):
            mask_rel: str | None = None
            if defect != GOOD:
                mask = base / "ground_truth" / defect / f"{img.stem}_mask.png"
                if not mask.is_file():
                    raise DatasetNotFoundError(f"missing ground-truth mask {mask}")
                mask_rel = mask.relative_to(root).as_posix()
            samples.append(Sample(img.relative_to(root).as_posix(), "test", defect, mask_rel))
    if not any(s.source_split == "train" for s in samples):
        raise DatasetNotFoundError(f"no training images in {train_good}")
    return sorted(samples, key=lambda s: s.path)


def available_categories(root: Path) -> list[str]:
    """Known MVTec AD categories present under root."""
    return [c for c in CATEGORIES if (root / c / "train").is_dir()]
