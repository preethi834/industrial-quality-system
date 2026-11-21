"""Read-only adapter for VisA (Zou et al., ECCV 2022) with its official 1-class split.

It never downloads (blueprint Rev 2 §3.9): missing data raises DatasetNotFoundError with the
import command. Anomalib's Visa datamodule is not used because it downloads the archive and
rewrites the images into a new folder tree.

Layout (VisA_20220922.tar, extracted with --only <categories> split_csv):
    split_csv/1cls.csv         columns: object, split (train/test), label (normal/anomaly),
                               image, mask; paths relative to the dataset root
    <category>/Data/Images/{Normal,Anomaly}/*.JPG
    <category>/Data/Masks/Anomaly/*.png    nonzero pixels = defect (values are region ids,
                                           not 255; the mask loader treats any nonzero as 1)

VisA does not name defect types in the split file, so every anomalous image gets
defect_type "anomaly".
"""

from __future__ import annotations

import csv
from pathlib import Path

from iqs_vision.data.mvtec import GOOD, DatasetNotFoundError, Sample, SourceSplit

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
ANOMALY = "anomaly"
SPLIT_FILE = Path("split_csv") / "1cls.csv"
_SPLITS: dict[str, SourceSplit] = {"train": "train", "test": "test"}
_LABELS = {"normal": GOOD, "anomaly": ANOMALY}


def scan_category(root: Path, category: str) -> list[Sample]:
    """All samples of one category from the 1-class split, sorted by path."""
    split_file = root / SPLIT_FILE
    if not split_file.is_file() or not (root / category).is_dir():
        raise DatasetNotFoundError(
            f"VisA category {category!r} not found under {root}. Import the archive with: "
            "uv run python scripts/download_dataset.py visa --archive <path> "
            f"--only {category} split_csv --approve"
        )
    samples: list[Sample] = []
    with split_file.open(encoding="utf-8", newline="") as fh:
        rows = csv.DictReader(fh)
        expected = {"object", "split", "label", "image", "mask"}
        if rows.fieldnames is None or not expected <= set(rows.fieldnames):
            raise ValueError(f"{split_file}: expected columns {sorted(expected)}")
        for row in rows:
            if row["object"] != category:
                continue
            split, label = _SPLITS.get(row["split"]), _LABELS.get(row["label"])
            if split is None or label is None:
                raise ValueError(f"{split_file}: unexpected split/label in row {row}")
            image = Path(row["image"])
            if image.is_absolute() or ".." in image.parts or image.parts[0] != category:
                raise ValueError(f"{split_file}: unsafe or foreign image path {row['image']!r}")
            if not (root / image).is_file():
                raise DatasetNotFoundError(f"missing image {root / image}")
            mask_rel: str | None = None
            if label == ANOMALY:
                if split == "train":
                    raise ValueError(f"{split_file}: anomalous image in train: {row['image']}")
                mask = Path(row["mask"])
                if mask.is_absolute() or ".." in mask.parts or not (root / mask).is_file():
                    raise DatasetNotFoundError(f"missing or unsafe mask {row['mask']!r}")
                mask_rel = mask.as_posix()
            samples.append(Sample(image.as_posix(), split, label, mask_rel))
    if not any(s.source_split == "train" for s in samples):
        raise DatasetNotFoundError(f"no training rows for {category!r} in {split_file}")
    return sorted(samples, key=lambda s: s.path)
