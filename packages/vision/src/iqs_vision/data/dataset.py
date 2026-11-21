"""Torch dataset over manifest entries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import Dataset

from iqs_vision.data.preprocess import image_to_tensor, load_image, load_mask
from iqs_vision.data.splits import ManifestEntry


class Item(TypedDict):
    image: torch.Tensor  # (3, H, W) normalised
    mask: torch.Tensor  # (H, W) uint8
    label: int
    path: str


class ManifestDataset(Dataset[Item]):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        root: Path,
        image_size: int,
        *,
        normalize: bool = True,
    ) -> None:
        self.entries = list(entries)
        self.root = root
        self.image_size = image_size
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Item:
        e = self.entries[i]
        img = image_to_tensor(
            load_image(self.root / e.path), self.image_size, normalize=self.normalize
        )
        mask = load_mask(self.root / e.mask if e.mask else None, self.image_size)
        return {"image": img, "mask": mask, "label": e.label, "path": e.path}
