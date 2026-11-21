"""EfficientAD (Batzner et al., WACV 2024) — CPU inference of a model trained on a GPU.

Training needs a GPU and two downloads (ImageNette for the penalty loss, the
pre-trained teacher), so it runs in finetune/efficientad/ on Colab/Kaggle through
scripts/train_efficientad.py and comes back as a run bundle (iqs_vision.bundles).
This module only rebuilds anomalib's EfficientAdModel from the bundled state dict
(teacher, student, autoencoder, feature mean/std and map quantiles are all in it)
and scores images on CPU behind the same Detector interface as PatchCore.
Input: raw [0, 1] RGB at 256 px — the model applies ImageNet normalisation itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from iqs_vision.detectors.base import DetectorOutput, FitStats

FORMAT_VERSION = 1


class EfficientADConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_size: str = "small"  # anomalib EfficientAdModelSize value: "small" | "medium"
    teacher_out_channels: int = 384
    padding: bool = False
    pad_maps: bool = True
    image_size: int = Field(default=256, ge=64)
    max_steps: int = Field(default=70_000, ge=1)  # paper / anomalib default
    batch_size: int = 1  # EfficientAD trains with batch size 1
    seed: int = 42


def build_model(cfg: EfficientADConfig) -> Any:
    from anomalib.models.image.efficient_ad.torch_model import (
        EfficientAdModel,
        EfficientAdModelSize,
    )

    return EfficientAdModel(
        teacher_out_channels=cfg.teacher_out_channels,
        model_size=EfficientAdModelSize(cfg.model_size),
        padding=cfg.padding,
        pad_maps=cfg.pad_maps,
    )


class EfficientADDetector:
    name = "efficientad"
    normalized_input = False

    def __init__(self, cfg: EfficientADConfig, model: Any | None = None) -> None:
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.model = (model if model is not None else build_model(cfg)).eval()

    def fit(self, batches: Iterable[torch.Tensor], n_images: int) -> FitStats:
        raise NotImplementedError(
            "EfficientAD is trained on a GPU (finetune/efficientad) and imported as a run bundle"
        )

    def predict(self, images: torch.Tensor) -> DetectorOutput:
        with torch.no_grad():
            out = self.model(images)
        maps = out.anomaly_map
        if maps.shape[-2:] != images.shape[-2:]:
            maps = torch.nn.functional.interpolate(maps, size=images.shape[-2:], mode="bilinear")
        return DetectorOutput(
            scores=out.pred_score.reshape(-1).cpu().numpy().astype(np.float32),
            maps=maps[:, 0].cpu().numpy().astype(np.float32),
        )

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": FORMAT_VERSION,
                "detector": self.name,
                "config": self.cfg.model_dump(mode="json"),
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple[EfficientADDetector, dict[str, Any]]:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != FORMAT_VERSION or blob.get("detector") != cls.name:
            raise ValueError(f"{path} is not an EfficientAD model of format {FORMAT_VERSION}")
        cfg = EfficientADConfig.model_validate(blob["config"])
        model = build_model(cfg)
        model.load_state_dict(blob["state_dict"])
        return cls(cfg, model), dict(blob["metadata"])
