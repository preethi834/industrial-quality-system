"""PatchCore (Roth et al., CVPR 2022) on CPU, built on anomalib's torch components.

Reused from anomalib 2.7: the timm feature extractor, the 3×3 average pooling and
multi-layer embedding, nearest-neighbour scoring with re-weighting, the Gaussian-smoothed
anomaly map, and k-center-greedy coreset selection (with random projection).

Own code (why):
- Backbone weights come from a local, checksum-verified safetensors file, so nothing is
  downloaded at runtime (Rev 2 §3.9); anomalib's default would fetch timm weights.
- The memory bank is preallocated and filled batch by batch. Anomalib keeps a list of
  per-batch tensors and concatenates them, which needs twice the bank in RAM at the peak;
  on a 10 GB WSL VM that decides whether the largest categories fit.
- Greedy k-center selection uses a squared-distance update (detectors/coreset.py): same
  selections as anomalib's KCenterGreedy, ~2.8× faster per iteration on this CPU.
- Fit/predict work on plain tensors, without Lightning, so the API can load one file.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from iqs_vision.detectors.base import DetectorOutput, FitStats, peak_rss_mb
from iqs_vision.detectors.coreset import kcenter_greedy

FORMAT_VERSION = 1


class PatchCoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backbone: str = "wide_resnet50_2"
    weights: str | None = "wide_resnet50_2.tv_in1k"  # asset name; None = random init (tests)
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: int = Field(default=256, ge=32)
    coreset_sampling_ratio: float = Field(default=0.1, gt=0, le=1)
    num_neighbors: int = Field(default=9, ge=1)
    batch_size: int = Field(default=8, ge=1)
    seed: int = 42


def _build_model(cfg: PatchCoreConfig, weights_path: Path | None) -> Any:
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model = PatchcoreModel(
        layers=list(cfg.layers),
        backbone=cfg.backbone,
        pre_trained=False,
        num_neighbors=cfg.num_neighbors,
    )
    if weights_path is not None:
        from safetensors.torch import load_file

        extractor = model.feature_extractor.feature_extractor
        result = extractor.load_state_dict(load_file(str(weights_path)), strict=False)
        if result.missing_keys:
            raise ValueError(f"backbone weights miss {len(result.missing_keys)} keys")
    return model.eval()


class PatchCoreDetector:
    name = "patchcore"
    normalized_input = True

    def __init__(self, cfg: PatchCoreConfig, weights_path: Path | None = None) -> None:
        if cfg.weights is not None and weights_path is None:
            raise ValueError(f"config names weights {cfg.weights!r}: pass their local path")
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.weights_path = weights_path
        self.model = _build_model(cfg, weights_path)

    # -- fitting ---------------------------------------------------------------------
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) -> (B * h * w, D) patch embeddings, as in PatchcoreModel.forward."""
        m = self.model
        with torch.no_grad():
            feats = m.feature_extractor(images)
            feats = {k: m.feature_pooler(v) for k, v in feats.items()}
            emb = m.generate_embedding(feats)
        return m.reshape_embedding(emb)

    def fit(self, batches: Iterable[torch.Tensor], n_images: int) -> FitStats:
        torch.manual_seed(self.cfg.seed)
        t0 = time.perf_counter()
        bank: torch.Tensor | None = None
        row = seen = 0
        for batch in batches:
            emb = self._embed(batch)
            if bank is None:
                per_image = emb.shape[0] // batch.shape[0]
                bank = torch.empty((n_images * per_image, emb.shape[1]), dtype=torch.float32)
            if row + emb.shape[0] > bank.shape[0]:
                raise ValueError(f"more images than n_images={n_images}")
            bank[row : row + emb.shape[0]] = emb
            row += emb.shape[0]
            seen += batch.shape[0]
        if bank is None or seen != n_images:
            raise ValueError(f"expected {n_images} images, got {seen}")
        t1 = time.perf_counter()
        # The random projection draws from NumPy's global RNG, the start point from torch's.
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        coreset = kcenter_greedy(bank, self.cfg.coreset_sampling_ratio).clone()
        del bank
        self.model.memory_bank = coreset
        t2 = time.perf_counter()
        return FitStats(
            n_images=seen,
            n_patches=row,
            memory_bank_size=int(coreset.shape[0]),
            seconds_features=round(t1 - t0, 2),
            seconds_coreset=round(t2 - t1, 2),
            peak_rss_mb=round(peak_rss_mb(), 1),
        )

    # -- inference -------------------------------------------------------------------
    def predict(self, images: torch.Tensor) -> DetectorOutput:
        if self.model.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore is not fitted (empty memory bank)")
        with torch.no_grad():
            out = self.model(images)
        return DetectorOutput(
            scores=out.pred_score.reshape(-1).cpu().numpy().astype(np.float32),
            maps=out.anomaly_map[:, 0].cpu().numpy().astype(np.float32),
        )

    # -- persistence -----------------------------------------------------------------
    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": FORMAT_VERSION,
                "detector": self.name,
                "config": self.cfg.model_dump(mode="json"),
                "memory_bank": self.model.memory_bank,
                # Random-init backbones (tests) cannot be rebuilt from a weights file.
                "backbone_state": (
                    self.model.feature_extractor.state_dict() if self.cfg.weights is None else None
                ),
                "metadata": metadata or {},
            },
            path,
        )

    @staticmethod
    def read_config(path: Path) -> PatchCoreConfig:
        """Config stored in a model file (to resolve its weights before loading)."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return PatchCoreConfig.model_validate(blob["config"])

    @classmethod
    def load(
        cls, path: Path, weights_path: Path | None
    ) -> tuple[PatchCoreDetector, dict[str, Any]]:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") != FORMAT_VERSION or blob.get("detector") != cls.name:
            raise ValueError(f"{path} is not a PatchCore model of format {FORMAT_VERSION}")
        det = cls(PatchCoreConfig.model_validate(blob["config"]), weights_path)
        if blob.get("backbone_state") is not None:
            det.model.feature_extractor.load_state_dict(blob["backbone_state"])
        det.model.memory_bank = blob["memory_bank"]
        return det, dict(blob["metadata"])
