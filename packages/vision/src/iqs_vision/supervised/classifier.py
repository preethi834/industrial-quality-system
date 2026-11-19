"""Supervised baseline: ResNet-18 fine-tuned as a binary good/defect classifier.

Why it exists: it shows what labelled defects buy (and cost) compared with PatchCore,
which never sees a defect. MVTec AD has labelled defects only in its test set, so the
manifest re-splits the test set (sup_train / sup_val / sup_test, stratified by defect
type, disjoint by path and content). The model trains on sup_train, picks its epoch on
sup_val, and is scored once on sup_test — the same images PatchCore is scored on.

CPU-friendly: 224 px, small batches, a few epochs, flips as augmentation. ImageNet
weights come from the verified local safetensors file (never downloaded at runtime).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader

from iqs_vision.data.dataset import ManifestDataset
from iqs_vision.data.splits import ManifestEntry
from iqs_vision.evaluation.metrics import image_auroc


class SupervisedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backbone: str = "resnet18"
    weights: str | None = "resnet18.a1_in1k"
    image_size: int = Field(default=224, ge=32)
    epochs: int = Field(default=12, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    seed: int = 42


def build_classifier(cfg: SupervisedConfig, weights_path: Path | None) -> nn.Module:
    import timm

    model: nn.Module = timm.create_model(cfg.backbone, pretrained=False, num_classes=1)
    if weights_path is not None:
        from safetensors.torch import load_file

        state = {k: v for k, v in load_file(str(weights_path)).items() if not k.startswith("fc.")}
        result = model.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("fc.")]
        if missing:
            raise ValueError(f"classifier weights miss {len(missing)} keys, e.g. {missing[0]}")
    return model


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random horizontal / vertical flips per image (valid for normalised tensors)."""
    flip_h = torch.rand(x.shape[0], generator=gen) < 0.5
    flip_v = torch.rand(x.shape[0], generator=gen) < 0.5
    x = x.clone()
    x[flip_h] = x[flip_h].flip(-1)
    x[flip_v] = x[flip_v].flip(-2)
    return x


@torch.no_grad()
def predict_proba(
    model: nn.Module, entries: Sequence[ManifestEntry], root: Path, cfg: SupervisedConfig
) -> tuple[np.ndarray, np.ndarray]:
    """(labels, P(defect)) for the entries, in order."""
    model.eval()
    labels, probs = [], []
    loader = DataLoader(ManifestDataset(entries, root, cfg.image_size), batch_size=cfg.batch_size)
    for b in loader:
        logits = model(b["image"]).reshape(-1)
        probs.append(torch.sigmoid(logits).numpy())
        labels.append(np.asarray(b["label"]))
    return np.concatenate(labels), np.concatenate(probs)


@dataclass
class TrainResult:
    best_epoch: int
    val_auroc: float
    history: list[dict[str, float]]
    seconds: float


def train_classifier(
    model: nn.Module,
    train: Sequence[ManifestEntry],
    val: Sequence[ManifestEntry],
    root: Path,
    cfg: SupervisedConfig,
) -> TrainResult:
    """Train; keep the weights of the epoch with the best validation AUROC."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ds = ManifestDataset(train, root, cfg.image_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    n_pos = sum(e.label for e in train)
    n_neg = len(train) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training split needs both good and defective images")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best: tuple[float, int, dict[str, Any]] | None = None
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for b in loader:
            x = _augment(b["image"], gen)
            y = torch.as_tensor(b["label"], dtype=torch.float32)
            opt.zero_grad()
            loss = loss_fn(model(x).reshape(-1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        labels, probs = predict_proba(model, val, root, cfg)
        val_auc = image_auroc(labels, probs) if len(set(labels.tolist())) == 2 else float("nan")
        history.append({"epoch": epoch, "train_loss": total / count, "val_auroc": val_auc})
        if best is None or val_auc > best[0]:
            best = (val_auc, epoch, {k: v.clone() for k, v in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[2])
    return TrainResult(
        best_epoch=best[1],
        val_auroc=best[0],
        history=history,
        seconds=round(time.perf_counter() - t0, 1),
    )
