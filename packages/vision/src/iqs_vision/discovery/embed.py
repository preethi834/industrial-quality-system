"""Region embedding for the discovery trigger (step 7, owner decision D1).

The embedding of an anomalous image is the mean PatchCore backbone patch feature over the
``top_k`` patches with the highest anomaly-map values, L2-normalised. It reuses the
detector that is already loaded, so no second model (DINOv2) and no download is needed;
DINOv2 remains an option to compare in step 8c.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from iqs_vision.discovery.trigger import l2_normalise


def region_embedding(features: torch.Tensor, anomaly_map: np.ndarray, top_k: int) -> np.ndarray:
    """``features`` (D, h, w) of one image, ``anomaly_map`` (H, W) -> (D,) unit vector."""
    if features.ndim != 3:
        raise ValueError(f"features must be (D, h, w), got {tuple(features.shape)}")
    d, h, w = features.shape
    amap = torch.as_tensor(anomaly_map, dtype=torch.float32)[None, None]
    pooled = F.adaptive_avg_pool2d(amap, (h, w)).reshape(-1)  # one value per patch
    k = min(top_k, h * w)
    idx = torch.topk(pooled, k).indices
    flat = features.reshape(d, h * w)[:, idx]
    return l2_normalise(flat.mean(dim=1).cpu().numpy().astype(np.float32))
