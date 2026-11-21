"""Detection and localisation metrics (no I/O)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve of image scores (labels: 1 = defective)."""
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        raise ValueError("AUROC needs both good and defective images")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pixel-level AUROC over all pixels of all images (anomalib / MVTec convention)."""
    y = np.asarray(masks).reshape(-1).astype(np.uint8)
    if y.min() == y.max():
        raise ValueError("pixel AUROC needs both normal and anomalous pixels")
    return float(roc_auc_score(y, np.asarray(maps, dtype=np.float32).reshape(-1)))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over all thresholds and that threshold (optimistic: tuned on the test set)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(f1[i]), float(thresholds[i]) if len(thresholds) else float("inf")


def rates_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Realised false-alarm rate (good flagged) and miss rate (defect passed)."""
    labels = np.asarray(labels)
    flagged = np.asarray(scores) > threshold
    good, bad = labels == 0, labels == 1
    return {
        "false_alarm_rate": float(flagged[good].mean()) if good.any() else float("nan"),
        "miss_rate": float((~flagged[bad]).mean()) if bad.any() else float("nan"),
    }
