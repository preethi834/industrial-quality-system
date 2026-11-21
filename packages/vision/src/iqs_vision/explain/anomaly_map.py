"""Heatmap rendering for anomaly maps (no model code)."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from PIL import Image


def normalize_map(amap: np.ndarray, threshold: float | None) -> np.ndarray:
    """Map scores to [0, 1] for display.

    With a threshold, the scale is absolute: 0.5 = the image threshold, 1.0 = twice it,
    so heatmaps are comparable across images and a normal part stays cool. Without a
    threshold, fall back to min-max per image (relative only).
    """
    amap = np.asarray(amap, dtype=np.float32)
    if threshold is not None and threshold > 0:
        return np.clip(amap / (2.0 * threshold), 0.0, 1.0)
    lo, hi = float(amap.min()), float(amap.max())
    return np.zeros_like(amap) if hi <= lo else (amap - lo) / (hi - lo)


def overlay(image: Image.Image, norm_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 overlay of a JET-coloured map on the image (at the map's resolution)."""
    h, w = norm_map.shape
    base = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR))
    colored = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (base * (1 - alpha) + colored * alpha).astype(np.uint8)


def png_base64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
