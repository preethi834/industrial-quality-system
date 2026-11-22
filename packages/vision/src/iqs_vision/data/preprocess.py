"""Image and mask preprocessing shared by training, evaluation and the API.

Images: decode → RGB (MVTec has grayscale categories) → resize to size×size (bilinear,
antialiased) → float tensor in [0, 1] → ImageNet normalisation. Masks: nearest-neighbour
resize → {0, 1}. Anomalib's PatchCore default is the same (Resize 256, Normalize,
no centre crop).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Decompression-bomb guard for uploads: MVTec's largest images are 1024×1024.
MAX_PIXELS = 4096 * 4096


class InvalidImageError(ValueError):
    """The bytes are not a decodable image (or are absurdly large)."""


def decode_image(data: bytes) -> Image.Image:
    """Decode untrusted bytes into an RGB PIL image."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise InvalidImageError(f"image too large: {img.width}x{img.height}")
            img.load()
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"cannot decode image: {type(exc).__name__}") from exc


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_to_tensor(img: Image.Image, size: int, *, normalize: bool = True) -> torch.Tensor:
    """(3, size, size) float32 tensor."""
    resized = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
    return t


def load_mask(path: Path | None, size: int) -> torch.Tensor:
    """(size, size) uint8 tensor of {0, 1}; all zeros when there is no mask (good image).

    Any nonzero pixel is a defect: MVTec AD / AD 2 masks are 0/255, VisA masks hold small
    region ids (1, 2, ...). Nearest-neighbour resizing keeps the values, so 0/255 masks give
    exactly the same result as the former `> 127` rule.
    """
    if path is None:
        return torch.zeros((size, size), dtype=torch.uint8)
    with Image.open(path) as m:
        resized = m.convert("L").resize((size, size), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(resized) > 0).astype(np.uint8))
