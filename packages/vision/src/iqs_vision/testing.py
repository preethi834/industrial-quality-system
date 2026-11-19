"""Tiny synthetic MVTec-style datasets for tests and CI (no real data needed).

Good images: a fixed stripe texture plus mild noise. Defects: a bright square ("spot")
or a dark diagonal line ("scratch"), with matching ground-truth masks. The layout is
identical to MVTec AD, so the real adapter, manifests and detectors run unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _good(rng: np.random.Generator, size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    base = 110 + 40 * np.sin(x / 3.0) + 20 * np.cos(y / 5.0)
    img = base[..., None] + rng.normal(0, 4, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _defect(rng: np.random.Generator, size: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    img = _good(rng, size)
    mask = np.zeros((size, size), dtype=np.uint8)
    s = max(4, size // 6)
    r, c = rng.integers(s, size - 2 * s, size=2)
    if kind == "spot":
        mask[r : r + s, c : c + s] = 255
        img[mask > 0] = 250
    else:  # scratch
        for k in range(2 * s):
            rr, cc = min(size - 1, r + k), min(size - 1, c + k)
            mask[rr, max(0, cc - 1) : cc + 2] = 255
        img[mask > 0] = 10
    return img, mask


def make_mvtec_like(
    root: Path,
    category: str = "toy",
    *,
    n_train: int = 20,
    n_test_good: int = 6,
    n_test_defect: int = 6,
    size: int = 64,
    seed: int = 0,
    defect_types: tuple[str, ...] = ("spot", "scratch"),
) -> Path:
    """Write <root>/<category>/{train,test,ground_truth}; returns root."""
    rng = np.random.default_rng(seed)
    base = root / category
    (base / "train" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        Image.fromarray(_good(rng, size)).save(base / "train" / "good" / f"{i:03d}.png")
    (base / "test" / "good").mkdir(parents=True, exist_ok=True)
    for i in range(n_test_good):
        Image.fromarray(_good(rng, size)).save(base / "test" / "good" / f"{i:03d}.png")
    for kind in defect_types:
        (base / "test" / kind).mkdir(parents=True, exist_ok=True)
        (base / "ground_truth" / kind).mkdir(parents=True, exist_ok=True)
        for i in range(n_test_defect):
            img, mask = _defect(rng, size, kind)
            Image.fromarray(img).save(base / "test" / kind / f"{i:03d}.png")
            Image.fromarray(mask).save(base / "ground_truth" / kind / f"{i:03d}_mask.png")
    return root


def png_bytes(size: int = 64, *, defect: str | None = None, seed: int = 1) -> bytes:
    """One encoded PNG (good, or with a 'spot'/'scratch' defect), e.g. for API tests."""
    import io

    rng = np.random.default_rng(seed)
    arr = _good(rng, size) if defect is None else _defect(rng, size, defect)[0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_tiny_model(models_dir: Path, data_root: Path, category: str = "toy") -> Path:
    """Fit a tiny PatchCore (random-init ResNet-18, 64 px) on synthetic data and save it
    the way scripts/train_detector.py does (deploy protocol). Returns the model dir."""
    import json

    from iqs_vision.data.mvtec import scan_category
    from iqs_vision.data.splits import build_manifest
    from iqs_vision.detectors.patchcore import PatchCoreConfig, PatchCoreDetector
    from iqs_vision.pipeline import evaluate, fit_detector, seed_everything

    if not (data_root / category).is_dir():
        make_mvtec_like(data_root, category, n_train=15)
    manifest = build_manifest(
        scan_category(data_root, category), root=data_root, dataset="synthetic", category=category
    )
    cfg = PatchCoreConfig(backbone="resnet18", weights=None, image_size=64, batch_size=4)
    seed_everything(cfg.seed)
    det = PatchCoreDetector(cfg)
    fit_detector(det, manifest, data_root, "deploy", cfg.batch_size)
    metrics, _ = evaluate(det, manifest, data_root, "deploy", cfg.batch_size)
    out = models_dir / category / "deploy"
    meta = {
        "category": category,
        "protocol": "deploy",
        "manifest_fingerprint": manifest.fingerprint(),
        "mlflow_run_id": "test0000run",
        "metrics": metrics,
    }
    det.save(out / "model.pt", metadata=meta)
    (out / "metrics.json").write_text(json.dumps(meta))
    return out
