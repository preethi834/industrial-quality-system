"""PatchCore detector + evaluation pipeline on tiny synthetic images (no downloads:
random-init ResNet-18, 64 px)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from iqs_vision.data.mvtec import scan_category
from iqs_vision.data.splits import SplitManifest, build_manifest
from iqs_vision.detectors.base import Detector
from iqs_vision.detectors.patchcore import PatchCoreConfig, PatchCoreDetector
from iqs_vision.evaluation.metrics import f1_max, image_auroc, pixel_auroc, rates_at_threshold
from iqs_vision.pipeline import evaluate, fit_detector, score_entries, seed_everything
from iqs_vision.testing import make_mvtec_like

CFG = PatchCoreConfig(
    backbone="resnet18", weights=None, image_size=64, coreset_sampling_ratio=0.1, batch_size=4
)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SplitManifest]:
    root = make_mvtec_like(tmp_path_factory.mktemp("d") / "mvtec", "toy", n_train=15)
    m = build_manifest(scan_category(root, "toy"), root=root, dataset="synthetic", category="toy")
    return root, m


def _fitted(root: Path, m: SplitManifest, protocol: str = "deploy") -> PatchCoreDetector:
    seed_everything(CFG.seed)
    det = PatchCoreDetector(CFG)
    fit_detector(det, m, root, protocol, CFG.batch_size)  # type: ignore[arg-type]
    return det


def test_satisfies_protocol() -> None:
    assert isinstance(PatchCoreDetector(CFG), Detector)


def test_weights_name_without_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="local path"):
        PatchCoreDetector(PatchCoreConfig(backbone="resnet18", weights="x"))


def test_unfitted_predict_fails() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PatchCoreDetector(CFG).predict(torch.zeros(1, 3, 64, 64))


def test_fit_predict_shapes_and_bank_size(data: tuple[Path, SplitManifest]) -> None:
    root, m = data
    seed_everything(0)
    det = PatchCoreDetector(CFG)
    stats = fit_detector(det, m, root, "deploy", CFG.batch_size)
    n_fit = len(m.fit_entries("deploy"))
    # resnet18 layer2 at 64 px: 8x8 patches per image; coreset keeps 10 %
    assert stats.n_images == n_fit and stats.n_patches == n_fit * 64
    assert stats.memory_bank_size == int(n_fit * 64 * 0.1)
    out = det.predict(torch.randn(3, 3, 64, 64))
    assert out.scores.shape == (3,) and out.maps.shape == (3, 64, 64)
    assert np.isfinite(out.scores).all()


def test_fit_rejects_wrong_image_count(data: tuple[Path, SplitManifest]) -> None:
    det = PatchCoreDetector(CFG)
    with pytest.raises(ValueError):
        det.fit(iter([torch.randn(2, 3, 64, 64)]), n_images=3)
    with pytest.raises(ValueError):
        det.fit(iter([torch.randn(4, 3, 64, 64)]), n_images=3)


def test_detects_synthetic_defects(data: tuple[Path, SplitManifest]) -> None:
    root, m = data
    metrics, scored = evaluate(_fitted(root, m), m, root, "deploy", CFG.batch_size)
    assert metrics["image_auroc"] > 0.9
    assert metrics["pixel_auroc"] > 0.9
    assert {"threshold_max_calib", "test_false_alarm_rate", "test_miss_rate"} <= set(metrics)
    assert len(scored.paths) == len(m.select("test"))


def test_same_seed_same_scores(data: tuple[Path, SplitManifest]) -> None:
    """Reproducibility gate on synthetic data: same seed -> identical AUROC."""
    root, m = data
    a, _ = evaluate(_fitted(root, m), m, root, "deploy", CFG.batch_size)
    b, _ = evaluate(_fitted(root, m), m, root, "deploy", CFG.batch_size)
    assert a["image_auroc"] == pytest.approx(b["image_auroc"], abs=1e-6)
    assert a["pixel_auroc"] == pytest.approx(b["pixel_auroc"], abs=1e-6)


def test_save_load_roundtrip(data: tuple[Path, SplitManifest], tmp_path: Path) -> None:
    root, m = data
    det = _fitted(root, m)
    det.save(tmp_path / "model.pt", metadata={"category": "toy"})
    loaded, meta = PatchCoreDetector.load(tmp_path / "model.pt", None)
    assert meta == {"category": "toy"}
    entries = m.select("test")[:4]
    a = score_entries(det, entries, root, 4)
    b = score_entries(loaded, entries, root, 4)
    np.testing.assert_allclose(a.scores, b.scores, rtol=1e-6)
    assert a.maps is not None and b.maps is not None
    np.testing.assert_allclose(a.maps, b.maps, rtol=1e-5)


def test_load_rejects_foreign_file(tmp_path: Path) -> None:
    torch.save({"format": 99}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="not a PatchCore model"):
        PatchCoreDetector.load(tmp_path / "x.pt", None)


def test_metrics_hand_computed() -> None:
    labels = np.array([0, 0, 1, 1])
    assert image_auroc(labels, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    masks = np.array([[[0, 1], [0, 0]]])
    assert pixel_auroc(masks, np.array([[[0.1, 0.9], [0.2, 0.3]]])) == 1.0
    f1, thr = f1_max(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == 1.0 and thr == pytest.approx(0.8)
    r = rates_at_threshold(labels, np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert r == {"false_alarm_rate": 0.5, "miss_rate": 0.5}
    with pytest.raises(ValueError):
        image_auroc(np.array([0, 0]), np.array([0.1, 0.2]))


def test_coreset_matches_anomalib_kcenter_greedy() -> None:
    """Same seed -> the same selected points as anomalib's reference implementation."""
    from anomalib.models.components import KCenterGreedy

    from iqs_vision.detectors.coreset import kcenter_greedy_indices

    x = torch.randn(3000, 128)
    torch.manual_seed(5)
    np.random.seed(5)
    ref = KCenterGreedy(embedding=x, sampling_ratio=0.05).select_coreset_idxs()
    torch.manual_seed(5)
    np.random.seed(5)
    ours = kcenter_greedy_indices(x, 0.05).tolist()
    assert set(ours) == set(ref)
    assert len(ours) == len(set(ours)) == int(3000 * 0.05)


def test_coreset_rejects_empty_selection() -> None:
    from iqs_vision.detectors.coreset import kcenter_greedy_indices

    with pytest.raises(ValueError):
        kcenter_greedy_indices(torch.randn(5, 4), 0.1)
