"""Supervised baseline on synthetic data (random-init ResNet-18, no download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqs_vision.data.mvtec import scan_category
from iqs_vision.data.splits import SplitManifest, build_manifest
from iqs_vision.evaluation.metrics import image_auroc
from iqs_vision.supervised.classifier import (
    SupervisedConfig,
    build_classifier,
    predict_proba,
    train_classifier,
)
from iqs_vision.testing import make_mvtec_like

CFG = SupervisedConfig(weights=None, image_size=64, epochs=3, batch_size=8, lr=1e-3)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SplitManifest]:
    root = make_mvtec_like(
        tmp_path_factory.mktemp("s") / "mvtec", "toy", n_train=4, n_test_good=20, n_test_defect=10
    )
    m = build_manifest(scan_category(root, "toy"), root=root, dataset="synthetic", category="toy")
    return root, m


def _run(root: Path, m: SplitManifest) -> tuple[float, np.ndarray]:
    model = build_classifier(CFG, None)
    res = train_classifier(
        model, m.select(sup_split="sup_train"), m.select(sup_split="sup_val"), root, CFG
    )
    assert 1 <= res.best_epoch <= CFG.epochs and len(res.history) == CFG.epochs
    labels, probs = predict_proba(model, m.select(sup_split="sup_test"), root, CFG)
    assert ((probs >= 0) & (probs <= 1)).all()
    return image_auroc(labels, probs), probs


def test_trains_and_scores(data: tuple[Path, SplitManifest]) -> None:
    auc, _ = _run(*data)
    assert auc >= 0.5  # tiny random-init net, 3 epochs: only a sanity floor


def test_same_seed_same_predictions(data: tuple[Path, SplitManifest]) -> None:
    _, a = _run(*data)
    _, b = _run(*data)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_needs_both_classes(data: tuple[Path, SplitManifest]) -> None:
    root, m = data
    goods = [e for e in m.select(sup_split="sup_train") if e.label == 0]
    with pytest.raises(ValueError, match="both"):
        train_classifier(build_classifier(CFG, None), goods, goods, root, CFG)


def test_supervised_splits_disjoint(data: tuple[Path, SplitManifest]) -> None:
    _, m = data
    sets = [{e.sha256 for e in m.select(sup_split=s)} for s in ("sup_train", "sup_val", "sup_test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    fit = {e.sha256 for e in m.select("fit")}
    assert not (fit & (sets[0] | sets[1] | sets[2]))


REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("name", "kind"),
    [("resnet18.a1_in1k", "classifier"), ("wide_resnet50_2.tv_in1k", "patchcore")],
)
def test_local_pretrained_weights_load_without_missing_keys(name: str, kind: str) -> None:
    """Runs only where the approved weights were fetched (not in CI)."""
    from iqs_common.assets import AssetMissingError, load_assets, resolve_model_file

    try:
        path = resolve_model_file(load_assets(REPO / "configs/assets.yaml"), name, REPO)
    except AssetMissingError:
        pytest.skip(f"{name} not fetched")
    if kind == "classifier":
        build_classifier(SupervisedConfig(), path)
    else:
        from iqs_vision.detectors.patchcore import PatchCoreConfig, PatchCoreDetector

        PatchCoreDetector(PatchCoreConfig(), path)
