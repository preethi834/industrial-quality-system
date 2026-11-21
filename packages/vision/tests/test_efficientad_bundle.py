"""EfficientAD CPU wrapper and run-bundle import (synthetic, random-init model, no GPU,
no downloads)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from iqs_vision.bundles import BundleError, check_against_repo, create_bundle, read_bundle
from iqs_vision.data.mvtec import scan_category
from iqs_vision.data.splits import SplitManifest, build_manifest
from iqs_vision.detectors.base import Detector
from iqs_vision.detectors.efficientad import EfficientADConfig, EfficientADDetector
from iqs_vision.pipeline import evaluate
from iqs_vision.testing import make_mvtec_like

REPO = Path(__file__).resolve().parents[3]
CFG = EfficientADConfig()  # 256 px: the autoencoder needs at least that


def _detector() -> EfficientADDetector:
    torch.manual_seed(0)
    det = EfficientADDetector(CFG)
    # A trained model carries map quantiles; a random one must not divide by zero.
    det.model.quantiles["qb_st"].data.fill_(1.0)
    det.model.quantiles["qb_ae"].data.fill_(1.0)
    return det


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SplitManifest, Path]:
    base = tmp_path_factory.mktemp("ead")
    root = make_mvtec_like(base / "mvtec", "toy", n_train=4, n_test_good=4, n_test_defect=3)
    m = build_manifest(scan_category(root, "toy"), root=root, dataset="synthetic", category="toy")
    (base / "manifests").mkdir()
    m.save(base / "manifests" / "toy.json")
    return root, m, base


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _bundle(
    tmp: Path,
    det: EfficientADDetector,
    m: SplitManifest,
    metrics: dict[str, float],
    commit: str | None = None,
) -> Path:
    det.save(tmp / "model.pt")
    return create_bundle(
        tmp / "b.zip",
        tmp / "model.pt",
        detector="efficientad",
        category="toy",
        protocol="official",
        git_commit=commit or _head(),
        manifest_fingerprint=m.fingerprint(),
        config=CFG.model_dump(mode="json"),
        metrics=metrics,
        environment={"cuda_device": "test"},
    )


def test_wrapper_shapes_and_roundtrip(tmp_path: Path) -> None:
    det = _detector()
    assert isinstance(det, Detector) and det.normalized_input is False
    x = torch.rand(2, 3, 256, 256)
    out = det.predict(x)
    assert out.scores.shape == (2,) and out.maps.shape == (2, 256, 256)
    assert np.isfinite(out.scores).all()
    det.save(tmp_path / "m.pt", metadata={"k": 1})
    loaded, meta = EfficientADDetector.load(tmp_path / "m.pt")
    assert meta == {"k": 1}
    np.testing.assert_allclose(loaded.predict(x).scores, out.scores, rtol=1e-5)
    with pytest.raises(NotImplementedError):
        det.fit(iter([]), 0)


def test_bundle_roundtrip(tmp_path: Path, data: tuple[Path, SplitManifest, Path]) -> None:
    _, m, _ = data
    info, model = read_bundle(_bundle(tmp_path, _detector(), m, {"image_auroc": 0.9}))
    assert info.category == "toy" and info.metrics == {"image_auroc": 0.9}
    assert model[:2] == b"PK"  # torch.save writes a zip archive
    check_against_repo(info, m.fingerprint(), commit_known=True)


def test_tampered_model_is_rejected(tmp_path: Path, data: tuple[Path, SplitManifest, Path]) -> None:
    _, m, _ = data
    path = _bundle(tmp_path, _detector(), m, {})
    with zipfile.ZipFile(path) as z:
        meta, model = z.read("bundle.json"), z.read("model.pt")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("bundle.json", meta)
        z.writestr("model.pt", model + b"x")
    with pytest.raises(BundleError, match="checksum"):
        read_bundle(path)


@pytest.mark.parametrize("extra", ["../evil.py", "notes.txt"])
def test_unexpected_members_are_rejected(
    tmp_path: Path, data: tuple[Path, SplitManifest, Path], extra: str
) -> None:
    _, m, _ = data
    path = _bundle(tmp_path, _detector(), m, {})
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(extra, "x")
    with pytest.raises(BundleError, match="exactly"):
        read_bundle(path)


def test_not_a_zip(tmp_path: Path) -> None:
    (tmp_path / "x.zip").write_bytes(b"nope")
    with pytest.raises(BundleError, match="not a zip"):
        read_bundle(tmp_path / "x.zip")


def test_manifest_and_commit_mismatch(
    tmp_path: Path, data: tuple[Path, SplitManifest, Path]
) -> None:
    _, m, _ = data
    info, _ = read_bundle(_bundle(tmp_path, _detector(), m, {}))
    with pytest.raises(BundleError, match="manifest fingerprint"):
        check_against_repo(info, "0" * 64, commit_known=True)
    with pytest.raises(BundleError, match="not in this repository"):
        check_against_repo(info, m.fingerprint(), commit_known=False)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("imp", REPO / "scripts/import_run_bundle.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_script_reproduces_cloud_metrics(
    tmp_path: Path, data: tuple[Path, SplitManifest, Path]
) -> None:
    root, m, base = data
    det = _detector()
    cloud, _ = evaluate(det, m, root, "official", 4)  # what the notebook would record
    bundle = _bundle(tmp_path, det, m, cloud)
    args = [
        "--data-root",
        str(root),
        "--manifests",
        str(base / "manifests"),
        "--models-root",
        str(tmp_path / "models"),
        "--no-mlflow",
    ]
    assert _script().main([str(bundle), *args]) == 0
    out = json.loads((tmp_path / "models/efficientad/toy/official/metrics.json").read_text())
    assert out["reproduced"] is True


def test_import_script_rejects_unknown_commit(
    tmp_path: Path, data: tuple[Path, SplitManifest, Path]
) -> None:
    _, m, base = data
    bundle = _bundle(tmp_path, _detector(), m, {}, commit="f" * 40)
    rc = _script().main(
        [
            str(bundle),
            "--manifests",
            str(base / "manifests"),
            "--models-root",
            str(tmp_path / "models"),
            "--no-mlflow",
        ]
    )
    assert rc == 2
    assert not (tmp_path / "models").exists()
