"""MVTec adapter, split manifests (integrity, determinism, leakage) and preprocessing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from iqs_vision.data.dataset import ManifestDataset
from iqs_vision.data.mvtec import (
    CATEGORIES,
    DatasetNotFoundError,
    available_categories,
    scan_category,
)
from iqs_vision.data.preprocess import (
    InvalidImageError,
    decode_image,
    image_to_tensor,
    load_mask,
)
from iqs_vision.data.splits import (
    ManifestEntry,
    SplitLeakageError,
    SplitManifest,
    build_manifest,
    validate_entries,
    verify_on_disk,
)
from iqs_vision.testing import make_mvtec_like, png_bytes


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return make_mvtec_like(tmp_path / "mvtec", "toy", n_train=20, n_test_good=6, n_test_defect=5)


def _manifest(root: Path, seed: int = 42) -> SplitManifest:
    return build_manifest(
        scan_category(root, "toy"), root=root, dataset="synthetic", category="toy", seed=seed
    )


def test_categories_constant() -> None:
    assert len(CATEGORIES) == 15
    assert {"bottle", "grid", "screw"} <= set(CATEGORIES)


def test_scan_category(root: Path) -> None:
    samples = scan_category(root, "toy")
    assert len(samples) == 20 + 6 + 2 * 5
    defects = [s for s in samples if s.label == 1]
    assert len(defects) == 10
    assert all(s.mask_path and s.mask_path.endswith("_mask.png") for s in defects)
    assert all(s.mask_path is None for s in samples if s.label == 0)
    assert [s.path for s in samples] == sorted(s.path for s in samples)


def test_missing_category_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="download_dataset.py mvtec_ad"):
        scan_category(tmp_path, "bottle")
    assert available_categories(tmp_path) == []


def test_missing_mask_is_an_error(root: Path) -> None:
    (root / "toy/ground_truth/spot/000_mask.png").unlink()
    with pytest.raises(DatasetNotFoundError, match="mask"):
        scan_category(root, "toy")


def test_manifest_splits_are_disjoint_and_complete(root: Path) -> None:
    m = _manifest(root)
    fit, calib, test = m.select("fit"), m.select("calib"), m.select("test")
    assert len(fit) + len(calib) + len(test) == len(m.entries) == 36
    assert len(calib) == 4  # 20 % of 20 train images
    assert {e.path for e in fit}.isdisjoint({e.path for e in calib})
    assert all(e.label == 0 for e in fit + calib)
    # calibration images are never fitted in the deploy protocol (blueprint Rev 2 §2.8)
    assert {e.path for e in m.fit_entries("deploy")}.isdisjoint({e.path for e in calib})
    assert len(m.fit_entries("official")) == 20
    sup = [m.select(sup_split=s) for s in ("sup_train", "sup_val", "sup_test")]
    assert sum(len(s) for s in sup) == len(test)
    for defect in ("good", "spot", "scratch"):  # stratified: every type in sup_test
        assert any(e.defect_type == defect for e in sup[2])


def test_manifest_is_deterministic_per_seed(root: Path) -> None:
    a, b, c = _manifest(root, 42), _manifest(root, 42), _manifest(root, 7)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_manifest_roundtrip_and_checksums(root: Path, tmp_path: Path) -> None:
    m = _manifest(root)
    path = tmp_path / "toy.json"
    m.save(path)
    loaded = SplitManifest.load(path)
    assert loaded == m and loaded.fingerprint() == m.fingerprint()
    assert verify_on_disk(m, root) == []
    (root / m.entries[0].path).write_bytes(b"changed")
    assert verify_on_disk(m, root) == [m.entries[0].path]


def test_leakage_by_content_is_detected(root: Path) -> None:
    m = _manifest(root)
    fit = m.select("fit")[0]
    test_good = next(e for e in m.select("test") if e.label == 0)
    dup = test_good.model_copy(update={"sha256": fit.sha256})
    entries = [dup if e.path == test_good.path else e for e in m.entries]
    with pytest.raises(SplitLeakageError, match="identical image content"):
        validate_entries(entries)


def test_leakage_by_path_is_detected(root: Path) -> None:
    m = _manifest(root)
    with pytest.raises(SplitLeakageError, match="twice"):
        validate_entries([*m.entries, m.entries[0]])


def test_defect_in_fit_split_is_rejected() -> None:
    bad = ManifestEntry(
        path="x.png", sha256="0" * 64, split="fit", label=1, defect_type="spot", mask="m.png"
    )
    with pytest.raises(ValueError, match="good images only"):
        validate_entries([bad])


def test_tampered_manifest_file_is_rejected(root: Path, tmp_path: Path) -> None:
    m = _manifest(root)
    path = tmp_path / "toy.json"
    m.save(path)
    text = path.read_text().replace('"split": "test"', '"split": "fit"', 1)
    path.write_text(text)
    with pytest.raises(ValueError):
        SplitManifest.load(path)


def test_preprocess_shapes_and_normalisation(root: Path) -> None:
    img = decode_image(png_bytes(64))
    t = image_to_tensor(img, 32)
    assert t.shape == (3, 32, 32) and t.dtype == torch.float32
    raw = image_to_tensor(img, 32, normalize=False)
    assert 0.0 <= raw.min() and raw.max() <= 1.0
    assert torch.allclose(t[0], (raw[0] - 0.485) / 0.229, atol=1e-6)
    mask = load_mask(root / "toy/ground_truth/spot/000_mask.png", 32)
    assert mask.shape == (32, 32) and set(mask.unique().tolist()) <= {0, 1}
    assert mask.sum() > 0
    assert load_mask(None, 16).sum() == 0


def test_grayscale_image_becomes_rgb(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "g.png"
    Image.new("L", (10, 10), 128).save(p)
    assert image_to_tensor(decode_image(p.read_bytes()), 8).shape == (3, 8, 8)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\0" * 20])
def test_decode_rejects_garbage(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_dataset_items(root: Path) -> None:
    m = _manifest(root)
    ds = ManifestDataset(m.select("test"), root, 32)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].shape == (32, 32)
    assert len(ds) == len(m.select("test"))


def test_real_manifests_match_their_schema() -> None:
    """Committed manifests (if any) parse and pass the leakage checks; checksums are
    verified against disk only when the dataset is present."""
    repo = Path(__file__).resolve().parents[3]
    manifests = sorted((repo / "eval_sets/vision/mvtec_ad").glob("*.json"))
    data_root = repo / "data/raw/mvtec_ad"
    for path in manifests:
        m = SplitManifest.load(path)
        assert m.category == path.stem
        if (data_root / m.category).is_dir():
            assert verify_on_disk(m, data_root) == []


def test_copy_does_not_change_fingerprint(root: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    assert _manifest(copy).fingerprint() == _manifest(root).fingerprint()
