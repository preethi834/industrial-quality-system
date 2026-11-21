"""Step 2 data: MVTec AD 2 and VisA adapters, validation -> calib, shift split, masks."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iqs_vision.data import mvtec_ad2, visa
from iqs_vision.data.mvtec import DatasetNotFoundError
from iqs_vision.data.preprocess import load_mask
from iqs_vision.data.splits import (
    ManifestEntry,
    SplitManifest,
    build_manifest,
    validate_entries,
)


def _png(path: Path, value: int = 128, size: int = 16, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(path.as_posix().encode()))
    shape = (size, size, 3) if mode == "RGB" else (size, size)
    arr = np.clip(rng.normal(value, 5, shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _mask(path: Path, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((16, 16), np.uint8)
    arr[4:8, 4:8] = fill
    Image.fromarray(arr).save(path)


def make_ad2_like(root: Path, cat: str = "vial") -> Path:
    base = root / cat
    for i in range(10):
        _png(base / "train/good" / f"{i:03d}_regular.png", mode="L")
    for i in range(4):
        _png(base / "validation/good" / f"{i:03d}_regular.png", mode="L")
    for cond in ("regular", "overexposed", "shift_1"):
        for i in range(2):
            _png(base / "test_public/good" / f"{i:03d}_{cond}.png", mode="L")
            _png(base / "test_public/bad" / f"{i:03d}_{cond}.png", value=200, mode="L")
            _mask(base / "test_public/ground_truth/bad" / f"{i:03d}_{cond}_mask.png", 255)
    _png(base / "test_private" / "000_regular.png", mode="L")  # unlabelled: ignored
    return root


def make_visa_like(root: Path, cats: tuple[str, ...] = ("cashew",)) -> Path:
    rows = ["object,split,label,image,mask"]
    for cat in cats:
        for i in range(8):
            rel = f"{cat}/Data/Images/Normal/{i:04d}.JPG"
            _png(root / rel)
            rows.append(f"{cat},{'train' if i < 6 else 'test'},normal,{rel},")
        for i in range(3):
            rel = f"{cat}/Data/Images/Anomaly/{i:03d}.JPG"
            mask = f"{cat}/Data/Masks/Anomaly/{i:03d}.png"
            _png(root / rel, value=200)
            _mask(root / mask, fill=1)  # VisA masks hold region ids, not 255
            rows.append(f"{cat},test,anomaly,{rel},{mask}")
    (root / "split_csv").mkdir(parents=True, exist_ok=True)
    (root / "split_csv/1cls.csv").write_text("\n".join(rows) + "\n")
    return root


# --- MVTec AD 2 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "cond"),
    [("000_regular", "regular"), ("012_overexposed", "overexposed"), ("3_shift_2", "shift_2")],
)
def test_ad2_condition_from_file_name(stem: str, cond: str) -> None:
    assert mvtec_ad2.condition_of(stem) == cond


@pytest.mark.parametrize("stem", ["regular", "000", "000_Regular", "000_shift_x"])
def test_ad2_unexpected_file_name_is_an_error(stem: str) -> None:
    with pytest.raises(ValueError, match="unexpected"):
        mvtec_ad2.condition_of(stem)


def test_ad2_scan_and_manifest(tmp_path: Path) -> None:
    root = make_ad2_like(tmp_path)
    samples = mvtec_ad2.scan_category(root, "vial")
    assert not any("test_private" in s.path for s in samples)
    m = build_manifest(samples, root=root, dataset="mvtec_ad2", category="vial")
    calib = m.select("calib")
    # N1(b): calib = the whole validation folder + 20 % of train; validation is never fitted.
    assert sum(e.source == "validation" for e in calib) == 4
    assert len(calib) == 4 + 2 and len(m.select("fit")) == 8
    assert all(e.source is None for e in m.fit_entries("official"))
    assert len(m.fit_entries("official")) == 10
    test, shift = m.select("test"), m.select("shift")
    assert {e.condition for e in test} == {"regular"} and len(test) == 4
    assert {e.condition for e in shift} == {"overexposed", "shift_1"} and len(shift) == 8
    assert all(e.sup_split is None for e in shift)
    assert all(e.mask for e in test + shift if e.label == 1)


def test_ad2_missing_category_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="download_dataset.py mvtec_ad2_vial"):
        mvtec_ad2.scan_category(tmp_path, "vial")


def test_ad2_missing_mask_is_an_error(tmp_path: Path) -> None:
    root = make_ad2_like(tmp_path)
    (root / "vial/test_public/ground_truth/bad/000_regular_mask.png").unlink()
    with pytest.raises(DatasetNotFoundError, match="mask"):
        mvtec_ad2.scan_category(root, "vial")


# --- VisA ---------------------------------------------------------------------------------


def test_visa_scan_reads_only_the_requested_category(tmp_path: Path) -> None:
    root = make_visa_like(tmp_path, ("cashew", "pipe_fryum"))
    samples = visa.scan_category(root, "cashew")
    assert {s.path.split("/")[0] for s in samples} == {"cashew"}
    assert sum(s.source_split == "train" for s in samples) == 6
    assert sum(s.label for s in samples) == 3
    m = build_manifest(samples, root=root, dataset="visa", category="cashew")
    assert len(m.select("fit")) + len(m.select("calib")) == 6
    assert len(m.select("test")) == 5 and not m.select("shift")


def test_visa_missing_data_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="--only cashew split_csv"):
        visa.scan_category(tmp_path, "cashew")


def test_visa_rejects_path_outside_category(tmp_path: Path) -> None:
    root = make_visa_like(tmp_path)
    csv = root / "split_csv/1cls.csv"
    csv.write_text(csv.read_text() + "cashew,test,normal,../evil.JPG,\n")
    with pytest.raises(ValueError, match="unsafe"):
        visa.scan_category(root, "cashew")


def test_visa_rejects_anomaly_in_train(tmp_path: Path) -> None:
    root = make_visa_like(tmp_path)
    csv = root / "split_csv/1cls.csv"
    text = csv.read_text().replace("cashew,test,anomaly", "cashew,train,anomaly", 1)
    csv.write_text(text)
    with pytest.raises(ValueError, match="anomalous image in train"):
        visa.scan_category(root, "cashew")


def test_region_id_masks_count_as_defect(tmp_path: Path) -> None:
    for fill in (1, 2, 255):
        p = tmp_path / f"m{fill}.png"
        _mask(p, fill)
        assert int(load_mask(p, 16).sum()) == 16


# --- manifest rules -----------------------------------------------------------------------


def _e(**kw: object) -> ManifestEntry:
    base: dict[str, object] = {
        "path": "a.png",
        "sha256": "0" * 64,
        "split": "calib",
        "label": 0,
        "defect_type": "good",
    }
    base.update(kw)
    return ManifestEntry.model_validate(base)


def test_validation_images_only_in_calib() -> None:
    with pytest.raises(ValueError, match="calib only"):
        validate_entries([_e(split="fit", source="validation")])


def test_shift_needs_a_non_regular_condition() -> None:
    with pytest.raises(ValueError, match="non-regular"):
        validate_entries([_e(split="shift", condition="regular")])
    with pytest.raises(ValueError, match="shift split"):
        validate_entries([_e(split="test", sup_split="sup_test", condition="overexposed")])


def test_optional_fields_are_omitted_from_json(tmp_path: Path) -> None:
    root = make_ad2_like(tmp_path)
    m = build_manifest(mvtec_ad2.scan_category(root, "vial"), root=root, dataset="d", category="c")
    text = m.to_json()
    assert '"source": null' not in text and '"condition": null' not in text
    assert SplitManifest.model_validate_json(text).fingerprint() == m.fingerprint()


def test_committed_mvtec_manifests_keep_their_exact_bytes() -> None:
    repo = Path(__file__).resolve().parents[3]
    for path in sorted((repo / "eval_sets/vision/mvtec_ad").glob("*.json")):
        assert SplitManifest.load(path).to_json() == path.read_text(), path
