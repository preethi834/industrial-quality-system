"""Asset allowlist: parsing, checksums, approval gate, safe extraction, no implicit download."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import yaml

from iqs_common.assets import (
    ApprovalRequiredError,
    AssetError,
    AssetMissingError,
    ChecksumMismatchError,
    extract_tar,
    load_assets,
    require_approval,
    resolve_model_file,
    sha256_file,
    verify_file,
)

PAYLOAD = b"weights"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _assets(tmp_path: Path, size: int | None = len(PAYLOAD)) -> Path:
    cfg = {
        "version": 1,
        "datasets": {
            "toy": {
                "title": "Toy",
                "source": "https://example.invalid/toy",
                "filename": "toy.tar.gz",
                "sha256": SHA,
                "licence": "CC0",
                "target": "data/raw/toy",
                "approved": "2026-11-19",
            }
        },
        "models": {
            "tiny": {
                "title": "Tiny",
                "repo": "org/tiny",
                "revision": "a" * 40,
                "filename": "model.safetensors",
                "size_bytes": size,
                "sha256": SHA,
                "licence": "MIT",
                "target": "models/weights/tiny/model.safetensors",
                "approved": "2026-11-19",
            }
        },
    }
    path = tmp_path / "assets.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_load_and_lookup(tmp_path: Path) -> None:
    assets = load_assets(_assets(tmp_path))
    m = assets.model("tiny")
    assert m.url == f"https://huggingface.co/org/tiny/resolve/{'a' * 40}/model.safetensors"
    assert assets.dataset("toy").licence == "CC0"
    with pytest.raises(AssetError, match="not in the asset allowlist"):
        assets.model("other")
    with pytest.raises(AssetError, match="not in the asset allowlist"):
        assets.dataset("other")


def test_rejects_bad_checksum_format(tmp_path: Path) -> None:
    path = _assets(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["models"]["tiny"]["sha256"] = "XYZ"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_assets(path)


def test_verify_file(tmp_path: Path) -> None:
    f = tmp_path / "w.bin"
    f.write_bytes(PAYLOAD)
    assert sha256_file(f) == SHA
    verify_file(f, SHA)
    with pytest.raises(ChecksumMismatchError):
        verify_file(f, "0" * 64)


def test_approval_gate() -> None:
    require_approval(True, "x")
    with pytest.raises(ApprovalRequiredError, match="--approve"):
        require_approval(False, "x")


def test_resolve_model_file_never_downloads(tmp_path: Path) -> None:
    assets = load_assets(_assets(tmp_path))
    with pytest.raises(AssetMissingError, match="fetch_models.py tiny --approve"):
        resolve_model_file(assets, "tiny", tmp_path)
    f = tmp_path / "models/weights/tiny/model.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(PAYLOAD)
    assert resolve_model_file(assets, "tiny", tmp_path) == f
    f.write_bytes(PAYLOAD + b"!")
    with pytest.raises(ChecksumMismatchError, match="size"):
        resolve_model_file(assets, "tiny", tmp_path)


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_tar(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    _tar(archive, {"bottle/train/good/000.png": b"x", "license.txt": b"y"})
    top = extract_tar(archive, tmp_path / "out")
    assert top == ["bottle", "license.txt"]
    assert (tmp_path / "out/bottle/train/good/000.png").read_bytes() == b"x"


def test_extract_tar_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    _tar(archive, {"../escaped.txt": b"x"})
    with pytest.raises(tarfile.TarError):
        extract_tar(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
