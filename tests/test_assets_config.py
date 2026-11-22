"""The real configs/assets.yaml and the approval-gated fetch scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from iqs_common.assets import load_assets

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "configs" / "assets.yaml"


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assets_config_is_valid_and_pinned() -> None:
    assets = load_assets(ASSETS)
    assert {"wide_resnet50_2.tv_in1k", "resnet18.a1_in1k"} <= set(assets.models)
    assert {"mvtec_ad", "visa", "mvtec_ad2_vial", "mvtec_ad2_fruit_jelly"} <= set(assets.datasets)
    for name in ("visa", "mvtec_ad2_vial", "mvtec_ad2_fruit_jelly"):
        assert assets.datasets[name].size_bytes is not None  # step 2 approvals record sizes
    for m in assets.models.values():
        assert m.size_bytes is not None
        assert str(m.target).startswith("models/weights/")
    for d in assets.datasets.values():
        assert str(d.target).startswith("data/raw/")


def test_fetch_models_refuses_without_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _script("fetch_models").main(["resnet18.a1_in1k", "--root", str(tmp_path)])
    assert rc == 2
    assert "--approve" in capsys.readouterr().err
    assert not any(tmp_path.rglob("*.safetensors*"))


def test_download_dataset_refuses_without_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "x.tar.xz"
    archive.write_bytes(b"")
    rc = _script("download_dataset").main(
        ["mvtec_ad", "--archive", str(archive), "--root", str(tmp_path)]
    )
    assert rc == 2
    assert "--approve" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_download_dataset_rejects_wrong_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "x.tar.xz"
    archive.write_bytes(b"not the real archive")
    rc = _script("download_dataset").main(
        ["mvtec_ad", "--archive", str(archive), "--root", str(tmp_path), "--approve"]
    )
    assert rc == 2
    assert "sha256" in capsys.readouterr().err
    assert not (tmp_path / "data/raw/mvtec_ad").exists()
