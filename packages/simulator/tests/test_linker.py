"""Image linker (step 7): deterministic, follows the hidden surface-defect truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pydantic import ValidationError

from iqs_common.taxonomy import Taxonomy
from iqs_sim.export import sha256_file
from iqs_sim.faults import FaultType
from iqs_sim.linker import (
    LINKAGE_DIR,
    MANIFEST_FILE,
    CategoryLinkage,
    LinkageConfig,
    read_manifest,
    write_linkage,
)
from iqs_sim.pipeline import run_scenario
from iqs_sim.scenarios import Scenario

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "simulator" / "image_linkage.yaml"


def _pool(root: Path) -> Path:
    """Tiny pool in MVTec layout: 3 good, 2 per defect type (contents irrelevant)."""
    for sub, n in [("train/good", 3), ("test/scratch", 2), ("test/spot", 2)]:
        d = root / "toy" / sub
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / f"{i:03d}.png")
    return root


@pytest.fixture(scope="module")
def tool_wear_run(
    tmp_path_factory: pytest.TempPathFactory,
    reference_scenarios: dict[str, Scenario],
    taxonomy: Taxonomy,
) -> Path:
    run = tmp_path_factory.mktemp("sim") / "run"
    run_scenario(reference_scenarios["tool_wear_m1"], run, taxonomy)
    return run


def test_repo_config_loads() -> None:
    cfg = LinkageConfig.load(CONFIG)
    assert set(cfg.categories) >= {"toy", "metal_nut"}
    assert cfg.categories["toy"].good_dir == "train/good"  # see the config comment


def test_dimension_only_fault_rejected() -> None:
    with pytest.raises(ValidationError, match="dimension-only"):
        CategoryLinkage(fault_to_defect={"fixture_offset": "scratch"})


WITH_BACKGROUND = LinkageConfig(
    categories={
        "toy": CategoryLinkage(
            good_dir="train/good",
            fault_to_defect={FaultType.ACCELERATED_TOOL_WEAR: "scratch"},
            background_defect="spot",
        )
    }
)


def test_manifest_follows_truth_and_is_deterministic(tool_wear_run: Path, tmp_path: Path) -> None:
    cfg = WITH_BACKGROUND
    pool = _pool(tmp_path / "pool")
    info = write_linkage(tool_wear_run, pool, "toy", cfg)
    m = read_manifest(tool_wear_run)
    parts = pd.read_parquet(tool_wear_run / "observed" / "parts.parquet")
    truth = pd.read_parquet(tool_wear_run / "ground_truth" / "part_truth.parquet")
    assert len(m) == len(parts) == info["n_parts"]
    assert (m["linkage"] == "synthetic").all()
    assert m["inspected_ts"].is_monotonic_increasing
    assert list(m["seq"]) == list(range(len(m)))

    merged = m.merge(truth, on="part_id")
    surface = merged["true_surface_defect"]
    assert (merged.loc[~surface, "shown_as"] == "good").all()
    assert set(merged.loc[surface, "shown_as"]) <= {"defect", "background"}
    defects = merged[merged["shown_as"] == "defect"]
    assert len(defects) > 0, "tool wear should produce mapped surface defects"
    assert (defects["defect_type"] == "scratch").all()
    assert defects["fault_id"].notna().all()
    assert merged.loc[merged["shown_as"] == "background", "defect_type"].eq("spot").all()
    # Images come from the matching folder.
    for row in m.itertuples():
        folder = f"test/{row.defect_type}" if isinstance(row.defect_type, str) else "train/good"
        assert row.image.startswith(f"toy/{folder}/")

    # Same inputs -> byte-identical manifest; second write refused.
    first = sha256_file(tool_wear_run / LINKAGE_DIR / MANIFEST_FILE)
    with pytest.raises(FileExistsError):
        write_linkage(tool_wear_run, pool, "toy", cfg)
    (tool_wear_run / LINKAGE_DIR / MANIFEST_FILE).unlink()
    write_linkage(tool_wear_run, pool, "toy", cfg)
    assert sha256_file(tool_wear_run / LINKAGE_DIR / MANIFEST_FILE) == first


def test_no_background_type_shows_good(tool_wear_run: Path, tmp_path: Path) -> None:
    cfg = LinkageConfig(categories={"toy": CategoryLinkage(good_dir="train/good")})
    run = tmp_path / "copy"
    for sub in ("observed", "ground_truth"):
        (run / sub).mkdir(parents=True)
        for f in (tool_wear_run / sub).iterdir():
            (run / sub / f.name).write_bytes(f.read_bytes())
    write_linkage(run, _pool(tmp_path / "pool"), "toy", cfg)
    assert set(read_manifest(run)["shown_as"]) == {"good"}


def test_unknown_category_and_missing_images(tool_wear_run: Path, tmp_path: Path) -> None:
    cfg = LinkageConfig.load(CONFIG)
    with pytest.raises(KeyError):
        write_linkage(tool_wear_run, tmp_path, "nope", cfg)
    with pytest.raises(FileNotFoundError):
        write_linkage(tool_wear_run, tmp_path / "empty", "toy", cfg)
