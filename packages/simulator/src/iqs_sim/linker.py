"""Image linker (step 7): give every simulated part an inspection image. **Synthetic linkage.**

MVTec-style images carry no process data, so the integration demo needs a stand-in camera:
for each part (in inspection order) the linker picks

- an image of the defect type mapped to the part's surface-defect fault, if the part truly
  has a surface defect and one of its faults is mapped (``configs/simulator/image_linkage.yaml``);
- an image of the ``background_defect`` type for surface defects with no mapped fault (the
  line's base defect rate), or a good image when no background type is configured;
- a good image otherwise.

The linker is part of the simulated *world*, like the fault injector, so it reads
``ground_truth/``. Its manifest (``<run>/linkage/``) holds the defect type per part, i.e.
ground truth, and is never given to a service: the driver posts only the image, the category,
the part ID and the run ID (access test in ``iqs_diagnosis`` / ``iqs_vision``).

Faults that act on the dimension only (fixture offset, handling error, gauge drift) cause no
visible surface defect in the simulator's quality model, so mapping them is a config error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from iqs_sim.faults import FaultType
from iqs_sim.rng import stream

LINKAGE_DIR: Final = "linkage"
MANIFEST_FILE: Final = "manifest.parquet"
INFO_FILE: Final = "linkage.json"
LINKAGE_LABEL: Final = "synthetic"
IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg", ".bmp")

# No surface effect in iqs_sim.sensors (they move the measured / true dimension only).
DIMENSION_ONLY_FAULTS: Final = frozenset(
    {
        FaultType.FIXTURE_OFFSET,
        FaultType.HANDLING_ERROR_ON_SHIFT,
        FaultType.SENSOR_CALIBRATION_DRIFT,
    }
)

MANIFEST_SCHEMA: Final = pa.schema(
    [
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("part_id", pa.string(), nullable=False),
        pa.field("inspected_ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("image", pa.string(), nullable=False),  # relative to the image pool root
        pa.field("shown_as", pa.string(), nullable=False),  # good | defect | background
        pa.field("defect_type", pa.string(), nullable=True),
        pa.field("fault_id", pa.string(), nullable=True),
        pa.field("linkage", pa.string(), nullable=False),
    ]
)


class CategoryLinkage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    good_dir: str = "train/good"  # relative to <pool>/<category>/
    defect_dir: str = "test/{defect_type}"
    fault_to_defect: dict[FaultType, str] = Field(default_factory=dict)
    background_defect: str | None = None

    @field_validator("fault_to_defect")
    @classmethod
    def _surface_faults_only(cls, value: dict[FaultType, str]) -> dict[FaultType, str]:
        bad = sorted(f.value for f in value if f in DIMENSION_ONLY_FAULTS)
        if bad:
            raise ValueError(f"dimension-only faults have no visible defect: {bad}")
        return value


class LinkageConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    seed: int = 0
    categories: dict[str, CategoryLinkage]

    @classmethod
    def load(cls, path: Path) -> LinkageConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def sha256(self) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LinkedPart:
    seq: int
    part_id: str
    inspected_ts: pd.Timestamp
    category: str
    image: str
    shown_as: str
    defect_type: str | None
    fault_id: str | None


class ImagePool:
    """Sorted image lists per (category, defect type or None = good) under ``root``."""

    def __init__(self, root: Path, category: str, cfg: CategoryLinkage) -> None:
        self.root = root
        self.category = category
        self.cfg = cfg
        self._cache: dict[str | None, list[str]] = {}

    def images(self, defect_type: str | None) -> list[str]:
        if defect_type not in self._cache:
            sub = (
                self.cfg.good_dir
                if defect_type is None
                else self.cfg.defect_dir.format(defect_type=defect_type)
            )
            folder = self.root / self.category / sub
            files = sorted(
                p.relative_to(self.root).as_posix()
                for p in folder.glob("*")
                if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
            )
            if not files:
                raise FileNotFoundError(f"no images in {folder}")
            self._cache[defect_type] = files
        return self._cache[defect_type]


def _fault_types(truth_dir: Path) -> dict[str, FaultType]:
    faults = pq.read_table(truth_dir / "faults.parquet").to_pandas()
    return {str(r.fault_id): FaultType(r.simulator_fault) for r in faults.itertuples()}


def _scenario_seed(truth_dir: Path) -> int:
    raw = json.loads((truth_dir / "scenario.json").read_text(encoding="utf-8"))
    return int(raw["seed"])


def plan_linkage(
    run_dir: Path, pool_root: Path, category: str, config: LinkageConfig
) -> Iterator[LinkedPart]:
    """One image per part, in inspection order (``completed_ts``, then ``part_id``)."""
    if category not in config.categories:
        raise KeyError(f"category {category!r} not in the linkage config")
    cfg = config.categories[category]
    pool = ImagePool(pool_root, category, cfg)
    truth_dir = run_dir / "ground_truth"
    parts = pq.read_table(
        run_dir / "observed" / "parts.parquet", columns=["part_id", "completed_ts"]
    ).to_pandas()
    truth = pq.read_table(
        truth_dir / "part_truth.parquet", columns=["part_id", "true_surface_defect", "fault_ids"]
    ).to_pandas()
    df = parts.merge(truth, on="part_id", how="left", validate="one_to_one")
    if df["true_surface_defect"].isna().any():
        raise ValueError("part_truth is missing parts that observed/parts has")
    df = df.sort_values(["completed_ts", "part_id"], kind="stable").reset_index(drop=True)
    fault_types = _fault_types(truth_dir)
    seed = _scenario_seed(truth_dir) * 1_000_003 + config.seed
    rng = stream(seed, f"linker:{category}")

    for seq, row in enumerate(df.itertuples()):
        defect_type: str | None = None
        fault_id: str | None = None
        shown_as = "good"
        if bool(row.true_surface_defect):
            for fid in row.fault_ids:
                mapped = cfg.fault_to_defect.get(fault_types[str(fid)])
                if mapped is not None:
                    defect_type, fault_id, shown_as = mapped, str(fid), "defect"
                    break
            if defect_type is None and cfg.background_defect is not None:
                defect_type, shown_as = cfg.background_defect, "background"
        images = pool.images(defect_type)
        # One draw per part whatever the outcome, so the stream stays aligned (ADR-008 rule).
        pick = int(rng.integers(0, 1 << 31)) % len(images)
        yield LinkedPart(
            seq=seq,
            part_id=str(row.part_id),
            inspected_ts=pd.Timestamp(row.completed_ts),
            category=category,
            image=images[pick],
            shown_as=shown_as,
            defect_type=defect_type,
            fault_id=fault_id,
        )


def write_linkage(
    run_dir: Path, pool_root: Path, category: str, config: LinkageConfig
) -> dict[str, object]:
    """Write ``<run>/linkage/manifest.parquet`` + ``linkage.json``; return the info dict."""
    out = run_dir / LINKAGE_DIR
    linked = list(plan_linkage(run_dir, pool_root, category, config))  # validates inputs
    if (out / MANIFEST_FILE).exists():
        raise FileExistsError(f"{out / MANIFEST_FILE} exists; linkage is written once per run")
    out.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "seq": p.seq,
                "part_id": p.part_id,
                "inspected_ts": p.inspected_ts.to_pydatetime(),
                "category": p.category,
                "image": p.image,
                "shown_as": p.shown_as,
                "defect_type": p.defect_type,
                "fault_id": p.fault_id,
                "linkage": LINKAGE_LABEL,
            }
            for p in linked
        ],
        schema=MANIFEST_SCHEMA,
    )
    pq.write_table(table, out / MANIFEST_FILE, compression="zstd", write_statistics=False)
    counts: dict[str, int] = {}
    for p in linked:
        key = p.shown_as if p.defect_type is None else f"{p.shown_as}:{p.defect_type}"
        counts[key] = counts.get(key, 0) + 1
    info: dict[str, object] = {
        "linkage": LINKAGE_LABEL,
        "note": "Synthetic image-to-part linkage for the integration demo; not plant data.",
        "category": category,
        "pool_root": str(pool_root),
        "config_sha256": config.sha256(),
        "n_parts": len(linked),
        "counts": dict(sorted(counts.items())),
    }
    (out / INFO_FILE).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def read_manifest(run_dir: Path) -> pd.DataFrame:
    return pq.read_table(run_dir / LINKAGE_DIR / MANIFEST_FILE).to_pandas()
