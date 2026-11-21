"""Helpers for step-7 vision-case tests: a cluster event over a copied simulator run."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from iqs_common.events import NewDefectClusterEvent, NewDefectClusterPayload
from iqs_common.ids import uuid7

RUN = "run-0123456789ab"
# First scratch image of tool_wear_m1 in the linker manifest (seed 7): 2026-01-06 07:26 UTC.
FIRST_SEEN = datetime(2026, 1, 6, 7, 26, tzinfo=UTC)


def copy_run(reference_runs: dict[str, Path], runs_root: Path, name: str = "tool_wear_m1") -> Path:
    """Copy only ``observed/`` (what a service may read) under a run-<hex> name."""
    dst = runs_root / RUN
    shutil.copytree(reference_runs[name] / "observed", dst / "observed")
    return runs_root


def cluster_event(cluster_id: UUID | None = None, size: int = 5) -> NewDefectClusterEvent:
    ids = tuple(uuid7() for _ in range(size))
    return NewDefectClusterEvent(
        payload=NewDefectClusterPayload(
            cluster_id=cluster_id or uuid7(),
            category="toy",
            size=size,
            first_seen=FIRST_SEEN,
            last_seen=FIRST_SEEN + timedelta(hours=2),
            inspection_ids=ids,
            part_ids=tuple(f"P{900 + i:06d}" for i in range(size)),
            model_version="patchcore-toy-deploy-test",
            line_id="L1",
            simulator_run_id=RUN,
            linkage="synthetic_simulator",
            representative_inspection_id=ids[0],
            score_max=3.0,
            score_median=2.0,
        )
    )
