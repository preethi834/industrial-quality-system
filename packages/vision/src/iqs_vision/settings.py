"""Vision service settings (IQS_VISION__*), on top of the shared CommonSettings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from iqs_common.settings import CommonSettings


class DiscoveryConfig(BaseModel):
    """Minimal discovery trigger (step 7). Off by default; the quality-loop stack turns it on."""

    enabled: bool = False
    config_path: Path = Path("configs/vision/discovery.yaml")
    # memory: process-local (tests, demo without Postgres); postgres: migration 0005 tables.
    store: Literal["memory", "postgres"] = "postgres"
    poll_s: float = Field(default=1.0, gt=0)


class VisionConfig(BaseModel):
    # Trained models: <models_dir>/<category>/<protocol>/{model.pt,metrics.json}
    models_dir: Path = Path("models/patchcore")
    # Which protocol's models are served. "deploy" keeps calibration images unseen.
    protocol: Literal["official", "deploy"] = "deploy"
    # Root that configs/assets.yaml targets (models/weights/...) are relative to.
    assets_root: Path = Path(".")
    assets_file: Path = Path("configs/assets.yaml")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    model_cache_size: int = Field(default=2, ge=1, le=15)
    # Text description of a defect (blueprint P2 Layer 1). "none" = disabled: the field is
    # absent from responses, never faked. A VLM needs an approval first (Rev 2 §3.9).
    vlm_provider: Literal["none"] = "none"
    discovery: DiscoveryConfig = DiscoveryConfig()
    # Cost assumptions for accept / reject / review (step 6). Used only if the file exists and
    # a model has a matching decision.json and a conformal threshold.
    cost_config: Path | None = Path("configs/costs/example_costs.yaml")


class VisionSettings(CommonSettings):
    vision: VisionConfig = VisionConfig()


@lru_cache(maxsize=1)
def get_vision_settings() -> VisionSettings:
    return VisionSettings()
