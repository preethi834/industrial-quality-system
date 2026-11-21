"""Discovers trained detectors on disk and loads them on demand (small LRU cache).

Never downloads: backbone weights are resolved through the asset allowlist and a missing
file raises AssetMissingError with the fetch command.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iqs_common.assets import load_assets, resolve_model_file
from iqs_vision.detectors.patchcore import PatchCoreDetector


@dataclass(frozen=True)
class ModelInfo:
    category: str
    protocol: str
    detector: str
    version: str
    threshold: float | None
    threshold_method: str | None
    mlflow_run_id: str | None
    metrics: dict[str, float]
    path: Path


class UnknownModelError(KeyError):
    """No trained model for this category."""


class ModelRegistry:
    def __init__(
        self,
        models_dir: Path,
        protocol: str,
        *,
        assets_file: Path | None = None,
        assets_root: Path = Path("."),
        cache_size: int = 2,
    ) -> None:
        self.models_dir = models_dir
        self.protocol = protocol
        self.assets_file = assets_file
        self.assets_root = assets_root
        self.cache_size = cache_size
        self._cache: OrderedDict[str, PatchCoreDetector] = OrderedDict()
        self._lock = threading.Lock()
        self._infos = self._scan()

    def _scan(self) -> dict[str, ModelInfo]:
        infos: dict[str, ModelInfo] = {}
        if not self.models_dir.is_dir():
            return infos
        for metrics_file in sorted(self.models_dir.glob(f"*/{self.protocol}/metrics.json")):
            meta: dict[str, Any] = json.loads(metrics_file.read_text())
            model_file = metrics_file.with_name("model.pt")
            if not model_file.is_file():
                continue
            m = meta.get("metrics", {})
            run = meta.get("mlflow_run_id")
            cat = meta["category"]
            threshold = m.get("threshold_max_calib")
            infos[cat] = ModelInfo(
                category=cat,
                protocol=self.protocol,
                detector="patchcore",
                version=f"patchcore-{cat}-{self.protocol}-{(run or 'norun')[:8]}",
                threshold=threshold,
                threshold_method="max_calibration_score" if threshold is not None else None,
                mlflow_run_id=run,
                metrics={k: float(v) for k, v in m.items()},
                path=model_file,
            )
        return infos

    def categories(self) -> list[str]:
        return sorted(self._infos)

    def info(self, category: str) -> ModelInfo:
        try:
            return self._infos[category]
        except KeyError:
            raise UnknownModelError(category) from None

    def get(self, category: str) -> tuple[PatchCoreDetector, ModelInfo]:
        info = self.info(category)
        with self._lock:
            det = self._cache.get(category)
            if det is None:
                weights_name = PatchCoreDetector.read_config(info.path).weights
                weights = None
                if weights_name is not None:
                    if self.assets_file is None:
                        raise ValueError("model needs weights but no assets file is configured")
                    weights = resolve_model_file(
                        load_assets(self.assets_file), weights_name, self.assets_root
                    )
                det, _ = PatchCoreDetector.load(info.path, weights)
                self._cache[category] = det
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(category)
        return det, info
