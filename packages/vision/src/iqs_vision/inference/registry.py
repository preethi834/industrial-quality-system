"""Discovers trained detectors on disk and loads them on demand (small LRU cache).

Never downloads: backbone weights are resolved through the asset allowlist and a missing
file raises AssetMissingError with the fetch command.

Threshold per model: the conformal threshold from calibration.json (scripts/calibrate.py,
step 2) when present, else the provisional largest calibration score from metrics.json.
POST /v1/calibrations can replace it in memory (models are mounted read-only in Docker), so
a restart returns to the file on disk.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from iqs_common.assets import load_assets, resolve_model_file
from iqs_vision.calibration.calibrate import CALIBRATION_FILE, SCORES_FILE
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
    threshold_alpha: float | None = None  # conformal level; None for the provisional threshold
    calibration: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    calibration_source: str | None = None  # "file" | "api"


def _with_calibration(info: ModelInfo, cal: dict[str, Any], source: str) -> ModelInfo:
    active = cal["active"]
    return replace(
        info,
        threshold=float(active["threshold"]),
        threshold_method=f"split_conformal_{active['method']}",
        threshold_alpha=float(active["alpha"]),
        calibration=cal,
        calibration_source=source,
    )


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
            info = ModelInfo(
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
            cal_file = metrics_file.with_name(CALIBRATION_FILE)
            if cal_file.is_file():
                info = _with_calibration(info, json.loads(cal_file.read_text()), "file")
            infos[cat] = info
        return infos

    def scores_file(self, category: str) -> Path:
        return self.info(category).path.with_name(SCORES_FILE)

    def set_calibration(self, category: str, calibration: dict[str, Any]) -> ModelInfo:
        """Activate a calibration in memory (not written to disk)."""
        with self._lock:
            info = _with_calibration(self.info(category), calibration, "api")
            self._infos[category] = info
        return info

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
