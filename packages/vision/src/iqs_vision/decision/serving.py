"""Serving side of the cost-aware decision (step 6c).

`decision.json` sits next to a model and holds its Platt calibrator. It is used only when it
was fitted for that exact model run, a cost config is loaded and the model has a conformal
threshold; otherwise the inspection response carries no cost fields (never faked).
The served policy is always the guaranteed one (accept at or below q_hat).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqs_vision.decision.calibrate_proba import PlattCalibrator
from iqs_vision.decision.costs import CostConfig
from iqs_vision.decision.policy import ACTIONS, decide_guaranteed, expected_costs

DECISION_FILE = "decision.json"
DECISION_VERSION = 1


def decision_artifact(
    cal: PlattCalibrator,
    *,
    dataset: str,
    category: str,
    model_run_id: str,
    manifest_fingerprint: str,
) -> dict[str, Any]:
    return {
        "version": DECISION_VERSION,
        "dataset": dataset,
        "category": category,
        "model_run_id": model_run_id,
        "manifest_fingerprint": manifest_fingerprint,
        "fitted_on": ["sup_train", "sup_val"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "calibrator": cal.to_dict(),
    }


def load_decision(path: Path, model_run_id: str | None) -> PlattCalibrator | None:
    """The calibrator if the file exists and belongs to this model run, else None."""
    if not path.is_file():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("version") != DECISION_VERSION or body.get("model_run_id") != model_run_id:
        return None
    return PlattCalibrator.from_dict(body["calibrator"])


def cost_decision(
    score: float,
    threshold: float,
    alpha: float,
    cal: PlattCalibrator,
    cfg: CostConfig,
    cfg_sha256: str,
) -> dict[str, Any]:
    p = float(cal.predict(np.array([score]), cfg.pi)[0])
    action = int(decide_guaranteed(np.array([p]), np.array([score]), threshold, cfg)[0])
    acc, rej, rev = (float(v) for v in expected_costs(p, cfg))
    return {
        "policy": "guaranteed",
        "action": ACTIONS[action],
        "p_defect": p,
        "expected_cost": {"accept": acc, "reject": rej, "review": rev},
        "currency": cfg.currency,
        "cost_config_id": cfg.id,
        "cost_config_sha256": cfg_sha256,
        "defect_rate_assumed": cfg.pi,
        "note": (
            f"costs and the {cfg.pi:.2%} defect rate are assumptions, not measurements; "
            f"accepted whenever score <= threshold, so the false-alarm bound alpha = {alpha:g} "
            "of the threshold still applies; p_defect is a Platt calibration on labelled "
            "benchmark images"
        ),
    }
