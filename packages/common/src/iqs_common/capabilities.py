"""Capability status (blueprint §2.9, step 7 gate 2): enabled / disabled / not_installed /
missing_artifact, with a human-readable reason, derived from each service's boolean flags.

One table for both services keeps the services' entry points unchanged: a flag that is
``True`` is ``enabled``; a ``False`` flag gets the state and reason listed here (unknown
flags are ``disabled``). The dashboard shows every non-enabled feature as "not available"
with this reason, so no button ever produces fake output.

``not_installed`` is used for features whose code is not built or not installed yet (the
reason names the roadmap step); ``missing_artifact`` for features whose code exists but
whose data or model file is absent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class FeatureStatus(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    NOT_INSTALLED = "not_installed"
    MISSING_ARTIFACT = "missing_artifact"


_N, _D, _M = FeatureStatus.NOT_INSTALLED, FeatureStatus.DISABLED, FeatureStatus.MISSING_ARTIFACT

# feature -> (state when the flag is False, reason)
WHEN_OFF: Final[dict[str, tuple[FeatureStatus, str]]] = {
    # vision
    "inspections": (_M, "no trained detector model found under IQS_VISION__MODELS_DIR"),
    "vlm_description": (_D, "no VLM provider approved (IQS_VISION__VLM_PROVIDER=none)"),
    "conformal_threshold": (_N, "planned: step 2"),
    "cost_decisions": (_N, "planned: step 6"),
    "discovery_trigger": (_D, "set IQS_VISION__DISCOVERY__ENABLED=true (step 7)"),
    "discovery": (_N, "planned: step 8c (evaluated discovery, extra 'discovery')"),
    "fewshot": (_N, "planned: step 8d"),
    "edge": (_N, "planned: step 8e (extra 'edge')"),
    "counterfactual": (_N, "planned: step 8g"),
    # diagnosis
    "cases": (_M, "no case library (IQS_DIAGNOSIS__LIBRARY_PATH)"),
    "retrieval": (_M, "no case library (IQS_DIAGNOSIS__LIBRARY_PATH)"),
    "report_8d": (_M, "no case library (IQS_DIAGNOSIS__LIBRARY_PATH)"),
    "graph": (_D, "graph writes off (IQS_DIAGNOSIS__GRAPH_WRITES=none)"),
    "question_loop": (_N, "planned: step 8a"),
    "conformal_cause_set": (_M, "no calibration run (IQS_DIAGNOSIS__CALIBRATION_RUN_PATH)"),
    "hypothesis_tests": (_D, "scientist disabled in this configuration"),
    "fmea_proposals": (_N, "planned: step 8b"),
    "feedback": (_D, "feedback off or no case library (IQS_DIAGNOSIS__FEEDBACK)"),
}


def feature_status(features: dict[str, bool]) -> tuple[dict[str, str], dict[str, str]]:
    """``(status, reason)`` maps for the given boolean flags."""
    status: dict[str, str] = {}
    reason: dict[str, str] = {}
    for name, on in features.items():
        if on:
            status[name] = FeatureStatus.ENABLED.value
            continue
        state, why = WHEN_OFF.get(name, (_D, "disabled in this configuration"))
        status[name] = state.value
        reason[name] = why
    return status, reason
