"""Display of the step-6 cost decision returned by POST /v1/inspections."""

from __future__ import annotations

from typing import Any

ACTION_LABEL = {"accept": "ACCEPT", "reject": "REJECT", "review": "SEND TO HUMAN REVIEW"}


def cost_decision_view(cd: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """Headline, one row per action (expected cost, chosen) and a caption with the hash."""
    cur = cd["currency"]
    headline = (
        f"Decision: {ACTION_LABEL[cd['action']]} (P(defect) = {cd['p_defect']:.4f} "
        f"at an assumed {cd['defect_rate_assumed']:.2%} defect rate)"
    )
    rows = [
        {
            "action": action,
            f"expected cost ({cur})": round(float(cost), 4),
            "chosen": "✓" if action == cd["action"] else "",
        }
        for action, cost in cd["expected_cost"].items()
    ]
    caption = (
        f"Policy {cd['policy']} · costs {cd['cost_config_id']} "
        f"sha256:{cd['cost_config_sha256'][:12]} · {cd['note']}"
    )
    return headline, rows, caption
