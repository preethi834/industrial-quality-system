"""Show unavailable features honestly (blueprint §2.9, step 7 gate 2).

Every page asks :func:`available` before it offers an action. A feature that is not
``enabled`` is shown with its state and reason (from ``/v1/capabilities``) and the page
offers no button for it, so nothing ever produces fake output.
"""

from __future__ import annotations

from typing import Any

LABELS = {
    "disabled": "disabled",
    "not_installed": "not installed",
    "missing_artifact": "missing artifact",
}


def status_of(caps: dict[str, Any], feature: str) -> tuple[str, str]:
    """``(state, reason)``; older services without the status map fall back to the flag."""
    flags: dict[str, bool] = caps.get("features", {})
    status: dict[str, str] = caps.get("status", {})
    reasons: dict[str, str] = caps.get("reason", {})
    if feature not in flags:
        return "not_installed", "the service does not report this feature"
    state = status.get(feature, "enabled" if flags[feature] else "disabled")
    return state, reasons.get(feature, "")


def unavailable_message(caps: dict[str, Any], feature: str) -> str | None:
    state, reason = status_of(caps, feature)
    if state == "enabled":
        return None
    service = caps.get("service", "service")
    text = f"{service} · {feature}: {LABELS.get(state, state)}"
    return f"{text} — {reason}" if reason else text


def available(st: Any, caps: dict[str, Any], feature: str) -> bool:
    """True if enabled; otherwise shows an info box with the reason and returns False."""
    msg = unavailable_message(caps, feature)
    if msg is None:
        return True
    st.info(msg)
    return False


def status_table(caps: dict[str, Any]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {"feature": [], "status": [], "reason": []}
    for feature in caps.get("features", {}):
        state, reason = status_of(caps, feature)
        rows["feature"].append(feature)
        rows["status"].append(LABELS.get(state, state))
        rows["reason"].append(reason)
    return rows
