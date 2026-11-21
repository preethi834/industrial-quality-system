"""Quality loop (step 7): new defect clusters -> diagnosis cases -> engineer confirmation.

Process data behind vision cases are SIMULATED and linked to images synthetically.
"""

from __future__ import annotations

import streamlit as st

from iqs_dashboard import loop_client as api
from iqs_dashboard.capabilities import available

st.set_page_config(page_title="Quality loop", layout="wide")
st.title("Quality loop")
st.caption(
    "New defect cluster (vision) → root-cause case (diagnosis) → engineer confirms → "
    "both systems learn. Simulator data with synthetic image-to-part linkage."
)

try:
    c = api.get_loop_client()
    vcaps = c.vision_capabilities()
    dcaps = c.diagnosis_capabilities()
except api.LoopAPIError as exc:
    st.error(f"{exc.service} API error {exc.status}: {exc.detail}")
    st.stop()
except Exception as exc:  # noqa: BLE001  (connection errors: show, do not crash)
    st.error(f"API not reachable: {type(exc).__name__}")
    st.stop()

vision_ok = available(st, vcaps, "discovery_trigger")
cases_ok = available(st, dcaps, "cases")
if not (vision_ok and cases_ok):
    st.stop()

clusters = c.clusters()
st.subheader("Defect clusters")
if clusters:
    st.dataframe(
        [
            {
                "cluster": x["cluster_id"][:8],
                "category": x["category"],
                "status": x["status"],
                "size": x["size"],
                "first seen": x["first_seen"],
                "class": x["defect_class_name"] or "",
                "cause": x["cause_code"] or "",
            }
            for x in clusters
        ],
        hide_index=True,
    )
else:
    st.write("No cluster has been alerted yet.")

cases = c.vision_cases()
st.subheader("Cases opened from clusters")
if not cases:
    st.write("No vision case yet.")
    st.stop()
labels = {f"{x['title']} · {x['status']}": x for x in cases}
choice = st.selectbox("Case", list(labels), key="case")
case = labels[choice]
st.caption(f"Linkage: {case.get('linkage') or 'none'} · case {case['case_id']}")

if case["phase"] == "not_started":
    st.write("The agent has not run this case yet.")
    st.stop()
cs = c.cause_set(case["case_id"])
st.write(
    "Top-3 causes:",
    ", ".join(
        f"{k} ({p:.2f})" for k, p in ((t["cause_code"], t["probability"]) for t in cs["top3"])
    ),
)
if cs.get("kind") == "conformal":
    st.write("Cause set:", ", ".join(cs["causes"]) or "empty (treat as a new problem)")
    if not cs.get("guarantee_applies", True):
        st.warning(cs.get("coverage_note", "The coverage guarantee does not apply to this case."))
if cs.get("novel"):
    st.warning("No close past case: possible new problem.")

if case["phase"] != "awaiting_confirmation":
    st.success(f"Case closed ({case['status']}).")
    st.stop()
if not available(st, dcaps, "feedback"):
    st.stop()
options = list(dict.fromkeys([t["cause_code"] for t in cs["top3"]] + list(cs.get("causes", []))))
with st.form("confirm"):
    cause = st.selectbox("Confirmed cause", [*options, "none of these"], key="cause")
    engineer = st.text_input("Engineer", key="engineer")
    name = st.text_input("Name for the new defect class (optional)", key="class_name")
    submitted = st.form_submit_button("Confirm root cause")
    if submitted and not engineer:
        st.error("Enter the engineer's name.")
    elif submitted:
        out = c.confirm(
            case["case_id"], None if cause == "none of these" else cause, engineer, name or None
        )
        fb = out.get("feedback") or {}
        st.success(
            f"Confirmed. Count {fb.get('count_before')} → {fb.get('count_after')} for "
            f"{fb.get('count_context')}; graph: {fb.get('graph', {}).get('status')}."
        )
