"""Streamlit entry point: `streamlit run packages/dashboard/src/iqs_dashboard/app.py`.

Home page: service status. Further pages live in pages/ (Streamlit multipage).
"""

from __future__ import annotations

import streamlit as st

from iqs_dashboard import client as api

st.set_page_config(page_title="Industrial Quality AI", layout="wide")
st.title("Industrial Quality AI")
st.caption("Defect inspection (vision) · root-cause diagnosis (from step 3)")

try:
    c = api.get_client()
    health = c.health()
    caps = c.capabilities()
    st.success(f"Vision API {health['version']} is up")
    st.subheader("Vision capabilities")
    st.table({"feature": list(caps["features"]), "available": list(caps["features"].values())})
except api.VisionAPIError as exc:
    st.error(f"Vision API error {exc.status}: {exc.detail}")
except Exception as exc:  # noqa: BLE001  (connection errors: show, do not crash)
    st.error(f"Vision API not reachable: {type(exc).__name__}")
