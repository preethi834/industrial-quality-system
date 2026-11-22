"""Defect inspection: upload an image, choose the product category, see score + heatmap."""

from __future__ import annotations

import base64

import streamlit as st

from iqs_dashboard import client as api

st.set_page_config(page_title="Defect inspection", layout="wide")
st.title("Defect inspection")
st.caption(
    "PatchCore anomaly detection. With a conformal calibration the threshold bounds the "
    "false-alarm rate on good parts (α per category); without one it is provisional — the "
    "largest score among held-out good images. The note under each result says which."
)

try:
    models = api.get_client().models()
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot load models from the vision API: {exc}")
    st.stop()
if not models:
    st.warning("No detector models installed. Train one with scripts/train_detector.py.")
    st.stop()

by_cat = {m["category"]: m for m in models}
category = st.selectbox("Product category", sorted(by_cat), key="category")
model = by_cat[category]
cols = st.columns(3)
cols[0].metric("Model", model["version"])
if model.get("image_auroc") is not None:
    cols[1].metric("Test image AUROC", f"{model['image_auroc']:.3f}")
if model.get("pixel_auroc") is not None:
    cols[2].metric("Test pixel AUROC", f"{model['pixel_auroc']:.3f}")

upload = st.file_uploader("Image (PNG, JPEG or BMP)", type=["png", "jpg", "jpeg", "bmp"])
if upload is not None:
    data = upload.getvalue()
    ctype = upload.type or "image/png"
    try:
        result = api.get_client().inspect(data, category, upload.name, ctype)
    except api.VisionAPIError as exc:
        st.error(f"Inspection failed ({exc.status}): {exc.detail}")
        st.stop()
    decision = result["decision"]
    label = {"anomalous": "ANOMALOUS", "normal": "NORMAL", "undecided": "NO THRESHOLD"}[decision]
    (st.error if decision == "anomalous" else st.success)(
        f"{label} — score {result['score']:.3f} vs threshold {result['threshold']:.3f}"
        if result["threshold"] is not None
        else label
    )
    left, right = st.columns(2)
    left.image(data, caption="Input", width="stretch")
    right.image(
        base64.b64decode(result["heatmap_png_base64"]),
        caption=f"Anomaly map ({result['heatmap_scale']})",
        width="stretch",
    )
    if "description" in result:
        st.info(result["description"])
    st.caption(
        f"Inspection {result['inspection_id']} · {result['latency_ms']:.0f} ms · "
        f"{result['threshold_note']}"
    )
