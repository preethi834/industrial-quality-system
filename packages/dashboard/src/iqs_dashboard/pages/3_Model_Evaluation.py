"""Model evaluation: detector metrics and the conformal false-alarm threshold per category."""

from __future__ import annotations

import streamlit as st

from iqs_dashboard import client as api

st.set_page_config(page_title="Model evaluation", layout="wide")
st.title("Model evaluation")
st.caption(
    "Per category: test AUROC, and the conformal threshold q̂ that keeps the false-alarm "
    "rate on good parts ≤ α (split conformal, held-out good images; blueprint §2.8). The "
    "guarantee assumes production parts are exchangeable with the calibration images; it does "
    "not cover the miss rate or shifted conditions."
)

try:
    c = api.get_client()
    models = {m["category"]: m for m in c.models()}
    cals = {r["category"]: r for r in c.calibrations()}
except Exception as exc:  # noqa: BLE001  (connection errors: show, do not crash)
    st.error(f"Cannot load models from the vision API: {exc}")
    st.stop()
if not models:
    st.warning("No detector models installed. Train one with scripts/train_detector.py.")
    st.stop()


def _f(x: float | None, d: int = 3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


rows = []
for cat in sorted(models):
    m, row = models[cat], cals.get(cat, {})
    rows.append(
        {
            "category": cat,
            "image AUROC": _f(m.get("image_auroc")),
            "pixel AUROC": _f(m.get("pixel_auroc")),
            "threshold": _f(row.get("threshold")),
            "method": row.get("threshold_method") or "–",
            "α": _f(row.get("alpha"), 2),
            "n calib": row.get("n_calib") or "–",
            "source": row.get("source") or "provisional",
        }
    )
st.dataframe(rows, hide_index=True, width="stretch")

category = st.selectbox("Category", sorted(models), key="category")
try:
    detail = c.calibration(category)
except api.VisionAPIError as exc:
    st.error(f"Calibration not available ({exc.status}): {exc.detail}")
    st.stop()
cal = detail.get("calibration")
if cal is None:
    st.info("No conformal calibration for this model yet (run scripts/calibrate.py).")
else:
    st.subheader(f"{category}: n = {cal['n_calib']} held-out good images")
    level_rows = []
    for lv in cal["levels"]:
        if not lv["feasible"]:
            level_rows.append({"α": lv["alpha"], "marginal q̂": "too few images"})
            continue
        mg, ad = lv["marginal"], lv["alpha_delta"]
        t = mg["test"]
        level_rows.append(
            {
                "α": lv["alpha"],
                "marginal q̂": _f(mg["threshold"]),
                "k": mg["k"],
                "test false alarms": f"{t.get('false_alarms')}/{t.get('n_good')}",
                "predicted mean": _f(mg.get("predicted_false_alarm", {}).get("mean")),
                "test miss rate": _f(t.get("miss_rate")),
                "(α, δ) q̂": _f(ad["threshold"]) if ad.get("feasible", True) else "infeasible",
            }
        )
    st.dataframe(level_rows, hide_index=True, width="stretch")
    st.caption(
        "Test false alarms: official good test images above q̂ (one fixed split). Predicted "
        "mean: Beta-binomial expectation for the same n and k. (α, δ = 0.1): threshold that "
        "holds with probability ≥ 0.9 for this calibration set, if n allows it."
    )

with st.form("recalibrate"):
    alpha = st.select_slider("Target false-alarm rate α", options=[0.02, 0.05, 0.10], value=0.05)
    if st.form_submit_button("Recalibrate (in memory, from saved scores)"):
        try:
            r = c.calibrate(category, alpha=float(alpha))
            st.success(f"Active threshold {r['threshold']:.3f} at α = {r['alpha']} (not saved)")
        except api.VisionAPIError as exc:
            st.error(f"Recalibration refused ({exc.status}): {exc.detail}")
