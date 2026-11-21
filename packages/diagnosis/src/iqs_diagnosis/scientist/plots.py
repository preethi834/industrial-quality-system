"""Evidence figures as plotly-compatible dictionaries (JSON, no plotting library needed).

The dashboard renders them with ``plotly.graph_objects.Figure(fig)``. Box plots carry
precomputed quartiles and whiskers instead of raw values, so a figure stays small.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _box_stats(v: np.ndarray) -> dict[str, float]:
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo = float(v[v >= q1 - 1.5 * iqr].min())
    hi = float(v[v <= q3 + 1.5 * iqr].max())
    return {"q1": q1, "median": med, "q3": q3, "lowerfence": lo, "upperfence": hi}


def box_figure(
    title: str,
    y_label: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    labels: tuple[str, str] = ("rejected", "passed"),
) -> dict[str, Any]:
    traces = []
    for name, v in zip(labels, (a, b), strict=True):
        s = _box_stats(np.asarray(v, dtype=float))
        traces.append(
            {
                "type": "box",
                "name": f"{name} (n={len(v)})",
                "q1": [s["q1"]],
                "median": [s["median"]],
                "q3": [s["q3"]],
                "lowerfence": [s["lowerfence"]],
                "upperfence": [s["upperfence"]],
            }
        )
    return {"data": traces, "layout": {"title": {"text": title}, "yaxis": {"title": y_label}}}


def bar_figure(
    title: str, labels: Sequence[str], rates: Sequence[float], sizes: Sequence[int]
) -> dict[str, Any]:
    return {
        "data": [
            {
                "type": "bar",
                "x": [f"{lab} (n={n})" for lab, n in zip(labels, sizes, strict=True)],
                "y": [float(r) for r in rates],
            }
        ],
        "layout": {
            "title": {"text": title},
            "yaxis": {"title": "reject rate", "rangemode": "tozero"},
        },
    }


def trend_figure(
    title: str, hours: np.ndarray, values_um: np.ndarray, intercept: float, slope: float
) -> dict[str, Any]:
    xs = [float(h) for h in hours]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "name": "gauge check",
                "x": xs,
                "y": [round(float(v), 4) for v in values_um],
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"fit: {slope:+.3f} µm/h",
                "x": [xs[0], xs[-1]],
                "y": [intercept + slope * xs[0], intercept + slope * xs[-1]],
            },
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": "hours since first check"},
            "yaxis": {"title": "reading - certified size (µm)"},
        },
    }
