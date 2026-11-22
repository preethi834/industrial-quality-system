"""Cost-aware decisions on saved scores: expected cost per 10,000 parts (step 6b).

    nice -n 19 uv run python scripts/evaluate_costs.py

Reads eval_sets/vision/<dataset>/<category>.json and models/patchcore/<category>/deploy/
{scores,calibration}.json. No inference. Writes reports/costs_vision.{json,md} and
reports/figures/costs_{sensitivity,defect_rate}.svg. Every euro figure is computed from the
cost-assumption file whose sha256 is printed next to it; the values are assumptions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from iqs_vision.calibration.beta_bound import clopper_pearson
from iqs_vision.calibration.calibrate import CALIBRATION_FILE, SCORES_FILE, load_scores
from iqs_vision.calibration.conformal import (
    InsufficientCalibrationError,
    split_conformal_threshold,
)
from iqs_vision.data import mvtec, mvtec_ad2, visa
from iqs_vision.data.splits import SplitManifest
from iqs_vision.decision.calibrate_proba import brier, ece, fit_isotonic, fit_platt, reliability
from iqs_vision.decision.costs import CostConfig, load_cost_config
from iqs_vision.decision.evaluate import (
    MIN_PER_CLASS,
    POLICIES,
    CategoryData,
    category_data,
    ci95,
    evaluate_category,
    point_costs,
)
from iqs_vision.decision.policy import cut_offs
from iqs_vision.decision.serving import DECISION_FILE, decision_artifact

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "mvtec_ad": mvtec.CATEGORIES,
    "visa": visa.CATEGORIES,
    "mvtec_ad2": mvtec_ad2.CATEGORIES,
}
# KNOWN_ISSUES #17: official good test images not exchangeable with calib -> no guarantee claim.
NO_GUARANTEE = {("mvtec_ad", "leather"), ("mvtec_ad", "carpet"), ("mvtec_ad", "toothbrush")}
ESCAPE_RATIOS = (0.4, 1.0, 4.0, 10.0, 40.0, 100.0, 400.0, 1000.0, 4000.0)  # C_esc / C_fr
REVIEW_RATIOS = (0.001, 0.01, 0.1, 1.0, 10.0)  # C_rev / C_fr
DEFECT_RATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
ALPHAS = (0.02, 0.05, 0.10)


def calibration_quality(d: CategoryData, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """Brier / ECE on sup_test at the fit prior (sup_test has about the same defect share)."""
    out: dict[str, Any] = {}
    for name, cal in (
        ("platt", fit_platt(d.fit_scores, d.fit_labels)),
        ("isotonic", fit_isotonic(d.fit_scores, d.fit_labels)),
    ):
        p, y = cal.predict_fit_prior(d.test_scores), d.test_labels
        idx = rng.integers(0, y.size, (n_boot, y.size))
        bs = np.array([brier(p[i], y[i]) for i in idx])
        out[name] = {
            "brier": brier(p, y),
            "brier_ci95": ci95(bs),
            "ece": ece(p, y),
            "reliability": reliability(p, y),
        }
    return out


def load_all(models: Path, manifests: Path) -> list[tuple[CategoryData, dict[str, Any]]]:
    out = []
    for dataset, cats in DATASETS.items():
        for cat in cats:
            run = models / cat / "deploy"
            if (
                not (run / SCORES_FILE).exists()
                or not (manifests / dataset / f"{cat}.json").exists()
            ):
                continue
            manifest = SplitManifest.load(manifests / dataset / f"{cat}.json")
            meta, records = load_scores(run / SCORES_FILE)
            if meta.get("dataset") != dataset:
                continue
            calib = json.loads((run / CALIBRATION_FILE).read_text(encoding="utf-8"))
            out.append((category_data(manifest, meta, records), calib))
    return out


def pooled_costs(
    data: list[tuple[CategoryData, Any, float]], c: CostConfig
) -> dict[str, dict[str, float]]:
    per = [point_costs(d, cal, q, c) for d, cal, q in data]
    return {
        name: {k: float(np.mean([p[name][k] for p in per])) for k in per[0][name]}
        for name in POLICIES
    }


def sensitivity(data: list[tuple[CategoryData, Any, float]], c: CostConfig) -> dict[str, Any]:
    grid = []
    for re in ESCAPE_RATIOS:
        for rr in REVIEW_RATIOS:
            cc = c.with_values(cost_escape=re * c.c_fr, cost_review=rr * c.c_fr)
            pc = pooled_costs(data, cc)
            fixed = pc["fixed_conformal"]["cost_per_10k"]
            grid.append(
                {
                    "escape_ratio": re,
                    "review_ratio": rr,
                    **{f"{n}_cost_per_10k": pc[n]["cost_per_10k"] for n in POLICIES},
                    "guaranteed_saving_pct": 100
                    * (fixed - pc["guaranteed"]["cost_per_10k"])
                    / fixed
                    if fixed > 0
                    else 0.0,
                    "guaranteed_review_share": pc["guaranteed"]["review_share"],
                    "unconstrained_review_share": pc["unconstrained"]["review_share"],
                }
            )
    rates = []
    for pi in DEFECT_RATES:
        pc = pooled_costs(data, c.with_values(defect_rate=pi))
        rates.append(
            {
                "defect_rate": pi,
                **{f"{n}_cost_per_10k": pc[n]["cost_per_10k"] for n in POLICIES},
                "guaranteed_review_share": pc["guaranteed"]["review_share"],
            }
        )
    alphas = []
    feasible = []
    for d, cal, _ in data:
        try:
            qs = {a: split_conformal_threshold(d.calib_scores, a).threshold for a in ALPHAS}
        except InsufficientCalibrationError:
            continue
        feasible.append((d, cal, qs))
    for a in ALPHAS:
        pc = pooled_costs([(d, cal, qs[a]) for d, cal, qs in feasible], c)
        flagged = sum(int((d.good_test_all > qs[a]).sum()) for d, _, qs in feasible)
        n_good = sum(int(d.good_test_all.size) for d, _, _ in feasible)
        alphas.append(
            {
                "alpha": a,
                "guaranteed_cost_per_10k": pc["guaranteed"]["cost_per_10k"],
                "fixed_conformal_cost_per_10k": pc["fixed_conformal"]["cost_per_10k"],
                "guaranteed_review_share": pc["guaranteed"]["review_share"],
                "pooled_flag_rate_all_test_good": flagged / n_good,
            }
        )
    return {
        "cost_grid": grid,
        "defect_rate": rates,
        "alpha": alphas,
        "alpha_categories": [f"{d.dataset}/{d.category}" for d, _, _ in feasible],
    }


# --- SVG ------------------------------------------------------------------------------------


def _svg_head(w: int, h: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="12" y="20" font-size="14">{title}</text>',
    ]


def svg_heatmap(grid: list[dict[str, Any]], title: str, note: str) -> str:
    cw, ch, left, top = 74, 46, 150, 56
    w, h = left + cw * len(ESCAPE_RATIOS) + 20, top + ch * len(REVIEW_RATIOS) + 70
    out = _svg_head(w, h, title)
    vmax = max(1.0, max(abs(g["guaranteed_saving_pct"]) for g in grid))
    for g in grid:
        i, j = ESCAPE_RATIOS.index(g["escape_ratio"]), REVIEW_RATIOS.index(g["review_ratio"])
        v = g["guaranteed_saving_pct"] / vmax  # >0 saving (blue), <0 loss (red)
        r, gg, b = (
            (int(255 - 180 * v), int(255 - 120 * v), 255)
            if v >= 0
            else (255, int(255 + 160 * v), int(255 + 160 * v))
        )
        x, y = left + i * cw, top + (len(REVIEW_RATIOS) - 1 - j) * ch
        out.append(
            f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="rgb({r},{gg},{b})"/>'
        )
        out.append(
            f'<text x="{x + cw / 2 - 1}" y="{y + 18}" text-anchor="middle">'
            f"{g['guaranteed_saving_pct']:.0f}%</text>"
        )
        out.append(
            f'<text x="{x + cw / 2 - 1}" y="{y + 34}" text-anchor="middle" font-size="10" '
            f'fill="#444">rev {100 * g["guaranteed_review_share"]:.1f}%</text>'
        )
    for i, re in enumerate(ESCAPE_RATIOS):
        out.append(
            f'<text x="{left + i * cw + cw / 2}" y="{top + ch * len(REVIEW_RATIOS) + 16}" '
            f'text-anchor="middle">{re:g}</text>'
        )
    for j, rr in enumerate(REVIEW_RATIOS):
        ty = top + (len(REVIEW_RATIOS) - 1 - j) * ch + ch / 2 + 4
        out.append(f'<text x="{left - 8}" y="{ty}" text-anchor="end">{rr:g}</text>')
    out.append(
        f'<text x="{left + cw * len(ESCAPE_RATIOS) / 2}" y="{h - 30}" text-anchor="middle">'
        "C_escape / C_false_reject</text>"
    )
    out.append(f'<text x="12" y="{top + 2 * ch}" >C_review /</text>')
    out.append(f'<text x="12" y="{top + 2 * ch + 14}">C_false_reject</text>')
    out.append(f'<text x="12" y="{h - 10}" font-size="10" fill="#444">{note}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def svg_lines(rates: list[dict[str, Any]], title: str, note: str) -> str:
    w, h, left, right, top, bottom = 720, 380, 70, 170, 40, 60
    xs = [np.log10(r["defect_rate"]) for r in rates]
    ymax = max(r[f"{n}_cost_per_10k"] for r in rates for n in POLICIES) * 1.05

    def px(v: float) -> float:
        return left + (w - left - right) * (v - xs[0]) / (xs[-1] - xs[0])

    def py(v: float) -> float:
        return top + (h - top - bottom) * (1 - v / ymax)

    out = _svg_head(w, h, title)
    for t in np.linspace(0, ymax, 6):
        out.append(
            f'<line x1="{left}" y1="{py(t):.1f}" x2="{w - right}" y2="{py(t):.1f}" stroke="#eee"/>'
        )
        out.append(f'<text x="{left - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{t:,.0f}</text>')
    for r, x in zip(rates, xs, strict=True):
        out.append(
            f'<text x="{px(x):.1f}" y="{h - bottom + 16}" text-anchor="middle">'
            f"{100 * r['defect_rate']:g}%</text>"
        )
    colours = {
        "guaranteed": "#1f5fa8",
        "unconstrained": "#7a3fb0",
        "fixed_conformal": "#d07000",
        "fixed_provisional": "#888",
    }
    for k, n in enumerate(POLICIES):
        pts = " ".join(
            f"{px(x):.1f},{py(r[f'{n}_cost_per_10k']):.1f}" for r, x in zip(rates, xs, strict=True)
        )
        dash = ' stroke-dasharray="5,4"' if n.startswith("fixed") else ""
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{colours[n]}" stroke-width="2"{dash}/>'
        )
        out.append(
            f'<text x="{w - right + 10}" y="{top + 16 + 18 * k}" fill="{colours[n]}">{n}</text>'
        )
    out.append(
        f'<text x="{(left + w - right) / 2}" y="{h - 26}" text-anchor="middle">'
        "assumed production defect rate (log scale)</text>"
    )
    out.append(f'<text x="12" y="{h - 8}" font-size="10" fill="#444">{note}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- markdown -------------------------------------------------------------------------------


def _ci(v: float, ci: list[float]) -> str:
    return f"{v:,.0f} [{ci[0]:,.0f}, {ci[1]:,.0f}]"


def markdown(rep: dict[str, Any]) -> str:
    h = rep["cost_config"]["sha256"][:12]
    tag = f"(costs sha256:{h})"
    c = rep["cost_config"]["values"]
    lines = [
        "# Cost-aware decisions (step 6)",
        "",
        "Euro values are **assumptions**, not measurements; results are on public benchmark",
        "images, not plant data. Every euro figure comes from "
        f"`{rep['cost_config']['path']}` sha256 `{rep['cost_config']['sha256']}`.",
        "",
        f"Assumed: false reject €{c['cost_false_reject']:g}, escape €{c['cost_escape']:g}, "
        f"review €{c['cost_review']:g}, reviewer miss {c['review_miss_rate']:.0%}, reviewer "
        f"false reject {c['review_false_reject_rate']:.0%}, production defect rate "
        f"{c['defect_rate']:.0%}. Cut-offs on p: accept ≤ {rep['cut_offs']['lo']:.4f}, "
        f"reject ≥ {rep['cut_offs']['hi']:.4f} (unconstrained); above q̂ reject iff p ≥ "
        f"{rep['cut_offs']['p_review_vs_reject']:.4f} (guaranteed).",
        "",
        "Policies: **guaranteed** = accept iff score ≤ q̂ (step-2 conformal threshold), else",
        "reject or review by expected cost; **unconstrained** = cost-optimal on p, *no*",
        "false-alarm guarantee; **fixed_conformal** = reject iff score > q̂, no review;",
        "**fixed_provisional** = step-1 threshold (largest calibration score), no review.",
        f"Intervals: {rep['settings']['bootstrap']} stratified bootstrap resamples of sup_test",
        "(calibrator and q̂ fixed: test-sample noise only).",
        "",
        f"## Pooled (equal weight over {len(rep['categories'])} categories) {tag}",
        "",
        "| policy | € per 10,000 parts [95 % CI] | review share | flag share |",
        "|---|---|---|---|",
    ]
    for n in POLICIES:
        p = rep["pooled"][n]
        lines.append(
            f"| {n} | {_ci(p['cost_per_10k'], p['cost_per_10k_ci95'])} {tag} | "
            f"{p['review_share']:.2%} | {p['flag_share']:.2%} |"
        )
    for n in ("guaranteed", "unconstrained"):
        s = rep["pooled"]["saving"][n]
        lines.append("")
        lines.append(
            f"Saving of {n} vs fixed_conformal: €{_ci(s['vs_fixed_conformal'], s['ci95'])} "
            f"per 10,000 parts {tag}."
        )
    fa = rep["pooled"]["false_alarm_guaranteed"]
    lines += [
        "",
        f"Realised false-alarm rate of the guaranteed policy (all official good test images, "
        f"categories where the guarantee is claimed): {fa['flagged']}/{fa['n_good']} = "
        f"{fa['rate']:.3f} [{fa['ci95'][0]:.3f}, {fa['ci95'][1]:.3f}], "
        f"mean α {fa['mean_alpha']:.3f}.",
        "",
        f"## Per category {tag}",
        "",
        "| category | α | guarantee | guaranteed € [CI] | unconstrained € [CI] | fixed q̂ € [CI] "
        "| step-1 € | review share | flag rate (all good test) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["categories"]:
        p, f = r["policies"], r["false_alarm"]["guaranteed_all_test_good"]
        fc = p["fixed_conformal"]
        lines.append(
            f"| {r['dataset']}/{r['category']} | {r['alpha']:.2f} | "
            f"{'claimed' if r['guarantee_claimed'] else 'not claimed (#17)'} | "
            f"{_ci(p['guaranteed']['cost_per_10k'], p['guaranteed']['cost_per_10k_ci95'])} | "
            f"{_ci(p['unconstrained']['cost_per_10k'], p['unconstrained']['cost_per_10k_ci95'])} | "
            f"{_ci(fc['cost_per_10k'], fc['cost_per_10k_ci95'])} | "
            f"{p['fixed_provisional']['cost_per_10k']:,.0f} | "
            f"{p['guaranteed']['review_share']:.2%} | "
            f"{f['flagged']}/{f['n_good']} = {f['rate']:.3f} |"
        )
    lines += [
        "",
        "Not evaluated: "
        + "; ".join(f"{e['dataset']}/{e['category']} ({e['reason']})" for e in rep["excluded"])
        + ".",
    ]
    lines += [
        "",
        "## Probability calibration on sup_test (Brier [95 % CI], ECE; at the fit prior)",
        "",
        "| category | Platt Brier | Platt ECE | isotonic Brier | isotonic ECE |",
        "|---|---|---|---|---|",
    ]
    for r in rep["categories"]:
        q = r["calibration_quality"]
        lines.append(
            f"| {r['category']} | {q['platt']['brier']:.3f} [{q['platt']['brier_ci95'][0]:.3f}, "
            f"{q['platt']['brier_ci95'][1]:.3f}] | {q['platt']['ece']:.3f} | "
            f"{q['isotonic']['brier']:.3f} [{q['isotonic']['brier_ci95'][0]:.3f}, "
            f"{q['isotonic']['brier_ci95'][1]:.3f}] | {q['isotonic']['ece']:.3f} |"
        )
    s = rep["sensitivity"]
    lines += [
        "",
        f"## Sensitivity {tag}",
        "",
        "Pooled saving of the guaranteed policy vs fixed_conformal over cost ratios "
        "(C_false_reject = €5 fixed): `figures/costs_sensitivity.svg`; over the defect rate: "
        "`figures/costs_defect_rate.svg`.",
        "",
        "| C_esc/C_fr | C_rev/C_fr | guaranteed € | unconstrained € | fixed q̂ € | saving "
        "| review share |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in s["cost_grid"]:
        lines.append(
            f"| {g['escape_ratio']:g} | {g['review_ratio']:g} | "
            f"{g['guaranteed_cost_per_10k']:,.0f} | {g['unconstrained_cost_per_10k']:,.0f} | "
            f"{g['fixed_conformal_cost_per_10k']:,.0f} | {g['guaranteed_saving_pct']:.0f}% | "
            f"{g['guaranteed_review_share']:.2%} |"
        )
    lines += [
        "",
        "| defect rate | guaranteed € | unconstrained € | fixed q̂ € | step-1 € | review share |",
        "|---|---|---|---|---|---|",
    ]
    for g in s["defect_rate"]:
        lines.append(
            f"| {g['defect_rate']:.1%} | {g['guaranteed_cost_per_10k']:,.0f} | "
            f"{g['unconstrained_cost_per_10k']:,.0f} | {g['fixed_conformal_cost_per_10k']:,.0f} | "
            f"{g['fixed_provisional_cost_per_10k']:,.0f} | {g['guaranteed_review_share']:.2%} |"
        )
    lines += [
        "",
        f"α sensitivity (pooled over the {len(s['alpha_categories'])} categories with "
        "n_calib ≥ 49, #17 categories included; "
        "the served α stays step 2's):",
        "",
        "| α | guaranteed € | fixed q̂ € | review share | flag rate (all good test) |",
        "|---|---|---|---|---|",
    ]
    for g in s["alpha"]:
        lines.append(
            f"| {g['alpha']:.2f} | {g['guaranteed_cost_per_10k']:,.0f} | "
            f"{g['fixed_conformal_cost_per_10k']:,.0f} | {g['guaranteed_review_share']:.2%} | "
            f"{g['pooled_flag_rate_all_test_good']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--costs", type=Path, default=ROOT / "configs/costs/example_costs.yaml")
    ap.add_argument("--models", type=Path, default=ROOT / "models/patchcore")
    ap.add_argument("--manifests", type=Path, default=ROOT / "eval_sets/vision")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write-decision",
        action="store_true",
        help="also write the Platt calibrator to <models>/<category>/deploy/decision.json",
    )
    args = ap.parse_args(argv)

    cfg, digest = load_cost_config(args.costs)
    rows, excluded, reps, used = [], [], [], []
    for i, (d, calib) in enumerate(load_all(args.models, args.manifests)):
        active = calib["active"]
        if d.n_test_good < MIN_PER_CLASS or d.n_test_defect < MIN_PER_CLASS:
            excluded.append(
                {
                    "dataset": d.dataset,
                    "category": d.category,
                    "reason": f"sup_test has {d.n_test_good} good / {d.n_test_defect} defective "
                    f"images, need >= {MIN_PER_CLASS} each",
                }
            )
            continue
        rng = np.random.default_rng([args.seed, i])
        row, rep = evaluate_category(
            d, float(active["threshold"]), float(active["alpha"]), cfg, args.bootstrap, rng
        )
        row["guarantee_claimed"] = (d.dataset, d.category) not in NO_GUARANTEE
        row["calibration_quality"] = calibration_quality(d, args.bootstrap, rng)
        rows.append(row)
        reps.append(rep)
        cal = fit_platt(d.fit_scores, d.fit_labels)
        used.append((d, cal, float(active["threshold"])))
        if args.write_decision:
            art = decision_artifact(
                cal,
                dataset=d.dataset,
                category=d.category,
                model_run_id=calib["model_run_id"],
                manifest_fingerprint=calib["manifest_fingerprint"],
            )
            (args.models / d.category / "deploy" / DECISION_FILE).write_text(
                json.dumps(art, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
    if not rows:
        print("no categories with saved scores", file=sys.stderr)
        return 1

    pooled: dict[str, Any] = {}
    for n in POLICIES:
        b = np.mean([r[n] for r in reps], axis=0)
        pooled[n] = {
            "cost_per_10k": float(np.mean([r["policies"][n]["cost_per_10k"] for r in rows])),
            "cost_per_10k_ci95": ci95(b),
            "review_share": float(np.mean([r["policies"][n]["review_share"] for r in rows])),
            "flag_share": float(np.mean([r["policies"][n]["flag_share"] for r in rows])),
        }
    fixed_b = np.mean([r["fixed_conformal"] for r in reps], axis=0)
    pooled["saving"] = {
        n: {
            "vs_fixed_conformal": pooled["fixed_conformal"]["cost_per_10k"]
            - pooled[n]["cost_per_10k"],
            "ci95": ci95(fixed_b - np.mean([r[n] for r in reps], axis=0)),
        }
        for n in ("guaranteed", "unconstrained")
    }
    claimed = [r for r in rows if r["guarantee_claimed"]]
    x = sum(r["false_alarm"]["guaranteed_all_test_good"]["flagged"] for r in claimed)
    m = sum(r["false_alarm"]["guaranteed_all_test_good"]["n_good"] for r in claimed)
    pooled["false_alarm_guaranteed"] = {
        "flagged": x,
        "n_good": m,
        "rate": x / m,
        "ci95": list(clopper_pearson(x, m)),
        "mean_alpha": float(np.mean([r["alpha"] for r in claimed])),
        "categories": len(claimed),
    }

    rel = args.costs.resolve()
    rep_out: dict[str, Any] = {
        "cost_config": {
            "path": str(rel.relative_to(ROOT)) if rel.is_relative_to(ROOT) else str(rel),
            "sha256": digest,
            "id": cfg.id,
            "currency": cfg.currency,
            "values": {k: v["value"] for k, v in cfg.model_dump().items() if isinstance(v, dict)},
            "note": "all values are assumptions, not measurements",
        },
        "cut_offs": cut_offs(cfg).to_dict(),
        "settings": {
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "min_per_class": MIN_PER_CLASS,
            "pooling": "equal weight per category",
        },
        "categories": rows,
        "excluded": excluded,
        "pooled": pooled,
        "sensitivity": sensitivity(used, cfg),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)
    (args.out_dir / "costs_vision.json").write_text(
        json.dumps(rep_out, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "costs_vision.md").write_text(markdown(rep_out), encoding="utf-8")
    note = (
        f"Assumed costs, not measured. Costs file sha256:{digest[:12]}. "
        f"Pooled over {len(rows)} categories."
    )
    (args.out_dir / "figures" / "costs_sensitivity.svg").write_text(
        svg_heatmap(
            rep_out["sensitivity"]["cost_grid"],
            "Guaranteed policy: saving vs fixed conformal threshold (review share below)",
            note,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "figures" / "costs_defect_rate.svg").write_text(
        svg_lines(
            rep_out["sensitivity"]["defect_rate"], "Expected cost per 10,000 parts (EUR)", note
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {args.out_dir / 'costs_vision.md'} "
        f"({len(rows)} categories, {len(excluded)} excluded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
