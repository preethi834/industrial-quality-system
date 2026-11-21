"""Validity report of the conformal thresholds (step 2 gate 3) from saved scores.

    uv run python scripts/evaluate_conformal.py --dataset mvtec_ad
    uv run python scripts/evaluate_conformal.py --dataset mvtec_ad2

Reads models/patchcore/<category>/deploy/{scores,calibration}.json (scripts/calibrate.py)
for every category of the dataset that has them. Writes reports/conformal_<dataset>.md,
reports/conformal_<dataset>.json and the guarantee plot reports/figures/conformal_<dataset>.svg.
Exit code 1 if any category fails the gate (mean realised false-alarm rate > alpha + 3 SE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from iqs_vision.calibration.calibrate import CALIBRATION_FILE, SCORES_FILE, load_scores, select
from iqs_vision.calibration.validity import DEFAULT_RESPLITS, resplit_check
from iqs_vision.data import mvtec, mvtec_ad2, visa

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "mvtec_ad": mvtec.CATEGORIES,
    "visa": visa.CATEGORIES,
    "mvtec_ad2": mvtec_ad2.CATEGORIES,
}


def _fmt(x: float | None, d: int = 3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def svg_plot(rows: list[dict[str, Any]], title: str) -> str:
    """Strip plot: realised false-alarm rate of each resplit per category, alpha marked."""
    w, row_h, left, right, top = 760, 26, 120, 30, 40
    h = top + row_h * len(rows) + 40
    xmax = max(0.2, *(max(r["resplit"]["false_alarm_rates"]) for r in rows)) * 1.05

    def x(v: float) -> float:
        return left + (w - left - right) * v / xmax

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="20" font-size="14">{title}</text>',
    ]
    for t in [i / 20 for i in range(int(xmax * 20) + 1)]:
        out.append(
            f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{h - 34}" stroke="#eee"/>'
        )
        out.append(f'<text x="{x(t):.1f}" y="{h - 20}" text-anchor="middle">{t:.2f}</text>')
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        res = r["resplit"]
        out.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{r["category"]}</text>')
        out.append(
            f'<line x1="{x(res["alpha"]):.1f}" y1="{y - 10}" x2="{x(res["alpha"]):.1f}" '
            f'y2="{y + 10}" stroke="#c00" stroke-width="2"/>'
        )
        for j, v in enumerate(res["false_alarm_rates"]):
            jitter = ((j * 37) % 17 - 8) * 0.9
            out.append(
                f'<circle cx="{x(v):.1f}" cy="{y + jitter:.1f}" r="2" '
                'fill="#4a7ab8" fill-opacity="0.35"/>'
            )
        mx = x(res["mean_false_alarm"])
        out.append(
            f'<path d="M{mx - 5:.1f},{y} L{mx:.1f},{y - 6} L{mx + 5:.1f},{y} '
            f'L{mx:.1f},{y + 6} Z" fill="#111"/>'
        )
        if r.get("fixed_far") is not None:
            out.append(
                f'<rect x="{x(r["fixed_far"]) - 4:.1f}" y="{y - 4}" width="8" height="8" '
                'fill="none" stroke="#e08000" stroke-width="2"/>'
            )
    out.append(
        f'<text x="{left}" y="{h - 4}">dots: realised false-alarm rate per resplit · ◆ mean · '
        "red line: α · orange square: fixed official split</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=sorted(CATEGORIES), default="mvtec_ad")
    p.add_argument("--protocol", default="deploy")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models/patchcore")
    p.add_argument("--resplits", type=int, default=DEFAULT_RESPLITS)
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = p.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for cat in CATEGORIES[args.dataset]:
        d = args.models_dir / cat / args.protocol
        if not (d / SCORES_FILE).is_file() or not (d / CALIBRATION_FILE).is_file():
            continue
        _, records = load_scores(d / SCORES_FILE)
        cal = json.loads((d / CALIBRATION_FILE).read_text())
        alpha = cal["active"]["alpha"]
        calib = select(records, "calib")
        good, bad = select(records, "test", label=0), select(records, "test", label=1)
        res = resplit_check(calib, good, bad, alpha, resplits=args.resplits, seed=42)
        level = next(lv for lv in cal["levels"] if lv["alpha"] == alpha)
        fixed = level["marginal"]["test"]
        ad = level["alpha_delta"]
        row: dict[str, Any] = {
            "category": cat,
            "n_calib": cal["n_calib"],
            "alpha": alpha,
            "k": cal["active"]["k"],
            "threshold": cal["threshold"],
            "fixed_far": fixed.get("false_alarm_rate"),
            "fixed_false_alarms": f"{fixed.get('false_alarms')}/{fixed.get('n_good')}",
            "fixed_far_ci95": fixed.get("false_alarm_ci95"),
            "fixed_miss": fixed.get("miss_rate"),
            "alpha_delta": ad.get("k") if ad.get("feasible", True) else "infeasible",
            "resplit": res.to_dict(),
        }
        if "validation_only" in cal:
            v = cal["validation_only"]
            vl = next((lv for lv in v["levels"] if lv["feasible"]), None)
            row["validation_only"] = (
                None
                if vl is None
                else {
                    "n_calib": v["n_calib"],
                    "alpha": vl["alpha"],
                    "threshold": vl["marginal"]["threshold"],
                    "test": vl["marginal"]["test"],
                }
            )
        rows.append(row)
    if not rows:
        print(f"no calibrated {args.protocol} models for {args.dataset}", file=sys.stderr)
        return 2
    failed = [r["category"] for r in rows if not r["resplit"]["passed"]]
    lines = [
        f"# Conformal false-alarm guarantee — {args.dataset} ({args.protocol} models)",
        "",
        "Generated by `scripts/evaluate_conformal.py` from each model's scores.json and "
        "calibration.json.",
        f"Gate (step 2, 3): over R = {args.resplits} resplits of the pooled unseen good images, "
        "mean realised false-alarm rate ≤ α + 3 SE.",
        "α by decision D1 (0.02 if n ≥ 49, 0.05 if n ≥ 19, else 0.10).",
        "Fixed = official calibration images → official good test images (not pooled).",
        "Predicted = Beta-binomial mean / sd for the same n, k, m (blueprint §2.8).",
        "",
        "| Category | n | α | k | q̂ | Fixed FA | Fixed FA 95 % CI | Fixed miss "
        "| Resplit mean FA ± SE | sd (pred.) | Pred. mean | Mean miss | (α, δ=0.1) k | Gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        res = r["resplit"]
        ci = r["fixed_far_ci95"]
        gate = "pass" if res["passed"] else "**FAIL**"
        lines.append(
            f"| {r['category']} | {r['n_calib']} | {r['alpha']:.2f} | {r['k']} "
            f"| {r['threshold']:.3f} | {r['fixed_false_alarms']} = {_fmt(r['fixed_far'])} "
            f"| [{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(r['fixed_miss'])} "
            f"| {res['mean_false_alarm']:.4f} ± {res['se_false_alarm']:.4f} "
            f"| {res['sd_false_alarm']:.3f} ({res['predicted_sd']:.3f}) "
            f"| {res['predicted_mean']:.4f} | {_fmt(res['mean_miss'])} "
            f"| {r['alpha_delta']} | {gate} |"
        )
    vrows = [r for r in rows if r.get("validation_only")]
    if vrows:
        lines += [
            "",
            "## Check: validation folder only as calibration set (decision N1)",
            "",
            "| Category | n | α | q̂ | Fixed FA | Fixed miss |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in vrows:
            v = r["validation_only"]
            t = v["test"]
            lines.append(
                f"| {r['category']} | {v['n_calib']} | {v['alpha']:.2f} | {v['threshold']:.3f} "
                f"| {t.get('false_alarms')}/{t.get('n_good')} = {_fmt(t.get('false_alarm_rate'))} "
                f"| {_fmt(t.get('miss_rate'))} |"
            )
    lines += ["", f"Gate: {'all pass' if not failed else 'FAILED for ' + ', '.join(failed)}.", ""]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)
    md = "\n".join(lines)
    (args.out_dir / f"conformal_{args.dataset}.md").write_text(md)
    (args.out_dir / f"conformal_{args.dataset}.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.out_dir / "figures" / f"conformal_{args.dataset}.svg").write_text(
        svg_plot(rows, f"Realised false-alarm rate, {args.dataset} ({args.resplits} resplits)")
    )
    print(md)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
