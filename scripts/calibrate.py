"""Score calib / test / shift images once per deployed model and compute conformal thresholds.

    uv run python scripts/calibrate.py --categories all
    uv run python scripts/calibrate.py --dataset mvtec_ad2 --categories vial fruit_jelly

For each models/patchcore/<category>/<protocol>/model.pt (default protocol deploy):
refuses if any calibration image was fitted (official protocol always is); scores every
calib, test and shift image without anomaly maps; writes scores.json and calibration.json
next to the model and logs an MLflow run in experiment "conformal-<dataset>".
alpha follows decision D1 unless --alpha is given; delta = 0.1 (D2).
--from-scores recomputes calibration.json from an existing scores.json (no images needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqs_common.assets import load_assets, resolve_model_file
from iqs_vision.calibration.calibrate import (
    CALIBRATION_FILE,
    DEFAULT_DELTA,
    SCORES_FILE,
    ScoreRecord,
    build_calibration,
    check_calibration_disjoint,
    load_scores,
    save_scores,
)
from iqs_vision.data import mvtec, mvtec_ad2, visa
from iqs_vision.data.splits import SplitManifest
from iqs_vision.detectors.patchcore import PatchCoreDetector
from iqs_vision.pipeline import score_entries
from iqs_vision.tracking import mlflow_run

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "mvtec_ad": mvtec.CATEGORIES,
    "visa": visa.CATEGORIES,
    "mvtec_ad2": mvtec_ad2.CATEGORIES,
}


def score_model(args: argparse.Namespace, cat: str, model_dir: Path) -> tuple[dict, list]:
    manifest = SplitManifest.load(args.manifests / f"{cat}.json")
    check_calibration_disjoint(manifest, args.protocol)
    path = model_dir / "model.pt"
    cfg_weights = PatchCoreDetector.read_config(path).weights
    assets = load_assets(ROOT / "configs/assets.yaml")
    weights = resolve_model_file(assets, cfg_weights, ROOT) if cfg_weights else None
    det, meta = PatchCoreDetector.load(path, weights)
    if manifest.fingerprint() != meta["manifest_fingerprint"]:
        raise SystemExit(f"{cat}: manifest changed since training")
    entries = [e for e in manifest.entries if e.split in ("calib", "test", "shift")]
    scored = score_entries(det, entries, args.data_root, det.cfg.batch_size, keep_maps=False)
    by_path = {e.path: e for e in entries}
    records = [
        ScoreRecord(
            path=p,
            sha256=by_path[p].sha256,
            split=by_path[p].split,
            label=by_path[p].label,
            defect_type=by_path[p].defect_type,
            score=float(s),
            source=by_path[p].source,
            condition=by_path[p].condition,
        )
        for p, s in zip(scored.paths, scored.scores, strict=True)
    ]
    info = {
        "dataset": manifest.dataset,
        "category": cat,
        "protocol": args.protocol,
        "model_run_id": meta.get("mlflow_run_id"),
        "manifest_fingerprint": manifest.fingerprint(),
    }
    return info, records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=sorted(CATEGORIES), default="mvtec_ad")
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--protocol", choices=["official", "deploy"], default="deploy")
    p.add_argument("--alpha", type=float, default=None, help="override the D1 policy")
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    p.add_argument("--data-root", type=Path, help="default: data/raw/<dataset>")
    p.add_argument("--manifests", type=Path, help="default: eval_sets/vision/<dataset>")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models/patchcore")
    p.add_argument("--mlruns", type=Path, default=ROOT / "mlruns")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--from-scores", action="store_true", help="reuse scores.json")
    args = p.parse_args(argv)
    args.data_root = args.data_root or ROOT / "data/raw" / args.dataset
    args.manifests = args.manifests or ROOT / "eval_sets/vision" / args.dataset
    cats = list(CATEGORIES[args.dataset]) if args.categories == ["all"] else args.categories
    status = 0
    for cat in cats:
        model_dir = args.models_dir / cat / args.protocol
        if not (model_dir / "model.pt").is_file():
            print(f"skip {cat}: no {args.protocol} model", file=sys.stderr)
            status = status or (0 if args.categories == ["all"] else 2)
            continue
        if args.from_scores:
            info, records = load_scores(model_dir / SCORES_FILE)
            info.pop("version", None)
        else:
            info, records = score_model(args, cat, model_dir)
            save_scores(model_dir / SCORES_FILE, info, records)
        result = build_calibration(records, info, alpha=args.alpha, delta=args.delta)
        (model_dir / CALIBRATION_FILE).write_text(json.dumps(result, indent=1) + "\n")
        active = result["active"]
        summary = {
            "category": cat,
            "n_calib": result["n_calib"],
            "alpha": active["alpha"],
            "k": active["k"],
            "threshold": round(active["threshold"], 4),
        }
        level = next(lv for lv in result["levels"] if lv["alpha"] == active["alpha"])
        test = level["marginal"]["test"]
        summary |= {k: test.get(k) for k in ("false_alarm_rate", "miss_rate")}
        with mlflow_run(
            f"conformal-{info['dataset']}",
            f"{cat}-{args.protocol}-calibrate",
            tracking_dir=args.mlruns,
            params={
                "category": cat,
                "protocol": args.protocol,
                "alpha": active["alpha"],
                "delta": args.delta,
                "n_calib": result["n_calib"],
                "model_run_id": info.get("model_run_id"),
                "manifest_fingerprint": info["manifest_fingerprint"],
            },
            tags={"kind": "calibration", "category": cat},
            enabled=not args.no_mlflow,
        ) as run:
            if run is not None:
                import mlflow

                mlflow.log_metrics(
                    {
                        "threshold": active["threshold"],
                        "k": active["k"],
                        **{
                            f"test_{k}": v
                            for k, v in summary.items()
                            if k.endswith("rate") and v is not None
                        },
                    }
                )
                mlflow.log_artifact(str(model_dir / CALIBRATION_FILE))
        print(json.dumps(summary))
    return status


if __name__ == "__main__":
    sys.exit(main())
