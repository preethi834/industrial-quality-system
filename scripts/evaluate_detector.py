"""Re-evaluate saved PatchCore models (models/patchcore/<category>/<protocol>/model.pt).

    uv run python scripts/evaluate_detector.py --categories bottle --protocol official

Loads the model file (no refit), scores the test split, logs an MLflow 'eval' run and
checks that the metrics equal the ones recorded at training time (|Δ| ≤ 1e-6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqs_common.assets import load_assets, resolve_model_file
from iqs_vision.data.splits import SplitManifest
from iqs_vision.detectors.patchcore import PatchCoreDetector
from iqs_vision.pipeline import evaluate
from iqs_vision.tracking import mlflow_run

ROOT = Path(__file__).resolve().parents[1]
COMPARED = ("image_auroc", "pixel_auroc", "sup_test_image_auroc", "threshold_max_calib")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--protocol", choices=["official", "deploy"], default="official")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/mvtec_ad")
    p.add_argument("--manifests", type=Path, default=ROOT / "eval_sets/vision/mvtec_ad")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models/patchcore")
    p.add_argument("--mlruns", type=Path, default=ROOT / "mlruns")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    assets = load_assets(ROOT / "configs/assets.yaml")
    status = 0
    for cat in args.categories:
        path = args.models_dir / cat / args.protocol / "model.pt"
        cfg_weights = PatchCoreDetector.read_config(path).weights
        weights = resolve_model_file(assets, cfg_weights, ROOT) if cfg_weights else None
        det, meta = PatchCoreDetector.load(path, weights)
        manifest = SplitManifest.load(args.manifests / f"{cat}.json")
        if manifest.fingerprint() != meta["manifest_fingerprint"]:
            print(f"error: {cat}: manifest changed since training", file=sys.stderr)
            status = 2
            continue
        with mlflow_run(
            "patchcore-mvtec_ad",
            f"{cat}-{args.protocol}-eval",
            tracking_dir=args.mlruns,
            params={
                "category": cat,
                "protocol": args.protocol,
                "model_file": str(path),
                "train_run_id": meta.get("mlflow_run_id"),
            },
            tags={"detector": "patchcore", "category": cat, "kind": "eval"},
            enabled=not args.no_mlflow,
        ) as run:
            metrics, _ = evaluate(det, manifest, args.data_root, args.protocol, det.cfg.batch_size)
            if run is not None:
                import mlflow

                mlflow.log_metrics(metrics)
        diffs = {k: abs(metrics[k] - meta["metrics"][k]) for k in COMPARED if k in metrics}
        ok = all(d <= 1e-6 for d in diffs.values())
        status = status or (0 if ok else 1)
        print(
            json.dumps(
                {
                    "category": cat,
                    "reproduced": ok,
                    **{k: round(metrics[k], 4) for k in COMPARED if k in metrics},
                    "max_abs_diff": max(diffs.values()),
                }
            )
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
