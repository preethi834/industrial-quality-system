"""Fit PatchCore on one or more MVTec AD categories, evaluate, save, log to MLflow.

    uv run python scripts/train_detector.py --categories bottle --protocol official
    uv run python scripts/train_detector.py --categories bottle grid screw --protocol deploy

protocol official: fit on all train/good images (comparable with published numbers).
protocol deploy:   fit on the 'fit' split only; the 'calib' split stays unseen and sets
                   the provisional threshold (the conformal one comes in step 2).
Outputs: models/patchcore/<category>/<protocol>/{model.pt,metrics.json} + an MLflow run.
Run one category per process for the largest categories (RAM is freed on exit).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from iqs_common.assets import load_assets, resolve_model_file
from iqs_vision.data.splits import SplitManifest, verify_on_disk
from iqs_vision.detectors.patchcore import PatchCoreConfig, PatchCoreDetector
from iqs_vision.pipeline import evaluate, fit_detector, seed_everything
from iqs_vision.tracking import mlflow_run

ROOT = Path(__file__).resolve().parents[1]


def train_one(args: argparse.Namespace, category: str, cfg: PatchCoreConfig) -> dict[str, float]:
    manifest = SplitManifest.load(args.manifests / f"{category}.json")
    if args.verify:
        bad = verify_on_disk(manifest, args.data_root)
        if bad:
            raise SystemExit(f"{category}: {len(bad)} files missing or changed, e.g. {bad[0]}")
    weights = (
        resolve_model_file(load_assets(ROOT / "configs/assets.yaml"), cfg.weights, ROOT)
        if cfg.weights
        else None
    )
    seed_everything(cfg.seed)
    det = PatchCoreDetector(cfg, weights)
    params = {
        **cfg.model_dump(mode="json"),
        "category": category,
        "protocol": args.protocol,
        "manifest_fingerprint": manifest.fingerprint(),
        "dataset": manifest.dataset,
    }
    with mlflow_run(
        f"patchcore-{manifest.dataset}",
        f"{category}-{args.protocol}",
        tracking_dir=args.mlruns,
        params=params,
        tags={"detector": "patchcore", "category": category, "protocol": args.protocol},
        enabled=not args.no_mlflow,
    ) as run:
        t0 = time.perf_counter()
        stats = fit_detector(det, manifest, args.data_root, args.protocol, cfg.batch_size)
        t1 = time.perf_counter()
        metrics, _ = evaluate(det, manifest, args.data_root, args.protocol, cfg.batch_size)
        t2 = time.perf_counter()
        metrics.update(
            {
                "fit_images": stats.n_images,
                "memory_bank_size": stats.memory_bank_size,
                "seconds_features": stats.seconds_features,
                "seconds_coreset": stats.seconds_coreset,
                "seconds_fit": round(t1 - t0, 2),
                "seconds_eval": round(t2 - t1, 2),
                "peak_rss_mb": stats.peak_rss_mb,
            }
        )
        run_id = run.info.run_id if run is not None else None
        out = args.models_dir / category / args.protocol
        meta = {
            "category": category,
            "protocol": args.protocol,
            "manifest_fingerprint": manifest.fingerprint(),
            "mlflow_run_id": run_id,
            "metrics": metrics,
        }
        det.save(out / "model.pt", metadata=meta)
        (out / "metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
        if run is not None:
            import mlflow

            mlflow.log_metrics(metrics)
            mlflow.log_dict(meta, "metrics.json")
    print(
        json.dumps(
            {
                "category": category,
                "protocol": args.protocol,
                "run_id": run_id,
                **{k: round(v, 4) for k, v in metrics.items()},
            }
        )
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--protocol", choices=["official", "deploy"], default="official")
    p.add_argument("--config", type=Path, default=ROOT / "configs/vision/patchcore.yaml")
    p.add_argument("--dataset", choices=["mvtec_ad", "visa", "mvtec_ad2"], default="mvtec_ad")
    p.add_argument("--data-root", type=Path, help="default: data/raw/<dataset>")
    p.add_argument("--manifests", type=Path, help="default: eval_sets/vision/<dataset>")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models/patchcore")
    p.add_argument("--mlruns", type=Path, default=ROOT / "mlruns")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the SHA-256 check of every image against the manifest",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. --set coreset_sampling_ratio=0.01",
    )
    args = p.parse_args(argv)
    args.data_root = args.data_root or ROOT / "data/raw" / args.dataset
    args.manifests = args.manifests or ROOT / "eval_sets/vision" / args.dataset
    raw = yaml.safe_load(args.config.read_text())
    for kv in args.set:
        k, v = kv.split("=", 1)
        raw[k] = yaml.safe_load(v)
    cfg = PatchCoreConfig.model_validate(raw)
    for cat in args.categories:
        train_one(args, cat, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
