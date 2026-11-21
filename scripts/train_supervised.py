"""Train and evaluate the supervised ResNet-18 baseline per MVTec AD category.

    uv run python scripts/train_supervised.py --categories bottle grid screw

Trains on sup_train, picks the epoch on sup_val, scores sup_test once. PatchCore's AUROC
on the same sup_test images (from its official-protocol metrics.json, if trained) is
logged next to it. Outputs: models/supervised/<cat>/{model.safetensors,metrics.json}
and an MLflow run in experiment "supervised-mvtec_ad".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from iqs_common.assets import load_assets, resolve_model_file
from iqs_vision.data.splits import SplitManifest, validate_entries
from iqs_vision.evaluation.metrics import image_auroc
from iqs_vision.pipeline import seed_everything
from iqs_vision.supervised.classifier import (
    SupervisedConfig,
    build_classifier,
    predict_proba,
    train_classifier,
)
from iqs_vision.tracking import mlflow_run

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/vision/supervised.yaml")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/mvtec_ad")
    p.add_argument("--manifests", type=Path, default=ROOT / "eval_sets/vision/mvtec_ad")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models/supervised")
    p.add_argument("--patchcore-dir", type=Path, default=ROOT / "models/patchcore")
    p.add_argument("--mlruns", type=Path, default=ROOT / "mlruns")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    cfg = SupervisedConfig.model_validate(yaml.safe_load(args.config.read_text()))
    weights = (
        resolve_model_file(load_assets(ROOT / "configs/assets.yaml"), cfg.weights, ROOT)
        if cfg.weights
        else None
    )
    for cat in args.categories:
        manifest = SplitManifest.load(args.manifests / f"{cat}.json")
        train = manifest.select(sup_split="sup_train")
        val = manifest.select(sup_split="sup_val")
        test = manifest.select(sup_split="sup_test")
        validate_entries([*train, *val, *test])  # leakage guard (path and content)
        seed_everything(cfg.seed)
        model = build_classifier(cfg, weights)
        params = {
            **cfg.model_dump(mode="json"),
            "category": cat,
            "manifest_fingerprint": manifest.fingerprint(),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        }
        with mlflow_run(
            "supervised-mvtec_ad",
            f"{cat}-resnet18",
            tracking_dir=args.mlruns,
            params=params,
            tags={"model": "resnet18-supervised", "category": cat},
            enabled=not args.no_mlflow,
        ) as run:
            result = train_classifier(model, train, val, args.data_root, cfg)
            labels, probs = predict_proba(model, test, args.data_root, cfg)
            metrics = {
                "sup_test_image_auroc": image_auroc(labels, probs),
                "val_auroc_best": result.val_auroc,
                "best_epoch": float(result.best_epoch),
                "seconds_train": result.seconds,
            }
            pc = args.patchcore_dir / cat / "official" / "metrics.json"
            if pc.is_file():
                pcm = json.loads(pc.read_text())["metrics"]
                if "sup_test_image_auroc" in pcm:
                    metrics["patchcore_sup_test_image_auroc"] = pcm["sup_test_image_auroc"]
            run_id = run.info.run_id if run is not None else None
            if run is not None:
                import mlflow

                mlflow.log_metrics(metrics)
                for h in result.history:
                    mlflow.log_metrics(
                        {"train_loss": h["train_loss"], "val_auroc": h["val_auroc"]},
                        step=int(h["epoch"]),
                    )
        out = args.models_dir / cat
        out.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        save_file(model.state_dict(), str(out / "model.safetensors"))
        meta = {
            "category": cat,
            "mlflow_run_id": run_id,
            "config": cfg.model_dump(mode="json"),
            "manifest_fingerprint": manifest.fingerprint(),
            "metrics": metrics,
            "history": result.history,
        }
        (out / "metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            json.dumps(
                {"category": cat, "run_id": run_id, **{k: round(v, 4) for k, v in metrics.items()}}
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
