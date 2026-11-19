"""Import a run bundle trained on a cloud GPU (finetune/efficientad) and verify it on CPU.

    uv run python scripts/import_run_bundle.py ~/Downloads/efficientad_bottle.zip

Checks the bundle (members, checksums, manifest fingerprint, git commit), installs the
model to models/<detector>/<category>/<protocol>/, re-evaluates it locally on CPU with
the same pipeline as PatchCore, and logs one MLflow run (tag source=cloud) with both
the cloud metrics and the local ones. Exit code 1 if the local image AUROC differs from
the cloud one by more than --tolerance (default 0.005).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from iqs_vision.bundles import BundleError, check_against_repo, read_bundle
from iqs_vision.data.splits import SplitManifest
from iqs_vision.detectors.efficientad import EfficientADDetector
from iqs_vision.pipeline import evaluate
from iqs_vision.tracking import mlflow_run

ROOT = Path(__file__).resolve().parents[1]


def commit_known(commit: str) -> bool:
    r = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, capture_output=True
    )
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("bundle", type=Path)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/mvtec_ad")
    p.add_argument("--manifests", type=Path, default=ROOT / "eval_sets/vision/mvtec_ad")
    p.add_argument("--models-root", type=Path, default=ROOT / "models")
    p.add_argument("--mlruns", type=Path, default=ROOT / "mlruns")
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--no-eval", action="store_true", help="install without CPU re-evaluation")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    try:
        info, model_bytes = read_bundle(args.bundle)
        if info.detector != "efficientad":
            raise BundleError(f"unsupported detector {info.detector!r}")
        manifest = SplitManifest.load(args.manifests / f"{info.category}.json")
        check_against_repo(info, manifest.fingerprint(), commit_known(info.git_commit))
    except (BundleError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = args.models_root / info.detector / info.category / info.protocol
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.pt").write_bytes(model_bytes)
    det, _ = EfficientADDetector.load(out / "model.pt")
    print(f"installed {info.detector}/{info.category}/{info.protocol} -> {out}")
    result: dict[str, object] = {"category": info.category, "cloud": info.metrics}
    status = 0
    with mlflow_run(
        "efficientad-mvtec_ad",
        f"{info.category}-{info.protocol}-import",
        tracking_dir=args.mlruns,
        params={
            **info.config,
            "category": info.category,
            "protocol": info.protocol,
            "manifest_fingerprint": info.manifest_fingerprint,
            "bundle_git_commit": info.git_commit,
        },
        tags={
            "detector": info.detector,
            "category": info.category,
            "source": "cloud",
            **{f"cloud_{k}": v for k, v in info.environment.items()},
        },
        enabled=not args.no_mlflow,
    ) as run:
        logged = {f"cloud_{k}": v for k, v in info.metrics.items()}
        if not args.no_eval:
            local, _ = evaluate(det, manifest, args.data_root, info.protocol, 8)  # type: ignore[arg-type]
            logged.update({f"local_{k}": v for k, v in local.items()})
            result["local"] = local
            if "image_auroc" in info.metrics:
                diff = abs(local["image_auroc"] - info.metrics["image_auroc"])
                logged["abs_diff_image_auroc"] = diff
                result["reproduced"] = diff <= args.tolerance
                status = 0 if diff <= args.tolerance else 1
        if run is not None:
            import mlflow

            mlflow.log_metrics(logged)
            mlflow.log_dict(info.raw, "bundle.json")
            result["mlflow_run_id"] = run.info.run_id
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, default=float))
    return status


if __name__ == "__main__":
    sys.exit(main())
