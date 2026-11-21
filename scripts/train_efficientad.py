"""Train EfficientAD on one MVTec AD category (GPU; run from finetune/efficientad on
Colab/Kaggle) and write a run bundle for scripts/import_run_bundle.py.

    python scripts/train_efficientad.py --category bottle --approve-downloads

Downloads (need approval, Rev 2 §3.9; skipped when already present): ImageNette
(imagenette2.tgz, used by the penalty loss) and anomalib's pre-trained teacher
(efficientad_pretrained_weights.zip). MVTec AD itself must already be extracted under
--data-root; it is checked against the committed manifest (same images, same splits)
and never downloaded. Protocol: official (full train set), evaluated with the same
pipeline as PatchCore.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

from iqs_vision.bundles import create_bundle
from iqs_vision.data.mvtec import scan_category
from iqs_vision.data.splits import SplitManifest, verify_on_disk
from iqs_vision.detectors.efficientad import EfficientADConfig, EfficientADDetector
from iqs_vision.pipeline import evaluate, seed_everything

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--category", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/vision/efficientad.yaml")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/mvtec_ad")
    p.add_argument("--manifests", type=Path, default=ROOT / "eval_sets/vision/mvtec_ad")
    p.add_argument("--imagenette-dir", type=Path, default=ROOT / "data/raw/imagenette")
    p.add_argument("--work-dir", type=Path, default=ROOT / "models/efficientad_work")
    p.add_argument("--out-dir", type=Path, default=ROOT / "release/bundles")
    p.add_argument("--max-steps", type=int, default=None, help="override config (smoke runs)")
    p.add_argument("--approve-downloads", action="store_true")
    args = p.parse_args(argv)

    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    raw = yaml.safe_load(args.config.read_text())
    if args.max_steps is not None:
        raw["max_steps"] = args.max_steps
    cfg = EfficientADConfig.model_validate(raw)
    scan_category(args.data_root, args.category)  # raises instead of letting anomalib download
    manifest = SplitManifest.load(args.manifests / f"{args.category}.json")
    bad = verify_on_disk(manifest, args.data_root)
    if bad:
        print(f"error: {len(bad)} images differ from the manifest, e.g. {bad[0]}", file=sys.stderr)
        return 2
    needs = not (args.imagenette_dir / "imagenette2").exists()
    if needs and not args.approve_downloads:
        print(
            "error: ImageNette and the EfficientAD teacher weights will be downloaded; "
            "re-run with --approve-downloads once approved",
            file=sys.stderr,
        )
        return 2

    seed_everything(cfg.seed)
    dm = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=cfg.batch_size,
        eval_batch_size=8,
        num_workers=2,
        seed=cfg.seed,
    )
    model = EfficientAd(
        imagenet_dir=args.imagenette_dir,
        model_size=cfg.model_size,
        teacher_out_channels=cfg.teacher_out_channels,
        padding=cfg.padding,
        pad_maps=cfg.pad_maps,
    )
    engine = Engine(
        max_steps=cfg.max_steps, default_root_dir=args.work_dir / args.category, logger=False
    )
    engine.fit(model=model, datamodule=dm)

    det = EfficientADDetector(cfg, model.model.cpu())
    metrics, _ = evaluate(det, manifest, args.data_root, "official", 8)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    model_file = args.work_dir / args.category / "model.pt"
    det.save(model_file, metadata={"category": args.category, "metrics": metrics})
    bundle = create_bundle(
        args.out_dir / f"efficientad_{args.category}.zip",
        model_file,
        detector="efficientad",
        category=args.category,
        protocol="official",
        git_commit=commit,
        manifest_fingerprint=manifest.fingerprint(),
        config=cfg.model_dump(mode="json"),
        metrics=metrics,
        environment=env,
    )
    print(json.dumps({"bundle": str(bundle), **metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
