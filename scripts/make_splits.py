"""Build split manifests (eval_sets/vision/<dataset>/<category>.json) from local data.

    uv run python scripts/make_splits.py --categories bottle grid screw
    uv run python scripts/make_splits.py --categories all
    uv run python scripts/make_splits.py --dataset visa --categories cashew pipe_fryum
    uv run python scripts/make_splits.py --dataset mvtec_ad2 --categories vial fruit_jelly

Datasets: mvtec_ad, visa (official 1-class split), mvtec_ad2 (train, validation -> calib,
test_public "regular" -> test, other lighting conditions -> shift).

Deterministic per seed; refuses to overwrite a manifest whose content would change
unless --force (a changed manifest invalidates every run that used it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqs_vision.data import mvtec, mvtec_ad2, visa
from iqs_vision.data.mvtec import DatasetNotFoundError
from iqs_vision.data.splits import SplitManifest, build_manifest

ADAPTERS = {"mvtec_ad": mvtec, "visa": visa, "mvtec_ad2": mvtec_ad2}

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=sorted(ADAPTERS), default="mvtec_ad")
    p.add_argument("--data-root", type=Path, help="default: data/raw/<dataset>")
    p.add_argument("--out", type=Path, help="default: eval_sets/vision/<dataset>")
    p.add_argument("--categories", nargs="+", default=["all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    adapter = ADAPTERS[args.dataset]
    data_root = args.data_root or ROOT / "data/raw" / args.dataset
    out = args.out or ROOT / "eval_sets/vision" / args.dataset
    cats = list(adapter.CATEGORIES) if args.categories == ["all"] else args.categories
    status = 0
    for cat in cats:
        try:
            samples = adapter.scan_category(data_root, cat)
        except DatasetNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 2
            continue
        m = build_manifest(
            samples,
            root=data_root,
            dataset=args.dataset,
            category=cat,
            seed=args.seed,
            calib_fraction=args.calib_fraction,
        )
        path = out / f"{cat}.json"
        if path.exists() and not args.force:
            old = SplitManifest.load(path)
            if old.fingerprint() != m.fingerprint():
                print(f"error: {path} exists with different content (use --force)", file=sys.stderr)
                status = 2
                continue
        m.save(path)
        counts = {s: len(m.select(s)) for s in ("fit", "calib", "test", "shift")}
        counts["calib_validation"] = sum(e.source == "validation" for e in m.select("calib"))
        print(
            f"{cat}: {counts} sup_test={len(m.select(sup_split='sup_test'))} "
            f"fingerprint={m.fingerprint()[:12]}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
