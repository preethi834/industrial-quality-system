"""Build split manifests (eval_sets/vision/mvtec_ad/<category>.json) from local MVTec AD.

    uv run python scripts/make_splits.py --categories bottle grid screw
    uv run python scripts/make_splits.py --categories all

Deterministic per seed; refuses to overwrite a manifest whose content would change
unless --force (a changed manifest invalidates every run that used it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqs_vision.data.mvtec import CATEGORIES, DatasetNotFoundError, scan_category
from iqs_vision.data.splits import SplitManifest, build_manifest

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/mvtec_ad")
    p.add_argument("--out", type=Path, default=ROOT / "eval_sets/vision/mvtec_ad")
    p.add_argument("--categories", nargs="+", default=["all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cats = list(CATEGORIES) if args.categories == ["all"] else args.categories
    status = 0
    for cat in cats:
        try:
            samples = scan_category(args.data_root, cat)
        except DatasetNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 2
            continue
        m = build_manifest(
            samples,
            root=args.data_root,
            dataset="mvtec_ad",
            category=cat,
            seed=args.seed,
            calib_fraction=args.calib_fraction,
        )
        path = args.out / f"{cat}.json"
        if path.exists() and not args.force:
            old = SplitManifest.load(path)
            if old.fingerprint() != m.fingerprint():
                print(f"error: {path} exists with different content (use --force)", file=sys.stderr)
                status = 2
                continue
        m.save(path)
        counts = {s: len(m.select(s)) for s in ("fit", "calib", "test")}
        print(
            f"{cat}: {counts} sup_test={len(m.select(sup_split='sup_test'))} "
            f"fingerprint={m.fingerprint()[:12]}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
