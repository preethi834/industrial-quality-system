#!/usr/bin/env python
"""Build the NHTSA recall labelling sheet (step 3f). Reads a local flat file; never downloads.

Usage::

    # 1. see which component groups exist and how many campaigns each has
    nice -n 19 uv run python scripts/prepare_nhtsa_labelling.py --list-groups
    # 2. build the sheet for the approved groups
    nice -n 19 uv run python scripts/prepare_nhtsa_labelling.py \\
        --groups "SERVICE BRAKES" "STEERING" "SUSPENSION" "POWER TRAIN" "STRUCTURE"

Input: ``data/raw/nhtsa/FLAT_RCL_POST_2010.txt`` (unzipped from FLAT_RCL_POST_2010.zip,
fetched by hand after approval). Output: ``eval_sets/diagnosis/nhtsa/recalls_v1.csv``
(one row per campaign, label columns empty) and ``recalls_v1.manifest.json``. An existing
sheet is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from iqs_diagnosis.evaluation.nhtsa_labels import (
    SampleConfig,
    matches_terms,
    sample_recalls,
    write_sheet,
)
from iqs_diagnosis.ingestion.nhtsa import read_recalls

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--recalls", type=Path, default=REPO_ROOT / "data/raw/nhtsa/FLAT_RCL_POST_2010.txt"
    )
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--groups", nargs="+", default=[])
    ap.add_argument("--since", default="20150101")
    ap.add_argument("--per-group-term", type=int, default=60)
    ap.add_argument("--per-group-control", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "eval_sets/diagnosis/nhtsa/recalls_v1.csv"
    )
    args = ap.parse_args()

    if not args.recalls.is_file():
        raise SystemExit(
            f"error: {args.recalls} not found (download needs approval; see LABELLING_GUIDE.md)"
        )
    recalls, stats = read_recalls(args.recalls)
    print(f"{len(recalls)} campaigns from {stats.rows} rows; {stats.model_dump()}")

    if args.list_groups:
        recent = [r for r in recalls if r.recall_type == "V" and r.rcdate >= args.since]
        n = Counter(g for r in recent for g in r.component_groups)
        t = Counter(
            g for r in recent if matches_terms(r.defect_summary) for g in r.component_groups
        )
        print(f"vehicle campaigns since {args.since}: {len(recent)}")
        print(f"{'component group':50s} {'campaigns':>9s} {'term match':>10s}")
        for g, k in n.most_common():
            print(f"{g:50s} {k:9d} {t[g]:10d}")
        return 0

    if not args.groups:
        raise SystemExit("error: pass --groups (or --list-groups first)")
    cfg = SampleConfig(
        groups=tuple(args.groups),
        since=args.since,
        per_group_term=args.per_group_term,
        per_group_control=args.per_group_control,
        seed=args.seed,
    )
    sample = sample_recalls(recalls, cfg)
    source = {
        "file": args.recalls.name,
        "url": SOURCE_URL,
        "bytes": args.recalls.stat().st_size,
        "sha256": _sha256(args.recalls),
    }
    manifest = args.out.with_suffix(".manifest.json")
    write_sheet(
        sample, args.out, manifest, cfg=cfg, source=source, read_stats=stats, n_recalls=len(recalls)
    )
    per = Counter((g, s) for _, g, s in sample)
    for (g, s), k in sorted(per.items()):
        print(f"  {g:40s} {s:8s} {k:4d}")
    print(f"wrote {len(sample)} rows to {args.out} and {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
