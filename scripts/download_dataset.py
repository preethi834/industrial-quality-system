"""Import an allowlisted dataset archive (configs/assets.yaml) into data/raw/.

MVTec requires a form before download, so the archive is fetched by hand and imported:

    uv run python scripts/download_dataset.py mvtec_ad \
        --archive /mnt/c/Users/<you>/Downloads/mvtec_anomaly_detection.tar.xz --approve

Steps: copy to data/raw/archives/ (WSL ext4, not /mnt/c), verify SHA-256, extract with
the tar 'data' safety filter into the target directory. Services never call this.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from iqs_common.assets import (
    AssetError,
    extract_tar,
    load_assets,
    require_approval,
    sha256_file,
    verify_file,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("name", help="dataset name from configs/assets.yaml")
    p.add_argument("--archive", type=Path, required=True, help="locally downloaded archive")
    p.add_argument("--approve", action="store_true", help="confirm the dataset is approved")
    p.add_argument("--assets", type=Path, default=ROOT / "configs" / "assets.yaml")
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args(argv)
    try:
        a = load_assets(args.assets).dataset(args.name)
        print(f"{args.name}: {a.title}\n  source {a.source}\n  licence {a.licence}")
        require_approval(args.approve, f"importing dataset {args.name}")
        if not args.archive.is_file():
            raise AssetError(f"archive not found: {args.archive}")
        local = args.root / "data" / "raw" / "archives" / a.filename
        local.parent.mkdir(parents=True, exist_ok=True)
        if args.archive.resolve() != local.resolve():
            print(f"  copying {args.archive.stat().st_size / 1e9:.2f} GB -> {local}")
            shutil.copyfile(args.archive, local)
        print("  verifying sha256 ...")
        verify_file(local, a.sha256)
        print(f"  sha256 ok ({sha256_file(local)[:16]}...)")
        target = args.root / a.target
        print(f"  extracting -> {target}")
        top = extract_tar(local, target)
        print(f"  extracted {len(top)} top-level entries: {', '.join(top)}")
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
