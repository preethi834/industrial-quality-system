"""Fetch allowlisted model weights (configs/assets.yaml) into models/weights/.

    uv run python scripts/fetch_models.py wide_resnet50_2.tv_in1k --approve

Prints source, size and licence; refuses without --approve; verifies SHA-256.
Already-present files are re-verified, not re-downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqs_common.assets import (
    AssetError,
    download_verified,
    load_assets,
    require_approval,
    verify_file,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("names", nargs="+", help="model names from configs/assets.yaml")
    p.add_argument("--approve", action="store_true", help="confirm the download is approved")
    p.add_argument("--assets", type=Path, default=ROOT / "configs" / "assets.yaml")
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args(argv)
    try:
        assets = load_assets(args.assets)
        for name in args.names:
            a = assets.model(name)
            dest = args.root / a.target
            size = f"{a.size_bytes / 1e6:.1f} MB" if a.size_bytes else "unknown size"
            print(f"{name}: {a.url}\n  {size}, licence {a.licence}\n  -> {dest}")
            if dest.is_file():
                verify_file(dest, a.sha256)
                print("  already present, checksum ok")
                continue
            require_approval(args.approve, f"downloading {name}")
            download_verified(a.url, dest, a.sha256)
            print("  downloaded, checksum ok")
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
