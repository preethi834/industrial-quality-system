"""Inspection latency through the running vision API (step-1 recorded metric).

    uv run python scripts/benchmark_inspections.py --category bottle --requests 100

Sends real MVTec test images (cycling through the category's test split) one at a time
after --warmup requests, and reports client-side p50/p95/mean latency plus the
server-reported latency. Results are printed as JSON and appended to
reports/inspection_latency.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from iqs_vision.data.splits import SplitManifest

ROOT = Path(__file__).resolve().parents[1]


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("IQS_API__KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("IQS_API__KEYS not found in .env")


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/raw/mvtec_ad")
    p.add_argument("--manifests", type=Path, default=ROOT / "eval_sets/vision/mvtec_ad")
    args = p.parse_args(argv)
    manifest = SplitManifest.load(args.manifests / f"{args.category}.json")
    images = [(args.data_root / e.path).read_bytes() for e in manifest.select("test")]
    headers = {"X-API-Key": _key()}
    client_ms: list[float] = []
    server_ms: list[float] = []
    with httpx.Client(base_url=args.url, timeout=120) as c:
        for i in range(args.warmup + args.requests):
            data = images[i % len(images)]
            t0 = time.perf_counter()
            r = c.post(
                "/v1/inspections",
                headers=headers,
                files={"file": ("x.png", data, "image/png")},
                data={"category": args.category},
            )
            dt = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            if i >= args.warmup:
                client_ms.append(dt)
                server_ms.append(float(r.json()["latency_ms"]))
    result = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "category": args.category,
        "requests": args.requests,
        "warmup": args.warmup,
        "client_p50_ms": round(_pct(client_ms, 0.5), 1),
        "client_p95_ms": round(_pct(client_ms, 0.95), 1),
        "client_mean_ms": round(statistics.fmean(client_ms), 1),
        "server_p50_ms": round(_pct(server_ms, 0.5), 1),
        "server_p95_ms": round(_pct(server_ms, 0.95), 1),
        "note": "sequential requests, docker vision container (2 GB limit), host CPU shared",
    }
    out = ROOT / "reports" / "inspection_latency.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
