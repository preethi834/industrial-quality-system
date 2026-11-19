#!/usr/bin/env bash
# Smoke test against a running compose stack.
#   make up PROFILES=apps   && scripts/smoke_test.sh            (vision + diagnosis)
#   make up PROFILES=vision && SMOKE_PORTS=8001 scripts/smoke_test.sh
# Reads IQS_API__KEYS from the repo-root .env. Exits non-zero on the first failure.
# Vision: if models are installed, a real MVTec image (if present locally) must come back
# with a score, decision and heatmap; otherwise POST /v1/inspections must answer 501.
# The dashboard (:8501) is checked when it is running.
set -euo pipefail
cd "$(dirname "$0")/.."

key=$(grep -E '^IQS_API__KEYS=' .env | cut -d= -f2- | cut -d, -f1)
[[ -n "$key" ]] || { echo "IQS_API__KEYS is empty in .env" >&2; exit 1; }
ports=${SMOKE_PORTS:-8001 8002}

expect() {  # expect <status> <url> [curl args...]
  local want=$1 url=$2; shift 2
  local got
  got=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$@" "$url")
  if [[ "$got" != "$want" ]]; then
    echo "FAIL $url: expected $want, got $got" >&2
    exit 1
  fi
  echo "ok   $want $url"  # never echo the curl arguments: they contain the API key
}

for port in $ports; do
  base="http://127.0.0.1:${port}"
  expect 200 "$base/health"
  expect 200 "$base/ready"
  expect 401 "$base/v1/capabilities"
  expect 401 "$base/v1/capabilities" -H "X-API-Key: wrong"
  expect 200 "$base/v1/capabilities" -H "X-API-Key: ${key}"
done

if [[ " $ports " == *" 8001 "* ]]; then
  base=http://127.0.0.1:8001
  models=$(curl -s --max-time 10 -H "X-API-Key: ${key}" "$base/v1/models")
  category=$(printf '%s' "$models" | python3 -c 'import json,sys; m=json.load(sys.stdin); print(m[0]["category"] if m else "")')
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  echo "not an image" > "$tmp/x.txt"
  if [[ -z "$category" ]]; then
    printf '\x89PNG' > "$tmp/x.png"
    expect 501 "$base/v1/inspections" -H "X-API-Key: ${key}" -F "file=@$tmp/x.png;type=image/png" -F category=bottle
  else
    img=$(ls data/raw/mvtec_ad/"$category"/test/*/000.png 2>/dev/null | tail -1 || true)
    expect 401 "$base/v1/inspections" -F "file=@$tmp/x.txt;type=image/png" -F "category=$category"
    expect 415 "$base/v1/inspections" -H "X-API-Key: ${key}" -F "file=@$tmp/x.txt;type=text/plain" -F "category=$category"
    if [[ -n "$img" ]]; then
      out=$(curl -s --max-time 120 -H "X-API-Key: ${key}" -F "file=@$img;type=image/png" -F "category=$category" "$base/v1/inspections")
      printf '%s' "$out" | python3 -c '
import base64, json, sys
r = json.load(sys.stdin)
assert r["decision"] in ("anomalous", "normal", "undecided"), r
assert base64.b64decode(r["heatmap_png_base64"]).startswith(b"\x89PNG")
assert "description" not in r
thr = "none" if r["threshold"] is None else format(r["threshold"], ".3f")
print("ok   inspection", r["category"], "score=" + format(r["score"], ".3f"),
      "thr=" + thr, r["decision"], format(r["latency_ms"], ".0f"), "ms")'
    else
      echo "skip real-image inspection (no local MVTec image for $category)"
    fi
  fi
fi

if curl -s -o /dev/null --max-time 2 http://127.0.0.1:8501/_stcore/health; then
  expect 200 http://127.0.0.1:8501/_stcore/health
fi
echo "smoke test passed"
