#!/usr/bin/env bash
# Smoke test against a running compose stack (make up PROFILES=apps).
# Reads IQS_API__KEYS from the repo-root .env. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

key=$(grep -E '^IQS_API__KEYS=' .env | cut -d= -f2- | cut -d, -f1)
[[ -n "$key" ]] || { echo "IQS_API__KEYS is empty in .env" >&2; exit 1; }

expect() {  # expect <status> <url> [curl args...]
  local want=$1 url=$2; shift 2
  local got
  got=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@" "$url")
  if [[ "$got" != "$want" ]]; then
    echo "FAIL $url: expected $want, got $got" >&2
    exit 1
  fi
  echo "ok   $want $url"
}

for port in 8001 8002; do
  base="http://127.0.0.1:${port}"
  expect 200 "$base/health"
  expect 200 "$base/ready"
  expect 401 "$base/v1/capabilities"
  expect 401 "$base/v1/capabilities" -H "X-API-Key: wrong"
  expect 200 "$base/v1/capabilities" -H "X-API-Key: ${key}"
done
echo "smoke test passed"
