#!/usr/bin/env bash
# Train PatchCore category by category, one process each (RAM is returned between runs).
# Usage: scripts/run_patchcore_queue.sh "<protocol>:<cat> <protocol>:<cat> ..."
# Logs: reports/logs/patchcore_<cat>_<protocol>.log (incl. /usr/bin/time peak RSS).
# Run with the compose stack stopped (make down): the WSL VM has ~10 GB.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/logs
for job in $1; do
  protocol=${job%%:*}; cat=${job#*:}
  log=reports/logs/patchcore_${cat}_${protocol}.log
  if [[ -f models/patchcore/$cat/$protocol/metrics.json ]]; then
    echo "skip $cat $protocol (done)"; continue
  fi
  echo "$(date -Is) start $cat $protocol"
  /usr/bin/time -v uv run --frozen python scripts/train_detector.py \
      --categories "$cat" --protocol "$protocol" >"$log" 2>&1
  rc=$?
  echo "$(date -Is) end   $cat $protocol rc=$rc $(grep -h '^{' "$log" | tail -1 | cut -c1-160)"
done
