#!/usr/bin/env bash
# The model-API error rule has two copies that stay byte-identical (rule I2 of docs/benchmarks/roboolympics-rules.md):
# core/metering/api_errors.py (copied into every task) = scripts/olympiad/api_errors.py (regrade_transient.py). The
# generator, the players list and the capabilities check themselves. (The legacy reference copies of think_clock.py
# and think_proxy.py under scripts/olympiad/tabletennis/ were removed on 2026-10-10: core/metering/think_clock.py and
# players/metering/think_proxy.py are the sources, and tools/generate.py check keeps every task's copy identical.)
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; rc=0
for pair in "core/metering/api_errors.py:$root/scripts/olympiad/api_errors.py"; do
  a="$root/${pair%%:*}"; b="${pair#*:}"
  cmp -s "$a" "$b" || { echo "FAIL [v0.2 shared] ${pair%%:*} differs from ${b#$root/}"; rc=1; }
done
python3 "$root/tools/generate.py" check || rc=1                  # v0.2 tasks are what tools/generate.py writes
python3 "$root/players/players.py" check || rc=1                 # the players list
python3 "$root/tools/capabilities.py" check || rc=1             # every sport's [capabilities]
[ $rc -eq 0 ] && echo "v0.2 shared files: identical; generator and players list ok"; exit $rc
