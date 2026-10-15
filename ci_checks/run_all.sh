#!/usr/bin/env bash
# Run every per-task check on the given task dirs (default: all of tasks/*), then repo-level checks.
# Needs python3 >= 3.11 (tomllib).
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"; rc=0
tasks=("$@"); [ ${#tasks[@]} -eq 0 ] && tasks=("$root"/tasks/*/)
for t in "${tasks[@]}"; do
  t="${t%/}"; [ -f "$t/task.toml" ] || continue
  for c in check-task-structure check-task-fields check-dockerfile check-instruction check-verifier-isolation \
           check-instance-copies check-standard-agent-image check-seat-isolation; do
    bash "$root/ci_checks/$c.sh" "$t" || rc=1
  done
done
bash "$root/ci_checks/check-dataset-manifest.sh" || rc=1
bash "$root/ci_checks/check-standard-client.sh" || rc=1              # agent + verifier images: the pinned client base and pins
bash "$root/ci_checks/check-model-lock.sh" || rc=1               # every model seat a run script starts has THINK_MODEL_<seat>
python3 "$root/ci_checks/check-robot-limits.py" || rc=1            # robot library: limits with sources, one limit / servo path (bodies/README.md)
python3 "$root/ci_checks/check-skilleval.py" || rc=1               # two modes: [modes.high] / [modes.low], toolkit sha256, hidden requests (skilleval/README.md)
# v0.2 framework (docs/framework_zh.md §9): core purity, sport isolation, one part per PR (needs RO_PR_BASE), no host
# paths or secrets in the v0.2 parts, shared metering copies + the generator's check mode (tools/generate.py check:
# tasks/ is what it writes) + the players list
for c in check-core-purity check-sport-isolation check-pr-scope check-no-host-paths check-v02-shared check-hidden-seeds; do
  bash "$root/ci_checks/$c.sh" || rc=1
done
# Protocol v1.0 task conventions (README, task.toml, tests/test.sh as scripts/task_conventions.py writes them; also a
# step of .github/workflows/static-checks.yml, kept here so a local run catches it)
if [ $# -eq 0 ]; then python3 "$root/scripts/task_conventions.py" --check "$root"/tasks/*/ || rc=1; else python3 "$root/scripts/task_conventions.py" --check "$@" || rc=1; fi
# Protocol v1.0: `legacy:` rules (image move, docs/PROTOCOL.md §7) warn; all other rules fail. RCB_PROTOCOL_STRICT=1 fails on legacy too.
if [ $# -eq 0 ]; then python3 "$root/ci_checks/check-protocol-v1.py" || rc=1; else python3 "$root/ci_checks/check-protocol-v1.py" "$@" || rc=1; fi
[ $rc -eq 0 ] && echo "all checks passed"; exit $rc
