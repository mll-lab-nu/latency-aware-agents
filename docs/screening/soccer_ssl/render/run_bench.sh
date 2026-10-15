#!/bin/bash
# The benchmark series of the report (one container at a time; host load was ~40 on 32 cores).
# Usage: run_bench.sh [stub|real] [suffix]
set -u
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
OUT=$REPO/docs/screening/soccer_ssl/render/results
IMG=ghcr.io/mll-lab-nu/eai-olympiad:$(cat "$REPO/images/olympiad/VERSION")
SCENE=${1:-stub}; SUF=${2:-}
run() {  # cpus tag args...
  local c=$1 tag=$2; shift 2
  docker run --rm --name "ssl-cam-bench-$tag" --cpus "$c" -e LP_NUM_THREADS="$c" -v "$REPO":/srv:ro -v "$OUT":/out "$IMG" \
    python /srv/docs/screening/soccer_ssl/render/bench.py --scene "$SCENE" --tag "$tag" "$@"
}
run 2 "${SCENE}_cpus2_msaa4$SUF" --extras
run 2 "${SCENE}_cpus2_msaa0$SUF" --offsamples 0
run 1 "${SCENE}_cpus1_msaa0$SUF" --offsamples 0 --quick
run 1 "${SCENE}_cpus1_msaa4$SUF" --quick
run 4 "${SCENE}_cpus4_msaa0$SUF" --offsamples 0 --quick
run 4 "${SCENE}_cpus4_msaa4$SUF" --quick
