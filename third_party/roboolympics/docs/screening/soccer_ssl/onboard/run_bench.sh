#!/bin/bash
# Render-cost series of the onboard cameras (one container at a time; host load 40-60 on 32 cores).
set -u
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
OUT=$REPO/docs/screening/soccer_ssl/onboard/results
IMG=ghcr.io/mll-lab-nu/eai-olympiad:$(cat "$REPO/images/olympiad/VERSION")
run() {  # cpus tag args...
  local c=$1 tag=$2; shift 2
  docker run --rm --name "ssl-onboard-bench-$tag" --cpus "$c" -e LP_NUM_THREADS="$c" -v "$REPO":/srv:ro -v "$OUT":/out \
    -w /srv "$IMG" python /srv/docs/screening/soccer_ssl/onboard/bench_onboard.py --tag "$tag" "$@"
}
run 2 cpus2 --extras
run 1 cpus1 --quick
run 4 cpus4 --quick
