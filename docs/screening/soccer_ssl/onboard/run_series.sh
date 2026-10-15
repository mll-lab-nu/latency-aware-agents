#!/bin/bash
# The onboard experiment series of the report, in lanes (host load 40-60 on 32 cores: at most 2 containers at once).
# Usage: run_series.sh a|b|c|d   (a and b first, in parallel; then c and d in parallel)
set -u
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
IMG=ghcr.io/mll-lab-nu/eai-olympiad:$(cat "$REPO/images/olympiad/VERSION")
R=docs/screening/soccer_ssl/onboard
run() {  # tag args...
  local tag=$1; shift
  docker run --rm --name "ssl-onboard-$tag" --cpus 2 -e LP_NUM_THREADS=2 -v "$REPO":/srv -w /srv "$IMG" \
    python $R/run_onboard.py --tag "$tag" "$@" > "$REPO/$R/results/log_$tag.txt" 2>&1
}
case $1 in
  a) run L2L3_s1 --rec $R/results/rec_L2L3_s1.npz ;;
  b) run L3L2_s2 --rec $R/results/rec_L3L2_s2.npz ;;
  c) run drill_s5 --rec $R/results/rec_drill_s5.npz --save-frames 60 --robots blue3,blue2,blue4,yellow2,yellow4
     run L2L3_s1_noband --rec $R/results/rec_L2L3_s1.npz --no-band --seconds 60
     run L2L3_s1_hfov60 --rec $R/results/rec_L2L3_s1.npz --hfov 60 --seconds 60 ;;
  d) run L2L3_s1_drift3 --rec $R/results/rec_L2L3_s1.npz --drift 3 --seconds 60
     run L2L3_s1_320 --rec $R/results/rec_L2L3_s1.npz --size 320x240 --seconds 60
     run L2L3_s1_nolines --rec $R/results/rec_L2L3_s1.npz --no-lines --seconds 60
     run L2L3_s1_noboundary --rec $R/results/rec_L2L3_s1.npz --no-boundary --seconds 60 ;;
esac
