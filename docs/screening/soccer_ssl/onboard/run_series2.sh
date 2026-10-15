#!/bin/bash
# The onboard experiment series of the report, in lanes (host load 40-60 on 32 cores: at most 2 containers at once).
# Usage: run_series2.sh e|f   (the robustness runs, after run_series.sh c and d)
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
  e) run L2L3_s1_jitter05 --rec $R/results/rec_L2L3_s1.npz --pitch-jitter 0.5 --seconds 60
     run L2L3_s1_noise8 --rec $R/results/rec_L2L3_s1.npz --noise 8 --seconds 60
     run L2L3_s1_global --rec $R/results/rec_L2L3_s1.npz --global-init --particles 3000 --seconds 20 \
         --robots blue0,blue1,blue2,blue3,blue4
     run L2L3_s1_jitter1_robust --rec $R/results/rec_L2L3_s1.npz --pitch-jitter 1.0 --pitch-robust --seconds 60
     docker run --rm --name ssl-onboard-bench-final --cpus 2 -e LP_NUM_THREADS=2 -v "$REPO":/srv:ro \
       -v "$REPO/$R/results":/out -w /srv "$IMG" python /srv/$R/bench_onboard.py --tag final_cpus2 --quick
     docker run --rm --name ssl-onboard-figs --cpus 1 -v "$REPO":/srv -w /srv "$IMG" python $R/make_figs.py ;;
  f) run L2L3_s1_kidnap --rec $R/results/rec_L2L3_s1.npz --kidnap 20 --seconds 60
     run L2L3_s1_jitter1 --rec $R/results/rec_L2L3_s1.npz --pitch-jitter 1.0 --seconds 60
     run L2L3_s1_det2 --rec $R/results/rec_L2L3_s1.npz --seconds 60 ;;
esac
