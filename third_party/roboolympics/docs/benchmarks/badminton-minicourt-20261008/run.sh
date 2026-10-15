#!/bin/sh
# Run a minicourt study script inside the eai-olympiad image (MuJoCo 3.3.7, CPU), one CPU, no network.
#   docs/benchmarks/badminton-minicourt-20261008/run.sh <script.py> [args]      (OUT=<dir for json>)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$HERE/results}; mkdir -p "$OUT"
NAME=${NAME:-bm-mini-$(basename "$1" .py)-$$}
exec docker run --rm --name "$NAME" --network none --cpus 1 -u "$(id -u):$(id -g)" \
  -e BM_OUT=/out -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -e REACT -e LEAD_SLOWDOWN -e SEEDS -e SEL -e TAG -e CAMERA -e SHARD -e NSHARD -e MAXTRIES -e COURT -e FAMS \
  -e AIMSET -e HEAD_ORACLE -e MIN_DEPTH -e FIRST -e LEVEL -e GRID -e PLAN_A -e HEAD -e PROBE_T -e Z -e T \
  $(for v in $PASS; do printf -- "-e %s " "$v"; done) \
  -v "$ROOT":/repo:ro -v "$OUT":/out -w /repo/docs/benchmarks/badminton-minicourt-20261008 \
  ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 python "$@"
