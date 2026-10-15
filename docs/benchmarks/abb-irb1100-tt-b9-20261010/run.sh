#!/bin/sh
# Run a script of the ABB IRB 1100 + gantry B9 study in the eai-olympiad image (MuJoCo 3.3.7, CPU), no network.
#   docs/benchmarks/abb-irb1100-tt-b9-20261010/run.sh <script.py> OUT_PREFIX     NAME=<container> NPROC=<n> (default 1)
#   runs n parts of the script in one container (<script.py> /out/OUT_PREFIX_<i>.json i n, logs /out/OUT_PREFIX_<i>.log)
# Mounts: this checkout at /repo (the robot library with bodies/abb_irb1100), the serve-receivability study's worktree
# (branch tt-receive-study 329ff155: the serve library, the stage-A strokes, lib.py / bodyrig.py / oracle.py, the
# tt-planner sport and bodies/trajopt.py) at /recv, both read-only; results in ./results (/out).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
RECV=${RECV:?set RECV to a checkout of the serve-receivability study (branch tt-receive-study 329ff155)}
NAME=${NAME:-abb-b9-$(basename "$1" .py)-$$}
N=${NPROC:-1}
exec docker run --rm --name "$NAME" --network none --cpus "$N" -u "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  -e TTRECV_OUT=/recv/prototypes/tabletennis_receive/results -e ABB_OUT=/out \
  -e REACT -e LEAD -e READY -e SEL -e N_EVAL -e K -e PART -e NPART -e TAG -e VARIANT -e SEEDS -e NPROC -e ABB_PADDLE -e ABB_DROP -e ABB_MOUNT_Y -e N_OPTS -e REHEARSE -e CAM_DT -e CAM_LAT -e CAM_NOISE -e ACT_LAT -e STANCE -e N_FIRST -e N_RE -e ORACLE -e RALLY_SERVES -e N_RALLIES \
  -v "$ROOT":/repo:ro -v "$RECV":/recv:ro -v "$HERE/results":/out \
  -w /repo/docs/benchmarks/abb-irb1100-tt-b9-20261010 \
  ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 sh -c '
    s=$1; o=$2; n=$3; i=0
    while [ $i -lt $n ]; do python "$s" "/out/${o}_$i.json" $i $n > "/out/${o}_$i.log" 2>&1 & i=$((i+1)); done; wait
    echo finished' sh "$1" "$2" "$N"
