#!/bin/sh
# run.sh NAME REPO SCRIPT [env...]: one-CPU container, repo read-only at /repo, scratch at /work, out at /work/out
NAME=$1; REPO=$2; SCRIPT=$3; shift 3
W=${W:-$(cd "$(dirname "$0")" && pwd)}
exec docker run --rm --name "$NAME" --network none --cpus 1 -u "$(id -u):$(id -g)" \
  -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e OUT=/work/out -e BM_OUT=/work/out \
  $(for v in "$@"; do printf -- "-e %s " "$v"; done) \
  -v "$REPO":/repo:ro -v "$W":/work -w /work ghcr.io/mll-lab-nu/eai-olympiad:0.1.0 python "/work/$SCRIPT"
