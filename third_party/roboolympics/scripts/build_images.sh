#!/usr/bin/env bash
# Build the RoboOlympics simulator images from images/<name>/ (build context: the image directory):
#   images/olympiad/            -> eai-olympiad            (every sport's `sim` service except billiards)
#   images/olympiad-billiards/  -> eai-olympiad-billiards  (billiards)
#   images/olympiad-soccer/     -> eai-olympiad-soccer     (soccer)
# Tag: $RCB_IMAGE_REGISTRY/eai-<name>:<images/<name>/VERSION> (default registry: the one the tasks pin).
# Usage: scripts/build_images.sh [name ...]      (default: every images/*/)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
REG="${RCB_IMAGE_REGISTRY:-ghcr.io/mll-lab-nu}"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=($(cd images && ls -d */ | tr -d /))
for name in "${names[@]}"; do
  dir=images/$name; VER="$(cat "$dir/VERSION")"
  echo "+ docker build -t $REG/eai-$name:$VER -f $dir/Dockerfile $dir"
  docker build -t "$REG/eai-$name:$VER" -f "$dir/Dockerfile" "$dir"
done
