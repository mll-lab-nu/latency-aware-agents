#!/usr/bin/env bash
set -euo pipefail

# Fix training-time absolute paths on a fresh eval box (e.g. vast.ai).
#
# The trained checkpoints bake in absolute paths from the TRAINING machine:
#   * task-model checkpoints:  /path/to/mount/latency_data/<model>/checkpoints/...
#       (referenced by the repo eval YAMLs' policy.checkpoint_path)
#   * starVLA base VLM:        /workspace/starVLA/playground/Pretrained_models/...
#       (referenced by each downloaded model's config.yaml framework.qwenvl.base_vlm)
#
# On vast.ai we download models onto the persistent /workspace volume, so those
# absolute paths do not exist. This script reproduces them with symlinks so the
# configs resolve unchanged. It is idempotent -- safe to re-run.
#
# Usage:
#   bash scripts/bash_scripts/download/link_eval_paths.sh
# Overrides:
#   SRC_ROOT (default /workspace)            where models were downloaded
#   DATA_ROOT (default /path/to/mount/latency_data) where eval YAMLs expect task models
#   BASE_VLM_DIR (default /workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct)

SRC_ROOT="${SRC_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/path/to/mount/latency_data}"
BASE_VLM_DIR="${BASE_VLM_DIR:-/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct}"

echo "== link_eval_paths =="
echo "  SRC_ROOT:     ${SRC_ROOT}"
echo "  DATA_ROOT:    ${DATA_ROOT}"
echo "  BASE_VLM_DIR: ${BASE_VLM_DIR}"

mkdir -p "${DATA_ROOT}"

# Link every downloaded task-model dir (one that has a checkpoints/ subdir)
# from SRC_ROOT into DATA_ROOT, matching the eval YAML checkpoint_path roots.
linked=0
shopt -s nullglob
for d in "${SRC_ROOT}"/*/; do
  name="$(basename "${d}")"
  [[ -d "${d}checkpoints" ]] || continue
  target="${d%/}"
  link="${DATA_ROOT}/${name}"
  ln -sfn "${target}" "${link}"
  echo "  linked ${link} -> ${target}"
  linked=$((linked + 1))
done
echo "  task-model links: ${linked}"

# Sanity-check the base VLM. We do not download it here (use
# download/qwen3_vl_4b_instruct.sh); just report whether it is in place.
if [[ -f "${BASE_VLM_DIR}/config.json" ]]; then
  echo "  base VLM: OK (${BASE_VLM_DIR})"
else
  echo "  base VLM: MISSING -- run: bash scripts/bash_scripts/download/qwen3_vl_4b_instruct.sh"
fi

echo "== done =="
