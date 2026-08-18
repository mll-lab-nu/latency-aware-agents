#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage: train_small_model.sh [flappy|demon_attack|deadly_corridor|defend_the_line|all] --gpus CSV
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor|defend_the_line) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor defend_the_line) ;;
    *) usage; exit 2 ;;
  esac
}

env_selector=all
case "${1:-}" in
  flappy|demon_attack|deadly_corridor|defend_the_line|all)
    env_selector="$1"
    shift
    ;;
esac

gpus=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      gpus="$2"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${gpus}" ]]; then
  usage
  exit 2
fi

select_envs "${env_selector}"
manifests=()
for env_name in "${ENVS[@]}"; do
  spec="configs/experiments/memory/rollout_teacher/${env_name}.yaml"
  manifest="configs/train/memory/rollout_teacher/${env_name}/manifest.tsv"

  conda run --no-capture-output -n latency python \
    scripts/sweeps/generate_sweep_configs.py \
    --spec "${spec}"

  manifests+=("${manifest}")
done

combined_manifest="configs/train/memory/rollout_teacher/manifest.tsv"
head -n 1 "${manifests[0]}" > "${combined_manifest}.tmp"
for manifest in "${manifests[@]}"; do
  tail -n +2 "${manifest}" >> "${combined_manifest}.tmp"
done
mv "${combined_manifest}.tmp" "${combined_manifest}"

conda run --no-capture-output -n latency python -m \
  scripts.experiment_runner.run_manifest_jobs \
  --manifest "${combined_manifest}" \
  --stage all \
  --gpus "${gpus}"
