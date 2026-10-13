#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPECS=(
  configs/experiments/jitter_robustness/flappy.yaml
  configs/experiments/jitter_robustness/demon_attack_capped.yaml
  configs/experiments/jitter_robustness/demon_attack_extended.yaml
  configs/experiments/jitter_robustness/deadly_corridor.yaml
)
TRAIN_MANIFESTS=(
  configs/train/jitter_robustness/flappy/manifest.tsv
  configs/train/jitter_robustness/demon_attack/capped/manifest.tsv
  configs/train/jitter_robustness/deadly_corridor/manifest.tsv
)
EVAL_MANIFESTS=(
  configs/train/jitter_robustness/flappy/manifest.tsv
  configs/train/jitter_robustness/demon_attack/capped/manifest.tsv
  configs/train/jitter_robustness/demon_attack/extended/manifest.tsv
  configs/train/jitter_robustness/deadly_corridor/manifest.tsv
)
RESULTS_ROOT="outputs/jitter_robustness"
ANALYSIS_DIR="scripts/experiments/jitter_robustness"
PAPER_OUTPUT_DIR="${RESULTS_ROOT}/paper"
PAPER_FIGURE_DIR="${PAPER_OUTPUT_DIR}/figures"

usage() {
  cat <<'EOF'
Usage:
  run.sh all --gpus CSV
  run.sh generate [--seeds CSV]
  run.sh train --gpus CSV [--match TEXT] [--dry-run]
  run.sh eval --gpus CSV [--match TEXT] [--dry-run]
  run.sh build-data [--results-root PATH] [--output PATH]
  run.sh plot [--data PATH] [--output-dir PATH]
EOF
}

generate_configs() {
  local seeds="$1"
  local args=()
  if [[ -n "${seeds}" ]]; then
    args+=(--seeds "${seeds}")
  fi
  for spec in "${SPECS[@]}"; do
    python scripts/sweeps/generate_sweep_configs.py --spec "${spec}" "${args[@]}"
  done
}

run_compute_stage() {
  local stage="$1"
  local gpus="$2"
  local match="$3"
  local dry_run="$4"
  shift 4
  local args=(--stage "${stage}" --gpus "${gpus}")
  if [[ -n "${match}" ]]; then
    args+=(--match "${match}")
  fi
  if [[ "${dry_run}" == "true" ]]; then
    args+=(--dry-run)
  fi
  for manifest in "$@"; do
    python scripts/experiment_runner/run_manifest_jobs.py --manifest "${manifest}" "${args[@]}"
  done
}


build_local_data() {
  local results_root="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  python "${ANALYSIS_DIR}/build_data.py" \
    --results-root "${results_root}" \
    --output "${output}"
}

plot_data() {
  local data="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  python "${ANALYSIS_DIR}/plot.py" --data "${data}" --output-dir "${output_dir}"
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
shift

case "${command}" in
  generate)
    seeds=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --seeds) seeds="$2"; shift 2 ;;
        *) usage; exit 2 ;;
      esac
    done
    generate_configs "${seeds}"
    ;;
  train|eval)
    gpus=""
    match=""
    dry_run=false
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --gpus) gpus="$2"; shift 2 ;;
        --match) match="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        *) usage; exit 2 ;;
      esac
    done
    if [[ -z "${gpus}" ]]; then
      usage
      exit 2
    fi
    if [[ "${command}" == "train" ]]; then
      run_compute_stage train "${gpus}" "${match}" "${dry_run}" "${TRAIN_MANIFESTS[@]}"
    else
      run_compute_stage eval "${gpus}" "${match}" "${dry_run}" "${EVAL_MANIFESTS[@]}"
    fi
    ;;
  build-data)
    results_root="${RESULTS_ROOT}"
    output="${PAPER_OUTPUT_DIR}/data.csv"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --results-root) results_root="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        *) usage; exit 2 ;;
      esac
    done
    build_local_data "${results_root}" "${output}"
    ;;
  plot)
    data="${PAPER_OUTPUT_DIR}/data.csv"
    output_dir="${PAPER_FIGURE_DIR}"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --data) data="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        *) usage; exit 2 ;;
      esac
    done
    plot_data "${data}" "${output_dir}"
    ;;
  all)
    gpus=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --gpus) gpus="$2"; shift 2 ;;
        *) usage; exit 2 ;;
      esac
    done
    if [[ -z "${gpus}" ]]; then
      usage
      exit 2
    fi
    generate_configs ""
    run_compute_stage train "${gpus}" "" false "${TRAIN_MANIFESTS[@]}"
    run_compute_stage eval "${gpus}" "" false "${EVAL_MANIFESTS[@]}"
    build_local_data "${RESULTS_ROOT}" "${PAPER_OUTPUT_DIR}/data.csv"
    plot_data "${PAPER_OUTPUT_DIR}/data.csv" "${PAPER_FIGURE_DIR}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
