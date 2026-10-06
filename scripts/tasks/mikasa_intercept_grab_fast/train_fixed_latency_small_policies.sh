#!/usr/bin/env bash
# Train/evaluate/upload native InterceptGrabFast PPO teachers at fixed latency L0-L4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="$(command -v "${MIKASA_LCB_PYTHON:-python}")"
OUTPUT_ROOT="${MIKASA_FIXED_LATENCY_OUTPUT_ROOT:-outputs/mikasa_fixed_latency_ppo}"
MODE="${1:-train}"
LATENCIES="${LATENCIES:-0 1 2 3 4}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/tasks/mikasa_intercept_grab_fast/train_fixed_latency_small_policies.sh [train|eval|upload|both|full|all]

Environment overrides:
  LATENCIES                         Space-separated latencies to run. Default: "0 1 2 3 4".
  MIKASA_FIXED_LATENCY_OUTPUT_ROOT  Output root. Default: outputs/mikasa_fixed_latency_ppo.
  MIKASA_LCB_PYTHON                 Python for repo launchers. Default: active python.
  CUDA_VISIBLE_DEVICES              GPU selection passed through to the native MIKASA trainer.
  HF_TOKEN                          Hugging Face token for upload, if not already logged in.

L0 uses latency.method=zero to match the zero-delay Intercept teacher recipe.
L1-L4 use fixed latency at 20 Hz: 50, 100, 150, and 200 ms.
full/all runs train -> eval -> upload for each selected latency.
USAGE
}

case "${MODE}" in
  train|eval|upload|both|full|all) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

run_one() {
  local stage="$1"
  local latency="$2"
  local config output_dir

  case "${stage}" in
    train)
      config="configs/training/latency/mikasa_fixed_latency_l${latency}_scratch.yaml"
      output_dir="${OUTPUT_ROOT}/fixed_l${latency}"
      ;;
    eval)
      config="configs/training/latency/mikasa_fixed_latency_l${latency}_eval.yaml"
      output_dir="${OUTPUT_ROOT}/fixed_l${latency}_eval"
      ;;
    upload)
      echo "[mikasa-fixed-latency] stage=upload latency=L${latency} output_root=${OUTPUT_ROOT}"
      "${PYTHON}" -m scripts.tasks.mikasa_intercept_grab_fast.publish_fixed_latency_small_policy \
        --latency "${latency}" \
        --output-root "${OUTPUT_ROOT}"
      return
      ;;
    *)
      echo "Internal error: unknown stage ${stage}" >&2
      exit 2
      ;;
  esac

  echo "[mikasa-fixed-latency] stage=${stage} latency=L${latency} config=${config} output=${output_dir}"
  "${PYTHON}" -m scripts.tasks.mikasa_intercept_grab_fast.run_profile_policy \
    --config "${config}" \
    --output-dir "${output_dir}"
}

cd "${REPO_ROOT}"
if [[ "${MODE}" != "upload" ]]; then
  bash scripts/mikasa/install.sh
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}"

for latency in ${LATENCIES}; do
  case "${latency}" in
    0|1|2|3|4) ;;
    *)
      echo "Unsupported latency ${latency}; expected one of 0 1 2 3 4." >&2
      exit 2
      ;;
  esac

  if [[ "${MODE}" == "train" || "${MODE}" == "both" ]]; then
    run_one train "${latency}"
  fi
  if [[ "${MODE}" == "eval" || "${MODE}" == "both" ]]; then
    run_one eval "${latency}"
  fi
  if [[ "${MODE}" == "upload" ]]; then
    run_one upload "${latency}"
  fi
  if [[ "${MODE}" == "full" || "${MODE}" == "all" ]]; then
    run_one train "${latency}"
    run_one eval "${latency}"
    run_one upload "${latency}"
  fi
done
