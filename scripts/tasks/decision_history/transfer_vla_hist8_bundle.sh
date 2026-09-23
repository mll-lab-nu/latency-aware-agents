#!/usr/bin/env bash
set -euo pipefail

TASK="$1"
MODEL="$2"
RUN_ID="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8/data/${TASK}/${MODEL}"
RUN_DIR="${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8_v2/train/${RUN_ID}"
BUNDLE="${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8_v2/bundles/${RUN_ID}"
REMOTE_BUNDLE="/mnt/ssd/codex/hist8-vla-v2-20260923/outputs/tasks/decision_history_vla_hist8_v2/bundles/${RUN_ID}"
KEY="${HIST8_TRANSFER_KEY}"
VLA_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA:${REPO_ROOT}/third_party/sample-factory"
WANDB_ID="$("${VLA_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wandb_run_id"])' "${RUN_DIR}/run_metadata.json")"
"${VLA_PY}" -m scripts.tasks.decision_history.eval_vla_hist8 bundle \
  --run-dir "${RUN_DIR}" --dataset-dir "${DATA_ROOT}/lerobot" \
  --collection-provenance "${DATA_ROOT}/provenance.json" \
  --output-dir "${BUNDLE}" --step 5000 --wandb-run-id "${WANDB_ID}"

source_sha="$(sha256sum "${BUNDLE}/checkpoints/model.pt" | cut -d ' ' -f 1)"
printf '%s  checkpoints/model.pt\n' "${source_sha}" > "${BUNDLE}/checkpoint.sha256"
ssh -i "${KEY}" -o BatchMode=yes ubuntu@10.164.0.2 "mkdir -p '${REMOTE_BUNDLE}'"
rsync -a --partial --append-verify \
  -e "ssh -i ${KEY} -o BatchMode=yes" \
  "${BUNDLE}/" "ubuntu@10.164.0.2:${REMOTE_BUNDLE}/"
destination_sha="$(ssh -i "${KEY}" -o BatchMode=yes ubuntu@10.164.0.2 \
  "sha256sum '${REMOTE_BUNDLE}/checkpoints/model.pt'" | cut -d ' ' -f 1)"
test "${source_sha}" = "${destination_sha}"
printf '%s %s %s\n' "${TASK}" "${MODEL}" "${source_sha}"
