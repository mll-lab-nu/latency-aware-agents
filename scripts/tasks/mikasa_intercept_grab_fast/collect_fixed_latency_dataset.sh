#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/tasks/mikasa_intercept_grab_fast/collect_fixed_latency_dataset.sh \
    <latency-frames> <physical-gpu-id> <ppo-checkpoint.pt> <output-root> <hf-dataset-repo>

Example: collect at latency 2 (100 ms) on physical GPU 3:
  bash scripts/tasks/mikasa_intercept_grab_fast/collect_fixed_latency_dataset.sh \
    2 3 outputs/mikasa_fixed_latency_ppo/publish/fixed_l2/checkpoint.pt \
    outputs/mikasa_successful_h1 your-org/mikasa-teacher-rollouts

Collects 250 successful training episodes plus 25 successful validation episodes.
Uses native MIKASA PPO checkpoints, success_once, and H1 issued-action labels.
No return threshold, minimum episode length, or additional step limit is applied.
The task's own 60-transition horizon still applies. Each attempt ends at success/done.
The collector uses one environment so successful episodes are selected individually.
Writes raw NPZ and StarVLA-compatible LeRobot train/validation data, then uploads
the LeRobot datasets and provenance under a latency-specific HF directory.

Optional environment variables:
  TRAIN_EPISODES       Default: 250.
  VAL_EPISODES         Default: 25.
  TASK_PY             Default: third_party/MIKASA-Robo/.venv/bin/python.
  RLDS_PY             Default: MIKASA RLDS builder's .venv/bin/python.
  LEROBOT_PY          Default: MIKASA LeRobot converter's .venv/bin/python.
  HF_PRIVATE          1 creates a private repo; default: 0.
  HF_PATH_PREFIX      Override the latency-specific upload directory.
  HF_TOKEN            HF authentication; an existing Hub login also works.

Run from an installed latency-sensitive-bench environment with the MIKASA submodule
and conversion environments installed. Use a fresh output root for new collection.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 5 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LATENCY="$1"
GPU_ID="$2"
CHECKPOINT="$3"
OUTPUT_ROOT="$4"
HF_REPO_ID="$5"
TASK_PY="${TASK_PY:-${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python}"
MIKASA_ROOT="${REPO_ROOT}/third_party/MIKASA-Robo"
RLDS_PY="${RLDS_PY:-${MIKASA_ROOT}/utils/convert_npz_to_rlds/rlds_dataset_builder/.venv/bin/python}"
LEROBOT_PY="${LEROBOT_PY:-${MIKASA_ROOT}/utils/convert_rlds_to_lerobot/.venv/bin/python}"
TRAIN_EPISODES="${TRAIN_EPISODES:-250}"
VAL_EPISODES="${VAL_EPISODES:-25}"
TOTAL_EPISODES="$((TRAIN_EPISODES + VAL_EPISODES))"

CHECKPOINT="$("${TASK_PY}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${CHECKPOINT}")"
OUTPUT_ROOT="$("${TASK_PY}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${OUTPUT_ROOT}")"
OUTPUT_DIR="${OUTPUT_ROOT}/fixed_l${LATENCY}"
LATENCY_CONFIG="${OUTPUT_DIR}/latency.yaml"
DATA_ROOT="${OUTPUT_DIR}/dataset"
HF_PATH_PREFIX="${HF_PATH_PREFIX:-fixed-latency-l${LATENCY}/mikasa-intercept-grab-fast/teacher-rollouts-h1-success-${TRAIN_EPISODES}train-${VAL_EPISODES}val}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${MIKASA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${TASK_PY}" - "${LATENCY}" "${LATENCY_CONFIG}" <<'PY'
import sys
from pathlib import Path

latency = int(sys.argv[1])
Path(sys.argv[2]).write_text(
    'env:\n  env_fps: 20\n  obs_fps: 20\n  noop_action: [0, 0, 0, 0, 0, 0, 0]\n'
    'executor:\n  simulated_worker_capacity: 1\n'
    'scheduler:\n  ordering_policy: latest_ready\n  hold_policy: hold\n'
    '  hold_last_chunk_action: true\n'
    f'latency:\n  method: fixed\n  fixed_latency_ms: {latency * 50.0}\n'
)
PY

printf '[collect] InterceptGrabFast L%s, physical GPU %s, %s train + %s validation successes\n' "${LATENCY}" "${GPU_ID}" "${TRAIN_EPISODES}" "${VAL_EPISODES}"
"${TASK_PY}" - "${MIKASA_ROOT}" "${CHECKPOINT}" "${DATA_ROOT}" "${TOTAL_EPISODES}" "${LATENCY_CONFIG}" <<'PY'
import importlib.util
import sys
from pathlib import Path

source = Path(sys.argv[1]) / 'mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py'
spec = importlib.util.spec_from_file_location('mikasa_successful_h1_collector', source)
collector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)
# Remove the native collector's minimum-length gate in both collection and unbatching.
collector.MIN_EPISODE_LENGTH_TO_SAVE = 0
collector.collect_for_env(
    env_id='InterceptGrabFast-VLA-v0', checkpoint=sys.argv[2],
    path_to_save_data=sys.argv[3], num_train_data=int(sys.argv[4]),
    latency_config=sys.argv[5], keep_failed=False, num_envs=1,
)
PY

"${RLDS_PY}" "${MIKASA_ROOT}/utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py" \
  --task InterceptGrabFast-VLA-v0 --repo-root "${MIKASA_ROOT}" \
  --npz-root "${DATA_ROOT}/data_npz" --rlds-root "${DATA_ROOT}/data_rlds"

"${LEROBOT_PY}" "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py" \
  --task InterceptGrabFast-VLA-v0 --repo-root "${MIKASA_ROOT}" \
  --rlds-root "${DATA_ROOT}/data_rlds" --lerobot-root "${DATA_ROOT}/data_lerobot" \
  --repo-id-template 'local/{task}_h1_train' --fps 20 --split "train[:${TRAIN_EPISODES}]"
"${LEROBOT_PY}" "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py" \
  --task InterceptGrabFast-VLA-v0 --repo-root "${MIKASA_ROOT}" \
  --rlds-root "${DATA_ROOT}/data_rlds" --lerobot-root "${DATA_ROOT}/data_lerobot" \
  --repo-id-template 'local/{task}_h1_val' --fps 20 --split "train[${TRAIN_EPISODES}:${TOTAL_EPISODES}]"

"${TASK_PY}" - "${DATA_ROOT}" "${CHECKPOINT}" "${LATENCY}" "${GPU_ID}" "${TRAIN_EPISODES}" "${VAL_EPISODES}" "${HF_REPO_ID}" "${HF_PATH_PREFIX}" "${HF_PRIVATE:-0}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi
from scripts.starvla.collect import _write_mikasa_contract

data = Path(sys.argv[1])
converted = data / 'data_lerobot/local'
shutil.move(str(converted / 'intercept_grab_fast_vla_v0_h1_train'), str(data / 'lerobot'))
shutil.move(str(converted / 'intercept_grab_fast_vla_v0_h1_val'), str(data / 'lerobot__val'))
_write_mikasa_contract(data / 'lerobot', data / 'lerobot__val')
metadata = json.loads((data / 'data_npz/intercept_grab_fast_vla_v0/metadata.json').read_text())
provenance = {
    'task': 'InterceptGrabFast-VLA-v0', 'teacher_checkpoint': sys.argv[2],
    'latency_frames': int(sys.argv[3]), 'latency_ms': int(sys.argv[3]) * 50.0,
    'env_fps': 20, 'gpu_id': sys.argv[4],
    'train_episodes': int(sys.argv[5]), 'val_episodes': int(sys.argv[6]),
    'acceptance': 'success_once', 'return_threshold': None, 'minimum_length': None,
    'collector_metadata': metadata, 'action_labels': 'issued H1 commands',
}
(data / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
api = HfApi()
api.create_repo(repo_id=sys.argv[7], repo_type='dataset', private=sys.argv[9] == '1', exist_ok=True)
commit = api.upload_folder(
    repo_id=sys.argv[7], repo_type='dataset', folder_path=str(data), path_in_repo=sys.argv[8],
    allow_patterns=['lerobot/**', 'lerobot__val/**', 'provenance.json'],
    ignore_patterns=['**/.cache/**'], commit_message=f'Collect successful InterceptGrabFast L{sys.argv[3]} H1 data',
)
print(commit.commit_url)
PY

printf '[done] local: %s\n' "${DATA_ROOT}"
printf '[done] HF: https://huggingface.co/datasets/%s/tree/main/%s\n' "${HF_REPO_ID}" "${HF_PATH_PREFIX}"
