#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/tasks/mikasa_intercept_grab_fast/collect_fixed_latency_dataset.sh \
    [--package-only] \
    <latency-frames> <physical-gpu-id> <ppo-checkpoint.pt> <output-root> <hf-dataset-repo>

Example: collect at latency 2 (100 ms) on physical GPU 3:
  bash scripts/tasks/mikasa_intercept_grab_fast/collect_fixed_latency_dataset.sh \
    2 3 outputs/mikasa_fixed_latency_ppo/publish/fixed_l2/checkpoint.pt \
    outputs/mikasa_successful_h1 your-org/mikasa-teacher-rollouts

Resume after collection completed (reuse existing NPZ episodes):
  bash scripts/tasks/mikasa_intercept_grab_fast/collect_fixed_latency_dataset.sh \
    --package-only 1 3 outputs/mikasa_fixed_latency_ppo/publish/fixed_l1/checkpoint.pt \
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
  UV                  uv executable; default: uv. Creates converter environments.
  LEROBOT_PY          Optional existing LeRobot interpreter instead of uv run.
  HF_PRIVATE          1 creates a private repo; default: 0.
  HF_PATH_PREFIX      Override the latency-specific upload directory.
  HF_TOKEN            HF authentication; an existing Hub login also works.

Run with the MIKASA simulator environment installed. Packaging converts NPZ
directly to LeRobot and uses uv to install its isolated conversion dependencies.
Use a fresh output root for new collection. --package-only skips collection and
preserves the saved NPZ episodes and latency config.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
PACKAGE_ONLY=0
if [[ "${1:-}" == "--package-only" ]]; then
  PACKAGE_ONLY=1
  shift
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
UV="${UV:-uv}"
if [[ -n "${LEROBOT_PY:-}" ]]; then
  LEROBOT_CMD=("${LEROBOT_PY}")
else
  LEROBOT_CMD=("${UV}" run --project "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot" --python 3.10 python)
fi
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

if [[ "${PACKAGE_ONLY}" == "0" ]]; then
"${TASK_PY}" - "${LATENCY}" "${LATENCY_CONFIG}" <<'PY'
import sys
from pathlib import Path

latency = int(sys.argv[1])
Path(sys.argv[2]).write_text(
    'env:\n  env_fps: 20\n  obs_fps: 20\n  noop_action: [0, 0, 0, 0, 0, 0, 0]\n'
    'executor:\n  simulated_worker_capacity: 1\n'
    'scheduler:\n  ordering_policy: latest_ready\n  hold_policy: hold\n'
    '  hold_last_chunk_action: true\n'
    f'latency:\n  method: fixed\n  fixed_latency_ms: {latency * 50.0}\n  seed: 0\n'
)
PY

printf '[collect] InterceptGrabFast L%s, physical GPU %s, %s train + %s validation successes\n' "${LATENCY}" "${GPU_ID}" "${TRAIN_EPISODES}" "${VAL_EPISODES}"
"${TASK_PY}" - "${CHECKPOINT}" "${DATA_ROOT}" "${TOTAL_EPISODES}" "${LATENCY_CONFIG}" <<'PYCFG'
import sys
from pathlib import Path
import yaml
from latency_bench.core.config import load_config
from latency_bench.data.rollout import collect_rollouts
checkpoint = Path(sys.argv[1])
config = load_config(sys.argv[4])
config.update(experiment={"name": "mikasa_h1_collection", "seed": 0},
    policy={"type": "mikasa_ppo", "source": "local", "root": str(checkpoint.parent), "checkpoint": checkpoint.name},
    evaluation={"episodes": int(sys.argv[3]), "max_env_steps": 60},
    logging={"output_dir": sys.argv[2]}, collection={"label_mode": "successful_h1"})
config["env"].update(name="mikasa_intercept_grab_fast", simulator_device="gpu")
config["executor"].update(mode="simulated", parallel_envs=1, inference_devices=["cuda:0"])
collect_rollouts(config)
PYCFG
else
  printf '[package] Reusing collected episodes in %s/data_npz\n' "${DATA_ROOT}"
fi

"${TASK_PY}" - "${DATA_ROOT}" "${TRAIN_EPISODES}" "${VAL_EPISODES}" <<'PYCFG'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
(root / "convert.yaml").write_text(yaml.safe_dump({"conversion": {
    "format": "lerobot", "task": "mikasa", "input_dir": str(root / "data_npz/intercept_grab_fast_vla_v0"),
    "output_dir": str(root / "lerobot"), "train_episodes": int(sys.argv[2]), "val_episodes": int(sys.argv[3])}}))
PYCFG
"${LEROBOT_CMD[@]}" -m latency_bench convert --config "${DATA_ROOT}/convert.yaml"

"${LEROBOT_CMD[@]}" - "${DATA_ROOT}" "${CHECKPOINT}" "${LATENCY}" "${GPU_ID}" "${TRAIN_EPISODES}" "${VAL_EPISODES}" "${HF_REPO_ID}" "${HF_PATH_PREFIX}" "${HF_PRIVATE:-0}" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

data = Path(sys.argv[1])
metadata = json.loads((data / 'data_npz/intercept_grab_fast_vla_v0/metadata.json').read_text())
provenance = {
    'task': 'InterceptGrabFast-VLA-v0', 'teacher_checkpoint': sys.argv[2],
    'latency_frames': int(sys.argv[3]), 'latency_ms': int(sys.argv[3]) * 50.0,
    'env_fps': 20, 'gpu_id': sys.argv[4],
    'train_episodes': int(sys.argv[5]), 'val_episodes': int(sys.argv[6]),
    'acceptance': 'success_once', 'return_threshold': None, 'minimum_length': None,
    'collector_metadata': metadata, 'action_labels': 'issued H1 commands',
    'packaging': json.loads((data / 'packaging.json').read_text()),
}
(data / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
api = HfApi()
api.create_repo(repo_id=sys.argv[7], repo_type='dataset', private=sys.argv[9] == '1', exist_ok=True)
commit = api.upload_folder(
    repo_id=sys.argv[7], repo_type='dataset', folder_path=str(data), path_in_repo=sys.argv[8],
    allow_patterns=['lerobot/**', 'lerobot__val/**', 'packaging.json', 'provenance.json'],
    ignore_patterns=['**/.cache/**'], commit_message=f'Collect successful InterceptGrabFast L{sys.argv[3]} H1 data',
)
print(commit.commit_url)
PY

printf '[done] local: %s\n' "${DATA_ROOT}"
printf '[done] HF: https://huggingface.co/datasets/%s/tree/main/%s\n' "${HF_REPO_ID}" "${HF_PATH_PREFIX}"
