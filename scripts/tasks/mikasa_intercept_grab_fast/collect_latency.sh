#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}"
exec third_party/MIKASA-Robo/.venv/bin/python \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
  --env-id InterceptGrabFast-VLA-v0 "$@"
