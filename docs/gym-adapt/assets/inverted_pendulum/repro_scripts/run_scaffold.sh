#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/inverted_pendulum
MUJOCO_GL=egl .venv/bin/python scripts/scaffold_gymnasium_mujoco.py \
  --report-path artifacts/inverted_pendulum/api_probe.json \
  --env-yaml-path artifacts/inverted_pendulum/env_block.yaml \
  | tee artifacts/inverted_pendulum/scaffold_stdout.txt
