#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import transformers  # noqa: F401
import vizdoom
import vizdoom.gymnasium_wrapper  # noqa: F401

env = gym.make(
    "VizdoomDefendLine-MultiBinary-v1",
    render_mode="rgb_array",
    frame_skip=4,
)
env.reset()
env.close()
print("ok-openvla-defend-the-line")
PY
