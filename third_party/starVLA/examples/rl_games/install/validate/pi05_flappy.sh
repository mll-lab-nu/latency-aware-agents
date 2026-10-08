#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import transformers  # noqa: F401

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3  # noqa: F401

env = gym.make("FlappyBird-v0")
env.reset()
env.close()
print("ok-pi05-flappy")
PY
