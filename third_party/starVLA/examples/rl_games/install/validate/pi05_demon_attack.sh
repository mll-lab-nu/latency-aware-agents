#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import ale_py  # noqa: F401
import gymnasium as gym

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3  # noqa: F401

env = gym.make("ALE/DemonAttack-v5", frameskip=4, repeat_action_probability=0.0)
env.reset()
env.close()
print("ok-pi05-demon-attack")
PY
