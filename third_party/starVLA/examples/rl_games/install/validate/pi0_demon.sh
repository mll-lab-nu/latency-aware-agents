#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import transformers  # noqa: F401
import gymnasium as gym
import ale_py  # noqa: F401

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI  # noqa: F401

env = gym.make("ALE/DemonAttack-v5", frameskip=4, repeat_action_probability=0.0)
env.close()
print("ok-pi0-demon")
PY
