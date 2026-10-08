#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import ale_py  # noqa: F401
import gymnasium as gym
import transformers  # noqa: F401

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T  # noqa: F401

env = gym.make("ALE/DemonAttack-v5", frameskip=4, repeat_action_probability=0.0)
env.reset()
env.close()
print("ok-gr00t-demon-attack")
PY
