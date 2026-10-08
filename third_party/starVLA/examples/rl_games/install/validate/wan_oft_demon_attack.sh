#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import ale_py  # noqa: F401
import gymnasium as gym
from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: F401
from starVLA.model.framework.WM4A.WanOFT import Wan_OFT  # noqa: F401
from transformers import UMT5EncoderModel  # noqa: F401

env = gym.make("ALE/DemonAttack-v5", frameskip=4, repeat_action_probability=0.0)
env.reset()
env.close()
print("ok-wan-oft-demon-attack")
PY
