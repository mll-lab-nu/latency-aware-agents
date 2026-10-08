#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import vizdoom
import vizdoom.gymnasium_wrapper  # noqa: F401
from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: F401
from starVLA.model.framework.WM4A.WanOFT import Wan_OFT  # noqa: F401
from transformers import UMT5EncoderModel  # noqa: F401

env = gym.make(
    "VizdoomDefendLine-MultiBinary-v1",
    render_mode="rgb_array",
    frame_skip=4,
)
env.reset()
env.close()
print("ok-wan-oft-defend-the-line")
PY
