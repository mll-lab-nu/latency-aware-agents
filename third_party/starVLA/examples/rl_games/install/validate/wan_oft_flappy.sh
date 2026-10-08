#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: F401
from starVLA.model.framework.WM4A.WanOFT import Wan_OFT  # noqa: F401
from transformers import UMT5EncoderModel  # noqa: F401

env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
env.reset()
frame = env.render()
env.close()

if frame is None:
    raise SystemExit("Flappy render returned None")

print("ok-wan-oft-flappy")
PY
