#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import os
import transformers  # noqa: F401

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI  # noqa: F401

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
env.reset()
frame = env.render()
env.close()
if frame is None:
    raise SystemExit("Flappy render returned None")
print("ok-pi0-flappy")
PY
