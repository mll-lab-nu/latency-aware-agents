#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import vizdoom
import vizdoom.gymnasium_wrapper  # noqa: F401

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI  # noqa: F401

attempts = [
    ("VizdoomDefendLine-MultiBinary-v1", {}),
    ("VizdoomDefendLine-MultiBinary-v0", {}),
    ("VizdoomDefendLine-v1", {"max_buttons_pressed": 0}),
    ("VizdoomDefendLine-v0", {"max_buttons_pressed": 0}),
]
last_exc = None
for env_id, kwargs in attempts:
    try:
        env = gym.make(env_id, render_mode="rgb_array", frame_skip=4, **kwargs)
        env.reset()
        env.close()
        print(f"ok-pi0-defend-the-line:{env_id}")
        break
    except Exception as exc:
        last_exc = exc
else:
    raise RuntimeError(f"could not create any Defend the Line VizDoom env: {last_exc}")
PY
