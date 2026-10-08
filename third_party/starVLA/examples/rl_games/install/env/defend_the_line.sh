#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "vizdoom==1.3.0" "gymnasium==0.29.1"
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401

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
        print(f"ok-defend-the-line-env:{env_id}")
        break
    except Exception as exc:
        last_exc = exc
else:
    raise RuntimeError(f"could not create any Defend the Line VizDoom env: {last_exc}")
PY

echo "[install/env/defend_the_line] done"
