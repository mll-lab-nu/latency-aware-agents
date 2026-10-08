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
    ("VizdoomDeadlyCorridor-MultiBinary-v1", {}),
    ("VizdoomDeadlyCorridor-MultiBinary-v0", {}),
    ("VizdoomDeadlyCorridor-v1", {"max_buttons_pressed": 0}),
    ("VizdoomDeadlyCorridor-v0", {"max_buttons_pressed": 0}),
]
last_exc = None
for env_id, kwargs in attempts:
    try:
        env = gym.make(env_id, render_mode="rgb_array", frame_skip=4, **kwargs)
        env.reset()
        env.close()
        print(f"ok-deadly-corridor-env:{env_id}")
        break
    except Exception as exc:
        last_exc = exc
else:
    raise RuntimeError(f"could not create any Deadly Corridor VizDoom env: {last_exc}")
PY

echo "[install/env/deadly_corridor] done"
