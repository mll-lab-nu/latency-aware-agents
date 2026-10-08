#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "ale-py==0.8.1" "gymnasium[atari]==0.29.1" "autorom==0.6.1"
if ! "$PYTHON_BIN" -c "import ale_py, gymnasium as gym; gym.make('ALE/DemonAttack-v5').close()" >/dev/null 2>&1; then
  AutoROM --accept-license
fi
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import ale_py  # noqa: F401

env = gym.make("ALE/DemonAttack-v5", frameskip=4, repeat_action_probability=0.0)
env.close()
print("ok-demon-attack-env")
PY

echo "[install/env/demon_attack] done"
