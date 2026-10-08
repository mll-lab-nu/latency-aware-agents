#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORIGINAL_CWD="$(pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  run_experiment.sh <config.yaml> [key=value ...]

Example:
  bash examples/rl_games/scripts/run_experiment.sh \
    examples/rl_games/experiments/openvla/scratch/mixed_latency/flappy.yaml \
    run_id=test_run trainer.max_train_steps=100

Notes:
  - Conda activation is controlled by config conda.enabled / conda.env_name.
  - Set STARVLA_CONDA_ENV to override the configured env name.
  - Set conda.enabled=false as an override to use the current Python env.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG_PATH="$1"
shift

if [[ "$CONFIG_PATH" == "-h" || "$CONFIG_PATH" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  if [[ -f "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
  elif [[ -f "$ORIGINAL_CWD/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$ORIGINAL_CWD/$CONFIG_PATH"
  fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

CONDA_ENABLED_OVERRIDE=""
CONDA_ENV_OVERRIDE="${STARVLA_CONDA_ENV:-}"
for arg in "$@"; do
  case "$arg" in
    conda.enabled=*) CONDA_ENABLED_OVERRIDE="${arg#conda.enabled=}" ;;
    conda.env_name=*) CONDA_ENV_OVERRIDE="${arg#conda.env_name=}" ;;
  esac
done

CONFIG_PROBE="$(
  python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])

try:
    import yaml
    data = yaml.safe_load(path.read_text()) or {}
    conda = data.get("conda") or {}
    enabled = conda.get("enabled", True)
    env_name = conda.get("env_name") or ""
    model = data.get("model") or ""
except Exception:
    enabled = True
    env_name = ""
    model = ""
    current = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_value = line.strip()
        if indent == 0:
            current = [key_value.split(":", 1)[0]]
            if key_value.startswith("model:"):
                model = key_value.split(":", 1)[1].strip().strip("'\"")
        elif current == ["conda"] and ":" in key_value:
            key, value = key_value.split(":", 1)
            value = value.strip().strip("'\"")
            if key == "enabled":
                enabled = value.lower() not in {"false", "0", "no", "off"}
            elif key == "env_name":
                env_name = value

if not env_name and model:
    env_name = f"starvla_rl_games_{model}"

print("true" if enabled else "false")
print(env_name)
PY
)"

CONDA_ENABLED=""
CONDA_ENV_NAME=""
CONFIG_PROBE_LINE=0
while IFS= read -r line || [[ -n "$line" ]]; do
  ((CONFIG_PROBE_LINE += 1))
  case "$CONFIG_PROBE_LINE" in
    1) CONDA_ENABLED="$line" ;;
    2) CONDA_ENV_NAME="$line" ;;
  esac
done <<< "$CONFIG_PROBE"
CONDA_ENABLED="${CONDA_ENABLED:-true}"

if [[ -n "$CONDA_ENABLED_OVERRIDE" ]]; then
  case "$CONDA_ENABLED_OVERRIDE" in
    false|False|FALSE|0|no|No|NO|off|Off|OFF) CONDA_ENABLED="false" ;;
    *) CONDA_ENABLED="true" ;;
  esac
fi
if [[ -n "$CONDA_ENV_OVERRIDE" ]]; then
  CONDA_ENV_NAME="$CONDA_ENV_OVERRIDE"
fi

if [[ "$CONDA_ENABLED" == "true" ]]; then
  if [[ -z "$CONDA_ENV_NAME" ]]; then
    echo "Could not determine conda env name from config. Set conda.env_name or STARVLA_CONDA_ENV." >&2
    exit 1
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required for conda.enabled=true. Override with conda.enabled=false if current env is ready." >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "Conda env '${CONDA_ENV_NAME}' does not exist." >&2
    INSTALL_MODEL="${CONDA_ENV_NAME#starvla_rl_games_}"
    if [[ -z "$INSTALL_MODEL" || "$INSTALL_MODEL" == "$CONDA_ENV_NAME" ]]; then
      INSTALL_MODEL="openvla"
    fi
    echo "Install it first with: bash examples/rl_games/install/install_stack.sh ${INSTALL_MODEL}" >&2
    exit 1
  fi
  conda activate "$CONDA_ENV_NAME"
  echo "Using conda env: ${CONDA_ENV_NAME}"
fi

exec python examples/rl_games/scripts/run_experiment.py "$CONFIG_PATH" "$@"
