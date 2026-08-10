#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  PROFILE_REF=<device>/<instance>/<workload>/<revision> bash scripts/bash_scripts/cross_device/download_profile.sh

Options:
  --profile-ref <ref>   Profile ref under profiles/, e.g. rtx4090/<instance>/openvla-flappy/fit-0-4.
  --config <path>       Cross-device config. Default: configs/experiments/cross_device.yaml

Environment:
  HF_TOKEN              Required if the profile repo is private.

The profile is downloaded into:
  <artifact_root>/profiles/<profile-ref>
EOF
}

CONFIG="configs/experiments/cross_device.yaml"
PROFILE_REF="${PROFILE_REF:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --profile-ref)
      PROFILE_REF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${PROFILE_REF}" ]]; then
  echo "error: --profile-ref or PROFILE_REF is required" >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: Hugging Face CLI command 'hf' is not available in PATH" >&2
  exit 127
fi

read -r HF_REPO HF_REPO_TYPE ARTIFACT_ROOT < <(
  python - "${CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(
    cfg["hf_repo"],
    cfg.get("hf_repo_type", "dataset"),
    cfg.get("artifact_root", "results/cross_device"),
)
PY
)

REMOTE_DIR="profiles/${PROFILE_REF}"
LOCAL_DIR="${ARTIFACT_ROOT}/${REMOTE_DIR}"

echo "==> Downloading ${HF_REPO}:${REMOTE_DIR}"
hf download "${HF_REPO}" \
  --repo-type "${HF_REPO_TYPE}" \
  --include "${REMOTE_DIR}/**" \
  --local-dir "${ARTIFACT_ROOT}"

if [[ ! -f "${LOCAL_DIR}/DONE" ]]; then
  echo "error: downloaded profile is missing DONE marker: ${LOCAL_DIR}/DONE" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_DIR}/profile.json" ]]; then
  echo "error: downloaded profile is missing profile.json: ${LOCAL_DIR}/profile.json" >&2
  exit 1
fi

echo "PROFILE_REF=${PROFILE_REF}"
echo "PROFILE_DIR=${LOCAL_DIR}"
echo "PROFILE_PATH=${LOCAL_DIR}/profile.json"
