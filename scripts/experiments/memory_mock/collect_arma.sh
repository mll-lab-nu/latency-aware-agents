#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${REPO_ROOT}")}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/memory_mock/flappy_3090_rollouts_20min}"
export TEACHER_PROFILE_SET=irregular_20min
exec bash "${REPO_ROOT}/scripts/experiments/temporal_profiles/_collect_teacher.sh" arma "$@"
