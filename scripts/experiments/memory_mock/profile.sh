#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Usage: bash scripts/experiments/memory_mock/profile.sh

Run the established Flappy/QwenOFT live profiling workflow while irregular
memory copying runs on the SAME GPU. Fit Temporal, AR(1), ARMA(1,1) on shared
training sessions, compare on reserved sessions, and export a reusable bundle.

Defaults: GPU=0, REPEATS=5, EPISODES=10, FIT_SESSION_IDS=0,1,2,
MAX_ENV_STEPS=3600, METHODS=temporal,ar1,arma, LOAD_SEED=42, BUFFER_MIB=512.
OFF: 5-15s; ON: 80% 0.2-0.8s, 20% 2-5s. Every session gets its own load seed.
Episode count is not a minimum measurement duration; inspect observed ON events.

Set OUTPUT_DIR to a persistent fresh directory. Rerun the same settings to
resume completed sessions. SKIP_INSTALL=1 reuses the installed model environment.
MEMORY_MOCK_CONDITION=idle selects the allocation-matched control.
Uploads are ON after successful profiling, fitting and comparison.
HF_REPO defaults to latency-sensitive-bench/irregular-temporal-profiles.
Create a private dataset repo if absent; preserve existing visibility.
HF_TOKEN or hf auth login must allow writes. SKIP_UPLOAD=1 keeps results local.
See scripts/experiments/memory_mock/README.md for settings and interpretation.
EOF
    exit 0
fi
export MEMORY_MOCK_CONDITION="${MEMORY_MOCK_CONDITION:-irregular}"
export METHODS="${METHODS:-temporal,ar1,arma}"
export COMPARE_PROFILES=1
export SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
export HF_REPO="${HF_REPO:-latency-sensitive-bench/irregular-temporal-profiles}"
export HF_CREATE_REPO="${HF_CREATE_REPO:-1}"
export HF_PATH_PREFIX="${HF_PATH_PREFIX:-experiments/irregular-memory}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$REPO_ROOT")/outputs/memory_mock/profiling/$(date -u +%Y%m%dT%H%M%SZ)}"
exec bash "$REPO_ROOT/scripts/experiments/temporal_profiles/profile.sh" "$@"
