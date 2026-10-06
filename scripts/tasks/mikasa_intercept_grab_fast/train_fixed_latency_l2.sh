#!/usr/bin/env bash
# Train/evaluate the InterceptGrabFast small PPO teacher at latency L2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${1:?usage: $0 GPU_ID [train|eval|upload|both|full|all]}"
MODE="${2:-train}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LATENCIES="2"

exec bash "${SCRIPT_DIR}/train_fixed_latency_small_policies.sh" "${MODE}"
