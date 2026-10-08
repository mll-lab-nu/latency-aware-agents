#!/usr/bin/env bash
# Reasoning traces for the BASE (pre-flappy) bridge VLA, on the SAME balanced test set.
# The bridge checkpoint is self-contained (ships config.yaml + dataset_statistics.json + .pt);
# --checkpoint-kind bridge auto-rewrites its stale base_vlm path to the local base VLM.
# NOTE: the bridge action head is for robot (Bridge/RT-1) actions, so the action column is
# not meaningful for flappy -- the comparable artifact is the reasoning text.
set -euo pipefail

cd /workspace/starVLA
conda activate starvla_rl_games_openvla

HF_REPO_ID="StarVLA/Qwen3VL-OFT-Bridge-RT-1"
BASE_VLM_REPO="Qwen/Qwen3-VL-4B-Instruct"
LATENCIES="${LATENCIES:-0,3}"
CLASSES="${CLASSES:-NOOP,FLAP}"
PER_CLASS_SAMPLES="${PER_CLASS_SAMPLES:-20}"
SPLIT="${SPLIT:-validation}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUTPUT_CSV="/workspace/outputs/reasoning/bridge/reasoning_traces.csv"
HF_OUTPUT_REPO="talha15032/reasoning_trace_2"
HF_OUTPUT_SUBDIR="bridge"
LIVE_EVAL_EPISODES="${LIVE_EVAL_EPISODES:-2}"
LIVE_EVAL_LATENCIES="${LIVE_EVAL_LATENCIES:-0}"

python examples/rl_games/scripts/inspect_reasoning_trace.py \
    --checkpoint-kind bridge \
    --hf-repo-id "${HF_REPO_ID}" \
    --base-vlm-repo "${BASE_VLM_REPO}" \
    --env-name flappy \
    --latencies "${LATENCIES}" \
    --classes "${CLASSES}" \
    --per-class-samples "${PER_CLASS_SAMPLES}" \
    --split "${SPLIT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output-csv "${OUTPUT_CSV}" \
    --live-eval-episodes "${LIVE_EVAL_EPISODES}" \
    --live-eval-latencies "${LIVE_EVAL_LATENCIES}" \
    --two-forward-pass \
    --push-to-hub \
    --hf-output-repo "${HF_OUTPUT_REPO}" \
    --hf-output-subdir "${HF_OUTPUT_SUBDIR}"
