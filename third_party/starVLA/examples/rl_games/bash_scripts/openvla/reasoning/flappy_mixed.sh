#!/usr/bin/env bash
# Reasoning traces for the TRAINED mixed-latency flappy VLA.
# Builds a balanced per-latency test set (per-class samples per latency, deterministic
# first-N so it matches the bridge run on the same frames) and dumps a CSV + frames.
set -euo pipefail

cd /workspace/starVLA
conda activate starvla_rl_games_openvla

HF_REPO_ID="talha15032/openvla_bridge_flappy_latency_mixed_exp2"
HF_INCLUDE="steps_5000_state/**"
CKPT_SUBPATH="steps_5000_state"
BASE_VLM_REPO="Qwen/Qwen3-VL-4B-Instruct"
LATENCIES="${LATENCIES:-0,3}"
CLASSES="${CLASSES:-NOOP,FLAP}"
PER_CLASS_SAMPLES="${PER_CLASS_SAMPLES:-10}"
SPLIT="${SPLIT:-validation}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUTPUT_CSV="/workspace/outputs/reasoning/flappy_mixed/reasoning_traces.csv"
HF_OUTPUT_REPO="talha15032/reasoning_trace_2"
HF_OUTPUT_SUBDIR="flappy_mixed"
LIVE_EVAL_EPISODES="${LIVE_EVAL_EPISODES:-2}"
LIVE_EVAL_LATENCIES="${LIVE_EVAL_LATENCIES:-0}"

python examples/rl_games/scripts/inspect_reasoning_trace.py \
    --checkpoint-kind trained \
    --hf-repo-id "${HF_REPO_ID}" \
    --hf-include "${HF_INCLUDE}" \
    --ckpt-subpath "${CKPT_SUBPATH}" \
    --base-vlm-repo "${BASE_VLM_REPO}" \
    --env-name flappy \
    --recon-mode mixed_latency \
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
