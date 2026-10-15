#!/bin/bash
# run_model.sh REPO TASK_DIR sol|astra LEVEL SEED TAG : one single-player match (Codex 0.158.0, medium, no web search),
# as scripts/olympiad/run_single.sh (astra) and run_duel.sh's sol seat (azure_sol route, credentials for seat A only,
# read from SOL_ENV_FILE). SOL holds the shared lock ${SOL_LOCK:-/tmp/sol.lock} for the whole match.
set -euo pipefail
repo=$1 task=$2 agent=$3 level=$4 seed=$5 tag=$6
cd "$repo"
source scripts/olympiad/v02_run.sh
export PYTHONPATH=$repo RCB_SEED=$seed RCB_LEVEL=$level RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-1} RCB_PLAYER_MODE=model
HARBOR=${HARBOR:-harbor}
seat_model A "$agent"
case $agent in
  astra) export THINK_OPENAI_ROUTE=azure THINK_OPENAI_ROUTE_A=azure
         export AZURE_DEPLOYMENTS=${AZURE_DEPLOYMENTS:-gpt-6-astra,gpt-6-astra-2,gpt-6-astra-3}
         model=openrouter/openai/gpt-6-astra; lock=/dev/null ;;
  sol)   export THINK_OPENAI_ROUTE_A=azure_sol
         f=${SOL_ENV_FILE:-.env.sol}
         vals=()
         while IFS= read -r -d '' v; do vals+=("$v"); done < <(set -a; . "$f"; printf '%s\0' "${AZURE_SOL_ENDPOINT:-}" "${AZURE_SOL_API_KEY:-}" "${AZURE_SOL_DEPLOYMENTS:-}")
         [ -n "${vals[0]}" ] && [ -n "${vals[1]}" ] || { echo "no SOL credentials" >&2; exit 2; }
         export "AZURE_SOL_ENDPOINT_A=${vals[0]}" "AZURE_SOL_API_KEY_A=${vals[1]}" "AZURE_SOL_DEPLOYMENTS_A=${vals[2]:-gpt-6.1-sol}"
         model=openrouter/openai/gpt-6-sol; lock=${SOL_LOCK:-/tmp/sol.lock} ;;
  *) echo "sol or astra" >&2; exit 2 ;;
esac
job=$tag-badminton-$agent-L$level-s$seed-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p jobs
flock "$lock" $HARBOR run -p "$task" \
  --seat A=players.harnesses.codex_openrouter:CodexOpenRouter@$model --seat-kwarg A:version=0.158.0 \
  --seat-kwarg A:reasoning_effort=medium --seat-kwarg A:web_search=disabled \
  --env-file .env -k 1 -n 1 -r 0 -o jobs --job-name "$job" > "jobs/$job.log" 2>&1 || true
r=$(ls jobs/$job/*/verifier/reward.json 2>/dev/null | head -1 || true)
echo "$job: $( [ -n "$r" ] && python3 -c "import json,sys;print(json.load(open(sys.argv[1])))" "$r" || echo NO-REWARD)"
