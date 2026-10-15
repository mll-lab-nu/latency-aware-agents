#!/bin/bash
# Run one configuration of a two-player Olympiad task (default tasks/tabletennis-duel-wallclock-program-standard).
#   run_duel.sh [--task NAME] --a opus|fable|astra|astra-sub|sol|sol-sub|luna|luna-sub|oracle --b opus|fable|astra|astra-sub|sol|sol-sub|luna|luna-sub|oracle [--slowdown S] [--seed N|hidden:I] [--first A|B] [--tag TAG]
# Players as in run_single.sh: opus and fable (Claude Code 2.1.284, subscription), astra (Codex 0.158.0, medium, Azure
# with failover; its route is set for its seat only), sol (Codex 0.158.0, medium, GPT-6.1 SOL on a second Azure resource:
# route azure_sol for its seat only; AZURE_SOL_ENDPOINT / AZURE_SOL_API_KEY [/ AZURE_SOL_DEPLOYMENTS] are read from
# $SOL_ENV_FILE, default .env.sol, and passed to that seat's metering proxy only) and luna (Codex 0.158.0, medium,
# OpenRouter). astra-sub: Astra on a ChatGPT subscription (Codex 0.158.0 logged in with ChatGPT); CODEX_ACCESS_TOKEN /
# CODEX_ACCOUNT_ID are read from $CODEX_ENV_FILE (default .env) and passed to that seat's metering proxy only. sol-sub
# (GPT-6.1 SOL: Codex sends gpt-6-sol, the seat's proxy pins every call to gpt-6.1-sol) and luna-sub: the same.
# Job: jobs/<tag>-duel-x<S>-s<seed, or hI for hidden:I>-<a>A-<b>B-<UTC time>.
set -euo pipefail
task=tabletennis-duel-wallclock-program-standard a= b= slowdown= seed=1 first=A tag=tt
while [ $# -gt 0 ]; do
  case $1 in
    --task) task=$2; shift 2;; --a) a=$2; shift 2;; --b) b=$2; shift 2;; --slowdown) slowdown=$2; shift 2;; --seed) seed=$2; shift 2;;
    --first) first=$2; shift 2;; --tag) tag=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$a" ] && [ -n "$b" ] || { sed -n 2,3p "$0" >&2; exit 2; }
source "$(dirname "$0")/v02_run.sh"     # task paths and per-seat accounts of v0.2 tasks
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
export PYTHONPATH=$repo     # players/harnesses/ (the metering adapters; harbor_agents/ keeps shims)
# Harbor with multi-seat support (docs/benchmarks/olympiad.md):
#   uv tool install git+https://github.com/CHIGUI0/harbor@3fc74678123a2eb1845938b099e9d7856f2c6f6d
HARBOR=${HARBOR:-harbor}
harbor_guard     # the pinned multi-seat fork, or stop: upstream harbor would run the agent with internet (v02_run.sh)
run_seed DUEL_SEED "$seed" "$(task_dir "$task")"     # a number, or hidden:I (v02_run.sh)
export FIRST_RECEIVER=$first FIRST_PLAYER=$first FIRST_KICKOFF=$first THINK_OPENAI_ROUTE=azure
[ -z "$slowdown" ] || export RCB_SLOWDOWN=$slowdown     # real-time sports only
export AZURE_DEPLOYMENTS=${AZURE_DEPLOYMENTS:-gpt-6-astra,gpt-6-astra-2,gpt-6-astra-3} RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-1}
# sol's Azure deployment. Codex sends its own slug for the family, gpt-6-sol (Codex 0.158.0 has metadata for it, none for
# gpt-6.1-sol), and the seat's pinned route azure_sol serves every request with this deployment (players.toml).
SOL_DEPLOYMENT=${SOL_DEPLOYMENT:-gpt-6.1-sol}
# No web search for the model (PROTOCOL §5: it runs at the provider, past the network isolation).
# Direct (no-code) play and its harness switches (DIRECT_HARNESS) are on branch archive/direct-play.
seat() {   # seat <A|B> <agent>: the harbor --seat/--seat-kwarg arguments
  case $2 in
    opus)   echo "--seat $1=players.harnesses.claude_think:ClaudeThink@anthropic/claude-opus-5-5 --seat-kwarg $1:version=2.1.284 --seat-kwarg $1:disallowed_tools=WebSearch,WebFetch";;
    fable)  echo "--seat $1=players.harnesses.claude_think:ClaudeThink@anthropic/claude-fable-5-1 --seat-kwarg $1:version=2.1.284 --seat-kwarg $1:disallowed_tools=WebSearch,WebFetch";;
    astra)  echo "--seat $1=players.harnesses.codex_openrouter:CodexOpenRouter@openrouter/openai/gpt-6-astra --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    astra-sub) echo "--seat $1=players.harnesses.codex_chatgpt_think:CodexChatGPTThink@openai/gpt-6-astra --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    sol-sub)   echo "--seat $1=players.harnesses.codex_chatgpt_think:CodexChatGPTThink@openai/gpt-6-sol --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    luna-sub)  echo "--seat $1=players.harnesses.codex_chatgpt_think:CodexChatGPTThink@openai/gpt-6-luna --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    sol)    echo "--seat $1=players.harnesses.codex_openrouter:CodexOpenRouter@openrouter/openai/gpt-6-sol --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    luna)   echo "--seat $1=players.harnesses.codex_openrouter:CodexOpenRouter@openrouter/openai/gpt-6-luna --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    oracle) echo "--seat $1=oracle";;
    *) echo "agent must be opus, fable, astra, astra-sub, sol, sol-sub, luna, luna-sub or oracle" >&2; exit 2;;
  esac
}
route() { case $1 in astra) echo azure;; sol) echo azure_sol;; *-sub) echo chatgpt;; *) echo openrouter;; esac; }   # each seat's route for OpenAI-style (Codex) calls (*-sub: the ChatGPT backend only; its proxy refuses every other call)
export THINK_OPENAI_ROUTE_A=$(route $a) THINK_OPENAI_ROUTE_B=$(route $b)
seat_model A $a; seat_model B $b     # each seat's model lock THINK_MODEL_<seat> (players.toml; v02_run.sh)
# sol: the second Azure resource's credentials go to its seat only (AZURE_SOL_<VAR>_<seat>; never exported unsuffixed)
sol_seat() {
  local f=${SOL_ENV_FILE:-.env.sol} v vals=()
  [ -r "$f" ] || { echo "sol needs $f (AZURE_SOL_ENDPOINT, AZURE_SOL_API_KEY)" >&2; exit 2; }
  while IFS= read -r -d '' v; do vals+=("$v"); done < <(set -a; . "$f"; printf '%s\0' "${AZURE_SOL_ENDPOINT:-}" "${AZURE_SOL_API_KEY:-}" "${AZURE_SOL_DEPLOYMENTS:-}")
  [ -n "${vals[0]}" ] && [ -n "${vals[1]}" ] || { echo "$f lacks AZURE_SOL_ENDPOINT or AZURE_SOL_API_KEY" >&2; exit 2; }
  export "AZURE_SOL_ENDPOINT_$1=${vals[0]}" "AZURE_SOL_API_KEY_$1=${vals[1]}" "AZURE_SOL_DEPLOYMENTS_$1=${vals[2]:-$SOL_DEPLOYMENT}"
}
[ "$a" != sol ] || sol_seat A; [ "$b" != sol ] || sol_seat B
seat_codex A $a; seat_codex B $b     # *-sub: the ChatGPT subscription (and pinned model) for its own seat's proxy only (v02_run.sh)
# With a second Claude subscription configured (CLAUDE_CODE_OAUTH_TOKEN_2 in .env, and .env.acct2), Opus runs on it by
# default and Fable on the first (the second account has no Fable allowance).
if [ -z "${OPUS_ACCOUNT:-}" ] && [ -f .env.acct2 ] && grep -q '^CLAUDE_CODE_OAUTH_TOKEN_2=.' .env 2>/dev/null; then OPUS_ACCOUNT=2; fi
# Claude subscription per seat: OPUS_ACCOUNT / FABLE_ACCOUNT (1 = CLAUDE_CODE_OAUTH_TOKEN, 2 = CLAUDE_CODE_OAUTH_TOKEN_2)
account() { case $1 in opus) echo ${OPUS_ACCOUNT:-1};; fable) echo ${FABLE_ACCOUNT:-1};; *) echo 1;; esac; }
seat_account A $(account $a); seat_account B $(account $b)
mode() { [ "$1" = oracle ] && echo oracle || echo model; }
export PLAYER_MODE_A=$(mode $a) PLAYER_MODE_B=$(mode $b)
tdir=$(task_dir "$task"); task=$(basename "$tdir")
job=$tag-${task%%-*}-duel${slowdown:+-x$slowdown}-s$seed_tag-${a}A-${b}B-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p jobs
# shellcheck disable=SC2046
[ -d "$tdir" ] || { echo "no task $tdir" >&2; exit 2; }
snapshot_watch $job     # program snapshots (v02_run.sh; RO_SNAPSHOTS=0: off)
$HARBOR run -p "$tdir" $(seat A $a) $(seat B $b) \
  --env-file .env -n 1 -k 1 -r 0 -o jobs --job-name $job > jobs/$job.log 2>&1
r=$(ls jobs/$job/*/verifier/reward.json 2>/dev/null | head -1 || true)
echo "$job: $( [ -n "$r" ] && python3 -c "import json,sys;print(json.load(open(sys.argv[1])))" "$r" || echo NO-REWARD)"
