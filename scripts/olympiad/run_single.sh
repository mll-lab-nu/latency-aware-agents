#!/bin/bash
# Run one configuration of a single-player Olympiad task (default tasks/tabletennis-match-wallclock-program-standard).
#   run_single.sh [--task NAME] --agent opus|fable|astra|astra-sub|sol-sub|luna|luna-sub|oracle|nop [--level L] [--slowdown S] [--seed N|hidden:I] [--tag TAG]
# (--level for a format with levels, else omitted; --slowdown for real-time sports, default the task's own)
# Without --seed: the first official seed of the task at that level and slowdown (plan.py: sport.toml [official] and
# the task's verifier). Players:
#   opus   Claude Code 2.1.284, claude-opus-5-5, on the Claude subscription token in .env
#   fable  Claude Code 2.1.284, claude-fable-5-1, on the same subscription
#   astra  Codex 0.158.0, gpt-6-astra, reasoning effort medium, on Azure with failover over gpt-6-astra/-2/-3
#          (one Astra match per key at a time)
#   astra-sub  Codex 0.158.0, gpt-6-astra, reasoning effort medium, on a ChatGPT (Pro) subscription: CODEX_ACCESS_TOKEN
#          and CODEX_ACCOUNT_ID from $CODEX_ENV_FILE (default .env), passed to the seat's metering proxy only
#   sol-sub    the same, GPT-6.1 SOL: Codex sends its family slug gpt-6-sol, the seat's proxy pins every call to
#          gpt-6.1-sol (players.toml deployment)
#   luna-sub   the same, gpt-6-luna
#   luna   Codex 0.158.0, openai/gpt-6-luna, reasoning effort medium, on OpenRouter
# The sim's metering proxy holds the credentials and strips the provider prefix for Azure.
# Job: jobs/<tag>-<sport>-<agent>[-L<L>][-x<S>]-s<seed, or hI for hidden:I>-<UTC time>; prints the job and its reward.json.
set -euo pipefail
task=tabletennis-match-wallclock-program-standard agent= level= slowdown= seed= tag=tt
while [ $# -gt 0 ]; do
  case $1 in
    --task) task=$2; shift 2;; --agent) agent=$2; shift 2;; --level) level=$2; shift 2;; --slowdown) slowdown=$2; shift 2;;
    --seed) seed=$2; shift 2;; --tag) tag=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$agent" ] || { sed -n 2,3p "$0" >&2; exit 2; }
source "$(dirname "$0")/v02_run.sh"     # task paths and per-seat accounts of v0.2 tasks
name=$(basename "$task")
if [ -z "$seed" ]; then
  seed=$(python3 "$(dirname "$0")/plan.py" seed "$name" ${level:+--level $level} ${slowdown:+--slowdown $slowdown}) \
    || { echo "no official seed for $task: give --seed" >&2; exit 2; }
fi
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
export PYTHONPATH=$repo     # players/harnesses/ (the metering adapters; harbor_agents/ keeps shims)
# Harbor with multi-seat support (docs/benchmarks/olympiad.md):
#   uv tool install git+https://github.com/CHIGUI0/harbor@3fc74678123a2eb1845938b099e9d7856f2c6f6d
HARBOR=${HARBOR:-harbor}
harbor_guard     # the pinned multi-seat fork, or stop: upstream harbor would run the agent with internet (v02_run.sh)
run_seed RCB_SEED "$seed" "$(task_dir "$task")" "$level"     # a number, or hidden:I (v02_run.sh)
if [ -n "$level" ]; then export RCB_LEVEL=$level; else unset RCB_LEVEL; fi   # a format with levels only
[ -z "$slowdown" ] || export RCB_SLOWDOWN=$slowdown     # real-time sports only
sport=${name%%-*}
seat_model A $agent     # the seat's model lock THINK_MODEL_A (players.toml; v02_run.sh)
seat_codex A $agent     # *-sub: the ChatGPT subscription (and pinned model) for seat A's proxy only (v02_run.sh); else cleared
task=$(task_dir "$task")
[ -d "$task" ] || { echo "no task $task" >&2; exit 2; }
job=$tag-$sport-$agent${level:+-L$level}${slowdown:+-x$slowdown}-s$seed_tag-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p jobs
# The task has one Harbor seat (A, service `main`): the agent is installed with internet, then plays offline.
# With a second Claude subscription configured (CLAUDE_CODE_OAUTH_TOKEN_2 in .env, and .env.acct2), Opus runs on it by
# default and Fable on the first (the second account has no Fable allowance).
if [ -z "${OPUS_ACCOUNT:-}" ] && [ -f .env.acct2 ] && grep -q '^CLAUDE_CODE_OAUTH_TOKEN_2=.' .env 2>/dev/null; then OPUS_ACCOUNT=2; fi
# No web search for the model (PROTOCOL §5: it runs at the provider, past the network isolation).
# Direct (no-code) play and its harness switches (DIRECT_HARNESS) are on branch archive/direct-play.
claude() {   # claude <model> <account 1|2>: account 2 runs from .env.acct2 (.env with CLAUDE_CODE_OAUTH_TOKEN_2's token)
  envf=.env; [ "$2" = 2 ] && envf=.env.acct2
  RCB_PLAYER_MODE=model RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-1} $HARBOR run -p $task \
    --seat A=players.harnesses.claude_think:ClaudeThink@anthropic/$1 --seat-kwarg A:version=2.1.284 \
    --seat-kwarg A:disallowed_tools=WebSearch,WebFetch \
    --env-file $envf -k 1 -n 1 -r 0 -o jobs --job-name $job
}
codex() {   # codex <route azure|openrouter> <openrouter-style model>
  RCB_PLAYER_MODE=model THINK_OPENAI_ROUTE=$1 RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-1} \
    AZURE_DEPLOYMENTS=${AZURE_DEPLOYMENTS:-gpt-6-astra,gpt-6-astra-2,gpt-6-astra-3} $HARBOR run -p $task \
    --seat A=players.harnesses.codex_openrouter:CodexOpenRouter@$2 \
    --seat-kwarg A:version=0.158.0 --seat-kwarg A:reasoning_effort=medium --seat-kwarg A:web_search=disabled \
    --env-file .env -k 1 -n 1 -r 0 -o jobs --job-name $job
}
codex_sub() {   # codex_sub <openai/model>: Codex logged in with ChatGPT; its seat's proxy holds the subscription
  RCB_PLAYER_MODE=model THINK_OPENAI_ROUTE=chatgpt RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-1} $HARBOR run -p $task \
    --seat A=players.harnesses.codex_chatgpt_think:CodexChatGPTThink@$1 \
    --seat-kwarg A:version=0.158.0 --seat-kwarg A:reasoning_effort=medium --seat-kwarg A:web_search=disabled \
    --env-file .env -k 1 -n 1 -r 0 -o jobs --job-name $job
}
snapshot_watch $job     # program snapshots (v02_run.sh; RO_SNAPSHOTS=0: off)
case $agent in
  opus)  claude claude-opus-5-5 ${OPUS_ACCOUNT:-1} ;;
  fable) claude claude-fable-5-1 ${FABLE_ACCOUNT:-1} ;;
  astra) codex azure openrouter/openai/gpt-6-astra ;;
  astra-sub) codex_sub openai/gpt-6-astra ;;
  sol-sub)   codex_sub openai/gpt-6-sol ;;      # pinned to gpt-6.1-sol by the seat's proxy
  luna-sub)  codex_sub openai/gpt-6-luna ;;
  luna)  codex openrouter openrouter/openai/gpt-6-luna ;;
  oracle|nop)
    RCB_PLAYER_MODE=oracle RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-0} $HARBOR run -p $task --seat A=$agent \
      -k 1 -n 1 -r 0 -o jobs --job-name $job ;;
  *) echo "agent must be opus, fable, astra, astra-sub, sol-sub, luna, luna-sub, oracle or nop" >&2; exit 2;;
esac > jobs/$job.log 2>&1
r=$(ls jobs/$job/*/verifier/reward.json 2>/dev/null | head -1 || true)
echo "$job: $( [ -n "$r" ] && python3 -c "import json,sys;print(json.load(open(sys.argv[1])))" "$r" || echo NO-REWARD)"
