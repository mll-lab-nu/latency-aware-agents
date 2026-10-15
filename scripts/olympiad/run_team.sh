#!/bin/bash
# Run one configuration of a one-agent-per-robot soccer task: the four-seat `team` task (A1, A2 vs B1, B2) or the
# two-seat `team_match` task (A1, A2 vs the scripted team: --task soccer-team-match-wallclock-program-standard --level L).
#   run_team.sh [--task NAME] --a opus|fable|astra|luna|oracle [--b opus|fable|astra|luna|oracle] [--level L] [--slowdown S] [--seed N|hidden:I] [--first A|B] [--tag TAG]
# Team A's seats are played by agents of model --a, team B's by agents of model --b (a team = the same model); a task
# without B seats plays team A against the scripted team of --level (default 2).
# Players as in run_duel.sh. Job: jobs/<tag>-<sport>-team-x<S>-s<seed>-<a>A-<b>B-<UTC time>.
set -euo pipefail
task=soccer-team-wallclock-program-standard a= b= level= slowdown= seed=1 first=A tag=team
while [ $# -gt 0 ]; do
  case $1 in
    --task) task=$2; shift 2;; --a) a=$2; shift 2;; --b) b=$2; shift 2;; --level) level=$2; shift 2;; --slowdown) slowdown=$2; shift 2;; --seed) seed=$2; shift 2;;
    --first) first=$2; shift 2;; --tag) tag=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$a" ] || { sed -n 2,4p "$0" >&2; exit 2; }
source "$(dirname "$0")/v02_run.sh"     # task paths and per-seat accounts of v0.2 tasks
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
export PYTHONPATH=$repo     # players/harnesses/ (the metering adapters; harbor_agents/ keeps shims)
HARBOR=${HARBOR:-harbor}
harbor_guard     # the pinned multi-seat fork, or stop: upstream harbor would run the agent with internet (v02_run.sh)
seats=$(task_seats "$(task_dir "$task")")
case " $seats " in *" B1 "*) [ -n "$b" ] || { echo "--b is required: the task has B seats" >&2; exit 2; }; [ -z "$level" ] || { echo "--level: only for a task against the scripted team" >&2; exit 2; };;
  *) [ -z "$b" ] || { echo "--b: the task has no B seats (team A plays the scripted team)" >&2; exit 2; }; level=${level:-2}; export RCB_LEVEL=$level;; esac
run_seed DUEL_SEED "$seed" "$(task_dir "$task")" $level     # a number, or hidden:I (v02_run.sh)
export FIRST_KICKOFF=$first THINK_OPENAI_ROUTE=azure
[ -z "$slowdown" ] || export RCB_SLOWDOWN=$slowdown
export AZURE_DEPLOYMENTS=${AZURE_DEPLOYMENTS:-gpt-6-astra,gpt-6-astra-2,gpt-6-astra-3} RCB_RENDER_VIDEO=${RCB_RENDER_VIDEO:-1}
# No web search for the model (PROTOCOL §5: it runs at the provider, past the network isolation).
seat() {   # seat <seat> <agent>: the harbor --seat/--seat-kwarg arguments
  case $2 in
    opus)   echo "--seat $1=players.harnesses.claude_think:ClaudeThink@anthropic/claude-opus-5-5 --seat-kwarg $1:version=2.1.284 --seat-kwarg $1:disallowed_tools=WebSearch,WebFetch";;
    fable)  echo "--seat $1=players.harnesses.claude_think:ClaudeThink@anthropic/claude-fable-5-1 --seat-kwarg $1:version=2.1.284 --seat-kwarg $1:disallowed_tools=WebSearch,WebFetch";;
    astra)  echo "--seat $1=players.harnesses.codex_openrouter:CodexOpenRouter@openrouter/openai/gpt-6-astra --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    luna)   echo "--seat $1=players.harnesses.codex_openrouter:CodexOpenRouter@openrouter/openai/gpt-6-luna --seat-kwarg $1:version=0.158.0 --seat-kwarg $1:reasoning_effort=medium --seat-kwarg $1:web_search=disabled";;
    oracle) echo "--seat $1=oracle";;
    *) echo "agent must be opus, fable, astra, luna or oracle" >&2; exit 2;;
  esac
}
route() { [ "$1" = astra ] && echo azure || echo openrouter; }
mode() { [ "$1" = oracle ] && echo oracle || echo model; }
if [ -z "${OPUS_ACCOUNT:-}" ] && [ -f .env.acct2 ] && grep -q '^CLAUDE_CODE_OAUTH_TOKEN_2=.' .env 2>/dev/null; then OPUS_ACCOUNT=2; fi
account() { case $1 in opus) echo ${OPUS_ACCOUNT:-1};; fable) echo ${FABLE_ACCOUNT:-1};; *) echo 1;; esac; }
args=""
for s in $seats; do
  p=$a; [ "${s:0:1}" = B ] && p=$b
  export THINK_OPENAI_ROUTE_$s=$(route $p) PLAYER_MODE_$s=$(mode $p)
  seat_account $s $(account $p)
  seat_model $s $p     # the seat's model lock THINK_MODEL_<seat> (players.toml; v02_run.sh)
  seat_codex $s $p     # no ChatGPT subscription inherited from the shell reaches this seat (v02_run.sh)
  args="$args $(seat $s $p)"
done
tdir=$(task_dir "$task"); task=$(basename "$tdir")
if [ -n "$b" ]; then vs=${a}A-${b}B; else vs=${a}A-L$level; fi
job=$tag-${task%%-*}-$( [ -n "$b" ] && echo team || echo teammatch )${slowdown:+-x$slowdown}-s$seed_tag-$vs-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p jobs
[ -d "$tdir" ] || { echo "no task $tdir" >&2; exit 2; }
# shellcheck disable=SC2086
snapshot_watch $job     # program snapshots (v02_run.sh; RO_SNAPSHOTS=0: off)
$HARBOR run -p "$tdir" $args --env-file .env -n 1 -k 1 -r 0 -o jobs --job-name $job > jobs/$job.log 2>&1
r=$(ls jobs/$job/*/verifier/reward.json 2>/dev/null | head -1 || true)
echo "$job: $( [ -n "$r" ] && python3 -c "import json,sys;print(json.load(open(sys.argv[1])))" "$r" || echo NO-REWARD)"
