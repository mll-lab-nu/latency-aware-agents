#!/bin/bash
# Run one configuration of a multi-car race (default tasks/racing-race-wallclock-program-standard): one agent per car.
#   run_race.sh [--task NAME] --players P1,P2,... (one per car in seat order A, B, ...) [--slowdown S] [--seed N|hidden:I] [--first A] [--tag TAG]
# Each player is opus, fable, astra, luna or oracle (as in run_duel.sh); one player per seat of the task (its task.toml
# [[seats]]: the format's seats in sport.toml, e.g. racing `race` = A-D, 4 cars).
# Job: jobs/<tag>-<sport>-race-x<S>-s<seed, or hI for hidden:I>-<p1>-<p2>-...-<UTC time>.
set -euo pipefail
task=racing-race-wallclock-program-standard players= slowdown= seed=1 first=A tag=race
while [ $# -gt 0 ]; do
  case $1 in
    --task) task=$2; shift 2;; --players) players=$2; shift 2;; --slowdown) slowdown=$2; shift 2;; --seed) seed=$2; shift 2;;
    --first) first=$2; shift 2;; --tag) tag=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
IFS=, read -r -a P <<< "$players"
n=${#P[@]}
[ "$n" -ge 1 ] || { sed -n 2,4p "$0" >&2; exit 2; }
source "$(dirname "$0")/v02_run.sh"     # task paths and per-seat accounts of v0.2 tasks
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
tdir=$(task_dir "$task"); task=$(basename "$tdir")
[ -d "$tdir" ] || { echo "no task $tdir" >&2; exit 2; }
read -r -a SEATS <<< "$(task_seats "$tdir")"
[ "$n" -eq "${#SEATS[@]}" ] || { echo "$task has ${#SEATS[@]} seats (${SEATS[*]}): give one player per seat" >&2; exit 2; }
export PYTHONPATH=$repo     # players/harnesses/ (the metering adapters; harbor_agents/ keeps shims)
HARBOR=${HARBOR:-harbor}
harbor_guard     # the pinned multi-seat fork, or stop: upstream harbor would run the agent with internet (v02_run.sh)
run_seed DUEL_SEED "$seed" "$tdir"     # a number, or hidden:I (v02_run.sh)
export FIRST_PLAYER=$first THINK_OPENAI_ROUTE=azure
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
i=0
for s in "${SEATS[@]}"; do
  p=${P[$i]}
  export THINK_OPENAI_ROUTE_$s=$(route $p) PLAYER_MODE_$s=$(mode $p)
  seat_account $s $(account $p)
  seat_model $s $p     # the seat's model lock THINK_MODEL_<seat> (players.toml; v02_run.sh)
  seat_codex $s $p     # no ChatGPT subscription inherited from the shell reaches this seat (v02_run.sh)
  args="$args $(seat $s $p)"
  i=$((i + 1))
done
job=$tag-${task%%-*}-race${slowdown:+-x$slowdown}-s$seed_tag-$(echo "$players" | tr , -)-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p jobs
# shellcheck disable=SC2086
snapshot_watch $job     # program snapshots (v02_run.sh; RO_SNAPSHOTS=0: off)
$HARBOR run -p "$tdir" $args --env-file .env -n 1 -k 1 -r 0 -o jobs --job-name $job > jobs/$job.log 2>&1
r=$(ls jobs/$job/*/verifier/reward.json 2>/dev/null | head -1 || true)
echo "$job: $( [ -n "$r" ] && python3 -c "import json,sys;print(json.load(open(sys.argv[1])))" "$r" || echo NO-REWARD)"
