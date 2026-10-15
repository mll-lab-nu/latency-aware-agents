# Sourced by run_single.sh, run_duel.sh, run_team.sh, run_race.sh: what a v0.2 task (tools/generate.py; the sim runs
# core/launch.py) needs beyond a legacy one.
#   task_dir NAME|PATH     tasks/NAME, or a task directory given as a path (e.g. build/tasks/<id> of an unpublished sport)
#   task_seats DIR         the task's seats (task.toml [[seats]] names, generated from the format's seats), space-separated
#   seat_account SEAT N    the Claude subscription of a seat: CLAUDE_ACCOUNT_<seat>=N for the legacy launchers, and for
#                          the v0.2 launcher RO_SEAT_SELECT_<seat> ("VAR=SOURCE": that seat's metering proxy takes
#                          CLAUDE_CODE_OAUTH_TOKEN from CLAUDE_CODE_OAUTH_TOKEN_2 when N = 2)
#   seat_model SEAT PLAYER   the seat's model lock THINK_MODEL_<seat> (its metering proxy refuses every other model) from
#                          players/players.toml (`players.py think-model`); empty for the built-in players oracle and
#                          nop. Every run script sets it for every seat it starts (user decision 2026-10-08: explicit
#                          lock; the proxy's first-model lock is only the safety net); a model player without one stops
#                          the run before Harbor starts.
#   run_seed VAR SEED TASK_DIR [LEVEL]  the match seed: a number, or hidden:I = entry I of the cell (format x level)
#                          of the sport's registered hidden official seed set (core 0.2.14), resolved on this host from
#                          the private set (tools/seeds.py resolve; RO_SEEDS_DIR): VAR = the seed, RCB_SEED_SALT = its
#                          salt and RCB_SEED_INDEX = I, which only the sim service gets; no record shows the seed.
#                          Sets seed_tag, the job name's seed part: the seed, or hI (a hidden seed never names a job)
#   seat_codex SEAT PLAYER the ChatGPT subscription of a seat (players.toml *-sub players: astra-sub, sol-sub, luna-sub;
#                          account_scope = "seat"): CODEX_ACCESS_TOKEN / CODEX_ACCOUNT_ID are read from $CODEX_ENV_FILE
#                          (default .env) and exported as CODEX_ACCESS_TOKEN_<seat> / CODEX_ACCOUNT_ID_<seat> only (the
#                          compose file passes them to that seat's metering proxy, never unsuffixed), with the seat's
#                          pinned backend model THINK_CHATGPT_MODEL_<seat> (`players.py chatgpt-model`, sol-sub:
#                          gpt-6.1-sol) and the route THINK_OPENAI_ROUTE_<seat>=chatgpt: the seat's proxy forwards the
#                          ChatGPT backend only and core/launch.py gives it no task-wide credential (no OpenRouter key);
#                          for any other player the three CODEX_* / THINK_CHATGPT_MODEL variables are cleared, so a duel's
#                          other seat never gets them.
#   harbor_guard           stop the run (exit 2) unless $HARBOR is the pinned multi-seat Harbor fork
#                          CHIGUI0/harbor@$RO_HARBOR_PIN. Upstream PyPI harbor 0.23.0 has no --seat, task.toml
#                          [[seats]] or per-seat setup_networks: it would run the agent in the `main` service on the
#                          default network, with internet. Checked: the installed `harbor` distribution of $HARBOR's
#                          interpreter (its #! line) records the pinned commit (pip/uv direct_url.json: a git install at
#                          the commit, or an editable install of a local checkout whose HEAD is the commit with no
#                          changes to tracked files; a non-editable install from a directory is refused: its code is
#                          a copy the directory's HEAD does not describe),
#                          and `$HARBOR run --help` lists --seat. Every path needs it (oracle and nop run as --seat too).
#   snapshot_watch JOB     program snapshots of the job's seats in the background (tools/snapshot_watch.py, core 0.2.25
#                          diagnostics): the seats' program files whenever one changes and at every ready, and their
#                          processes (command line + cwd), into jobs/JOB/<trial>/snapshots/<seat>/; it only reads the
#                          containers and stops with the trial. RO_SNAPSHOTS=0: off
task_dir() { case $1 in */*) echo "${1%/}";; *) echo "tasks/$1";; esac; }
task_seats() { sed -n '/^\[\[seats\]\]/,/^name/s/^name = "\(.*\)"$/\1/p' "$1/task.toml" | tr '\n' ' '; }
seat_account() {
  export "CLAUDE_ACCOUNT_$1=$2"
  if [ "$2" = 2 ]; then export "RO_SEAT_SELECT_$1=CLAUDE_CODE_OAUTH_TOKEN=CLAUDE_CODE_OAUTH_TOKEN_2"; else export "RO_SEAT_SELECT_$1="; fi
}
_v02_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RO_HARBOR_PIN=3fc74678123a2eb1845938b099e9d7856f2c6f6d     # docs/benchmarks/olympiad.md, README.md "Requirements"
harbor_guard() {
  local exe line py why help
  local fix="install it: uv tool install git+https://github.com/CHIGUI0/harbor@$RO_HARBOR_PIN (or set HARBOR=<that harbor>)"
  exe=$(command -v -- "${HARBOR:-harbor}") || { echo "harbor: '${HARBOR:-harbor}' not found; $fix" >&2; exit 2; }
  IFS= read -r line < "$exe" || line=
  read -r -a py <<< "${line#\#!}"
  case $line in
    '#!'*) [ "${py[0]##*/}" != env ] || py=("${py[@]:1}") ;;
    *) py=() ;;
  esac
  [ ${#py[@]} -gt 0 ] || { echo "harbor: $exe is not a Python entry point (no #! line); cannot check it is the pinned fork; $fix" >&2; exit 2; }
  why=$("${py[@]}" -I - "$RO_HARBOR_PIN" <<'PY' 2>&1
import importlib.metadata as md, json, subprocess, sys, urllib.parse, urllib.request
pin = sys.argv[1]
try:
    dist = md.distribution("harbor")
except md.PackageNotFoundError:
    sys.exit("no harbor distribution in " + sys.executable)
d = json.loads(dist.read_text("direct_url.json") or "{}")
url, commit = d.get("url", ""), d.get("vcs_info", {}).get("commit_id", "")
if not commit and "dir_info" in d and url.startswith("file:"):     # a local checkout: editable only (pip install -e <dir>)
    path = urllib.request.url2pathname(urllib.parse.urlparse(url).path)
    if not d["dir_info"].get("editable"):    # a copy made at install time: the directory's HEAD says nothing about it
        sys.exit(f"harbor {dist.version} installed (not editable) from the directory {path}: install the fork from "
                 "its git URL, or editable (-e) from a checkout at the pin")
    git = lambda *a: subprocess.run(["git", "-C", path, *a], capture_output=True, text=True)
    commit = git("rev-parse", "HEAD").stdout.strip()
    if commit == pin and git("diff", "--quiet", "HEAD").returncode != 0:
        sys.exit(f"harbor {dist.version} from {path}: HEAD is the pin but tracked files are changed")
if commit != pin:
    sys.exit(f"harbor {dist.version} from {url or 'PyPI'} at {commit or 'no recorded commit'}")
PY
) || { echo "harbor: $exe is not the pinned multi-seat fork CHIGUI0/harbor@${RO_HARBOR_PIN:0:7} ($why)." \
            "Upstream harbor has no --seat / [[seats]] / setup_networks and would run the agent with internet; $fix" >&2; exit 2; }
  help=$("$exe" run --help 2>&1) || true
  case $help in *--seat*) ;; *) echo "harbor: $exe run --help lists no --seat (not the multi-seat fork); $fix" >&2; exit 2;; esac
}
seat_model() {
  local m
  m=$(python3 "$_v02_root/players/players.py" think-model "$2") || { echo "seat $1: no model lock for player $2 (players/players.toml)" >&2; exit 2; }
  case $2 in
    oracle|nop) m= ;;
    *) [ -n "$m" ] || { echo "seat $1: player $2 plays a model but has no model lock" >&2; exit 2; } ;;
  esac
  export "THINK_MODEL_$1=$m"
}
run_seed() {
  local out
  case $2 in
    hidden:*)
      out=$(python3 "$_v02_root/tools/seeds.py" resolve "$3" "${2#hidden:}" ${4:+--level $4}) \
        || { echo "seed $2 of $3${4:+ at level $4}: not resolved" >&2; exit 2; }
      export "$1=${out%% *}" "RCB_SEED_SALT=${out#* }" "RCB_SEED_INDEX=${2#hidden:}"; seed_tag=h${2#hidden:} ;;
    *) export "$1=$2"; unset RCB_SEED_SALT RCB_SEED_INDEX; seed_tag=$2 ;;
  esac
}
seat_codex() {
  local f=${CODEX_ENV_FILE:-.env} v vals=()
  case $2 in *-sub) ;; *) export "CODEX_ACCESS_TOKEN_$1=" "CODEX_ACCOUNT_ID_$1=" "THINK_CHATGPT_MODEL_$1="; return;; esac
  v=$(python3 "$_v02_root/players/players.py" chatgpt-model "$2") || { echo "seat $1: no player $2 (players.toml)" >&2; exit 2; }
  export "THINK_CHATGPT_MODEL_$1=$v" "THINK_OPENAI_ROUTE_$1=chatgpt"     # ChatGPT backend only, every run script
  [ -r "$f" ] || { echo "$2 needs $f (CODEX_ACCESS_TOKEN, CODEX_ACCOUNT_ID)" >&2; exit 2; }
  while IFS= read -r -d '' v; do vals+=("$v"); done < <(set -a; . "$f"; printf '%s\0' "${CODEX_ACCESS_TOKEN:-}" "${CODEX_ACCOUNT_ID:-}")
  [ -n "${vals[0]}" ] && [ -n "${vals[1]}" ] || { echo "$f lacks CODEX_ACCESS_TOKEN or CODEX_ACCOUNT_ID ($2)" >&2; exit 2; }
  export "CODEX_ACCESS_TOKEN_$1=${vals[0]}" "CODEX_ACCOUNT_ID_$1=${vals[1]}"
}
snapshot_watch() {
  [ "${RO_SNAPSHOTS:-1}" = 0 ] && return 0
  python3 "$_v02_root/tools/snapshot_watch.py" --job-dir "jobs/$1" --runner-pid $$ > "jobs/$1.snapshots.log" 2>&1 &
}
