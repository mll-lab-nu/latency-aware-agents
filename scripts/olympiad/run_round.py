#!/usr/bin/env python3
"""One round of a sport's Olympiad experiments (one seed for everything), scheduled over model-account lanes.

    python3 scripts/olympiad/run_round.py <round 1|2|3> [--sport tabletennis] [--results DIR] [--dry] [--missing]
                                          [--lanes opus,fable,astra,luna] [--slowdown S] [--kind duel] [--level L]
                                          [--players opus,fable,astra,luna] [--action-level high|low]

The plan comes from the sport's declarations (plan.py: sports/<sport>/sport.toml [[tasks]], [formats], [official],
[timing], and the generated tasks/<id>/, whose verifier decides which seeds are official). Round r uses, for every
format and level, the r-th official seed in sport.toml order. Matches:
  * single player (a one-seat format: against the scripted opponent, or a solo score): every level x every player;
  * duels: every pair of players, both seat orders (the first player/receiver is seat A);
  * team matches (a format of two sides with several seats each, e.g. soccer `team`): every pair, both orders, two
    seats per model;
  * multi-car races (a format of more than two sides, e.g. racing `race`): one race per round with one car per player
    of --race-players (default --players), in that seat order; only when there is one player per seat of the task.
The task of a format is the sport's first [[tasks]] entry of that kind (or of --action-level). Real-time sports run at
each task's own default slowdown ([[tasks]] slowdown, else [timing] slowdown) unless --slowdown is given; a slowdown
at which no seed is official stops the plan. Turn-based sports have none.
Players: --players (default: the model players of the dispatcher's lanes, accounts.json, in players/players.toml
order). Built-in players (players.toml harness = builtin: oracle, nop) play on lane `builtin`, without an account.
Lanes and accounts: dispatch.py with accounts.json (a lane per limiting account, `workers` matches at once per host;
Claude accounts are probed for headroom before each match). Run the lanes of one round on several hosts by splitting
--lanes. A match that ends as a technical failure (or without a
result) is run once more. One line per finished match in <results>/<sport>-round<r>.jsonl (outside the repository:
PROTOCOL D3). --missing runs only the matches without a valid result in that file yet. A match whose model calls were
refused with HTTP 429 (an account limit) waits 30 minutes and is run again without counting the attempt (at most
8 hours). Needs HARBOR (the multi-seat Harbor) in the environment or on PATH.
"""
import argparse
import ast
import itertools
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "players")]
import dispatch  # noqa: E402
import plan  # noqa: E402  (sport.toml + tasks/: the one source of tasks, levels, seeds, slowdowns)
import players as player_list  # noqa: E402  (players/players.toml)

LOCK = threading.Lock()
DISPATCH = dispatch.Dispatcher(log=lambda *a, **k: _log(*a, **k))
ROSTER = player_list.load()
BUILTIN = {p["id"] for p in ROSTER if p.get("harness") == "builtin"}
_LANED = {p for v in DISPATCH.cfg["lanes"].values() for p in v["players"]}
PLAYERS = [p["id"] for p in ROSTER if p["id"] in _LANED]          # the default players: those with an account lane
LANES = {**DISPATCH.lanes(), "builtin": 2}                         # built-in players: no account, 2 matches at once


def _log(*a, **k):
    with LOCK:
        print(*a, **k)


def builtin(p):
    """A built-in player (players/players.toml harness = builtin: oracle, nop) needs no model account."""
    return p in BUILTIN


def lane_of(players):
    """The lane of a match: the account lane of its model players (dispatch.py), `builtin` when it has none."""
    models = [p for p in players if not builtin(p)]
    return DISPATCH.lane_of(models) if models else "builtin"


def _pairs(players):
    """Every pair of players in both seat orders (a player against itself once)."""
    out = []
    for x, y in itertools.combinations(players, 2):
        out += [(x, y)] if x == y else [(x, y), (y, x)]
    return out


def matches(r, sport, slowdown=None, players=None, race_players=None, action_level=None, levels=None):
    """Round r of a sport (plan.py: tasks, levels, slowdowns and official seeds from sport.toml and the tasks).
    slowdown None = each task's default (real-time sports; turn-based sports have none)."""
    players = list(players or PLAYERS)
    if sport not in plan.sports():
        raise ValueError(f"{sport}: not a published sport (sports/*/sport.toml)")
    out = []

    def common(t):
        slow = (slowdown or t["slowdown"]) if t["slowdown"] is not None else None
        return slow, (["--slowdown", f"{slow:g}"] if slow is not None else [])

    t = plan.task(sport, "single", action_level)
    if t and not plan.preview(t):
        slow, sd = common(t)
        for level in t["levels"]:
            if levels and level not in levels:
                continue
            seed = plan.round_seed(t, r, level, slow)
            lv = ["--level", str(level)] if level is not None else []
            for p in players:
                out.append(dict(sport=sport, kind="single", task=t["id"], player=p, level=level, seed=seed,
                                slowdown=slow, lane=lane_of([p]),
                                cmd=["scripts/olympiad/run_single.sh", "--task", t["id"], "--agent", p, *lv, *sd,
                                     "--seed", str(seed), "--tag", f"r{r}"]))
    for kind, script in (("duel", "run_duel.sh"), ("team", "run_team.sh")):
        t = plan.task(sport, kind, action_level)     # team: one agent per robot, two seats per model (soccer)
        if not t or plan.preview(t):             # a preview format of a frozen sport: not scheduled
            continue
        slow, sd = common(t)
        seed = plan.round_seed(t, r, None, slow)
        for a, b in _pairs(players):
            out.append(dict(sport=sport, kind=kind, task=t["id"], a=a, b=b, seed=seed, slowdown=slow,
                            lane=lane_of([a, b]),
                            cmd=[f"scripts/olympiad/{script}", "--task", t["id"], "--a", a, "--b", b, *sd,
                                 "--seed", str(seed), "--first", "A", "--tag", f"r{r}"]))
    t = plan.task(sport, "race", action_level)       # one agent per car, in seat order
    rp = list(race_players or players)
    if t and len(rp) == len(t["seats"]) and not plan.preview(t):
        slow, sd = common(t)
        seed = plan.round_seed(t, r, None, slow)
        out.append(dict(sport=sport, kind="race", task=t["id"], players=rp, seed=seed, slowdown=slow,
                        lane=lane_of(rp),
                        cmd=["scripts/olympiad/run_race.sh", "--task", t["id"], "--players", ",".join(rp), *sd,
                             "--seed", str(seed), "--first", "A", "--tag", f"r{r}"]))
    return out


def key(m):
    k = (m["kind"], m.get("player"), m.get("level"), m.get("a"), m.get("b"), m["seed"])
    return k + (tuple(m["players"]),) if m["kind"] == "race" else k


def rate_limited(job):
    """Whether a model call of this job was refused with HTTP 429 (an account limit, not the player's fault)."""
    for f in (ROOT / "jobs" / job).glob("*/artifacts/data/think_calls*.jsonl"):
        for line in open(f):
            if '"status": 429' in line:
                return True
    return False


# ---- accounts: dispatch.py and accounts.json (outside Harbor; PROTOCOL D11) ----------------------------------------
# A match starts only when every Claude player has an account with headroom (dispatch.Dispatcher.claude_env). Without
# --skip-no-quota it waits and probes again every 10 minutes; with it, the match is skipped for a later --missing pass.
SKIP_NO_QUOTA = False        # --skip-no-quota


def run(m, log):
    """Play a match unless another runner on this host is already playing it (a lock file next to the results)."""
    lockdir = pathlib.Path(log).parent / ".running"
    lockdir.mkdir(exist_ok=True)
    lock = lockdir / ("_".join(str(x) for x in (m["sport"], *key(m))) + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(lock.read_text() or 0), 0)
            _log(time.strftime("%H:%M:%S"), "already running elsewhere:", " ".join(m["cmd"][-10:]), "- skipped", flush=True)
            return
        except (ProcessLookupError, ValueError):     # a stale lock: its runner is gone
            lock.unlink(missing_ok=True)
            return run(m, log)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        return _run(m, log)
    finally:
        lock.unlink(missing_ok=True)


def _run(m, log):
    attempt, waits = 0, 0
    while attempt < 2:
        attempt += 1
        players = ([m["player"]] if m["kind"] == "single" else [m["a"], m["a"], m["b"], m["b"]] if m["kind"] == "team"
                   else list(m["players"]) if m["kind"] == "race"
                   else [m["a"], m["b"]])          # a team match holds two seats per model; a race one per listed player
        while True:                   # every Claude player of the match needs an account with headroom
            got = DISPATCH.claude_env(players)
            if got is not None:
                break
            _log(time.strftime("%H:%M:%S"), "no Claude quota for", " ".join(m["cmd"][-10:]),
                 "- skipped" if SKIP_NO_QUOTA else "- waiting", flush=True)
            if SKIP_NO_QUOTA:
                return
            time.sleep(600)
        acc, accounts = got
        t0 = time.time()
        DISPATCH.hold(accounts)
        try:
            p = subprocess.run(m["cmd"], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **acc})
        finally:
            DISPATCH.release(accounts)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        job, _, res = line.partition(": ")
        try:
            reward = ast.literal_eval(res) if res.startswith("{") else None   # the run script prints a dict literal
        except Exception:  # noqa: BLE001
            reward = None
        rec = {k: v for k, v in m.items() if k != "cmd"} | dict(attempt=attempt, job=job, reward=reward,
                                                                 minutes=round((time.time() - t0) / 60, 1))
        with LOCK:
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if reward and reward.get("valid_trial") == 1:
            return
        logf = ROOT / "jobs" / f"{job}.log"
        if job and logf.exists() and "address pools have been fully subnetted" in logf.read_text(errors="ignore") \
                and waits < 48:
            waits += 1                # docker ran out of network address pools (busy shared host): wait, don't count
            attempt -= 1
            time.sleep(300)
            continue
        if job and logf.exists() and "Could not resolve container for compose service" in logf.read_text(errors="ignore") \
                and waits < 48:
            waits += 1                # the environment did not come up (docker race on a busy host): wait, don't count
            attempt -= 1
            time.sleep(120)
            continue
        if job and rate_limited(job) and SKIP_NO_QUOTA:
            return                    # an account limit: a later --missing pass plays it once there is quota
        if job and rate_limited(job) and waits < 16:      # account limit: wait for the reset, don't count the attempt
            waits += 1
            attempt -= 1
            time.sleep(1800)
        # otherwise a technical failure (or no result): once more


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int, choices=(1, 2, 3))
    ap.add_argument("--results", default=str(ROOT.parent / "olympiad_results"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--lanes", default=",".join(LANES), help="run only these lanes")
    ap.add_argument("--sport", default="tabletennis", choices=plan.sports())
    ap.add_argument("--slowdown", type=float, default=None,
                    help="real-time sports only (default: each task's own, sport.toml [[tasks]] / [timing] slowdown)")
    ap.add_argument("--level", type=int, action="append", default=None,
                    help="single player: only this level (repeatable; default every level of the format)")
    ap.add_argument("--action-level", choices=("high", "low"), default=None,
                    help="the tasks of this action level (default: the sport's first task of each format kind)")
    ap.add_argument("--players", default=",".join(PLAYERS), help="the players of this sport (singles and their duels)")
    ap.add_argument("--completed-results", action="append", default=[],
                    help="additional read-only result directories used with --missing (other hosts)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="cap workers per selected lane, for example 1 for a presentation")
    ap.add_argument("--kind", choices=("single", "duel", "team", "race"),
                    help="only singles, only duels (Season 1 plan A), only team matches (soccer, one agent per robot) "
                         "or only multi-car races (racing)")
    ap.add_argument("--race-players", default=None,
                    help="the cars of a multi-car race, one player per seat in order (default --players)")
    ap.add_argument("--skip-no-quota", action="store_true",
                    help="skip matches whose Claude player has no quota now instead of waiting (use with --missing passes)")
    a = ap.parse_args()
    global SKIP_NO_QUOTA
    SKIP_NO_QUOTA = a.skip_no_quota
    if a.concurrency is not None and a.concurrency < 1:
        ap.error("--concurrency must be at least 1")
    ms = [m for m in matches(a.round, a.sport, a.slowdown, a.players.split(","),
                             a.race_players.split(",") if a.race_players else None, a.action_level, a.level)
          if m["lane"] in a.lanes.split(",") and (not a.kind or m["kind"] == a.kind)]
    log = os.path.join(a.results, f"{a.sport}-round{a.round}.jsonl")
    if a.missing:
        done = set()
        for result_dir in [a.results, *a.completed_results]:
            result_file = pathlib.Path(result_dir) / f"{a.sport}-round{a.round}.jsonl"
            if not result_file.exists():
                continue
            for line in result_file.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get("reward") or {}).get("valid_trial") == 1:
                    done.add(key(r))
        ms = [m for m in ms if key(m) not in done]
    if a.dry:
        for m in ms:
            print(m["lane"], " ".join(m["cmd"]))
        print(len(ms), "matches")
        return
    os.makedirs(a.results, exist_ok=True)
    qs = {lane: queue.Queue() for lane in LANES}
    for m in ms:
        qs[m["lane"]].put(m)

    def worker(lane):
        while True:
            try:
                m = qs[lane].get_nowait()
            except queue.Empty:
                return
            run(m, log)
            time.sleep(20)            # stagger starts (image builds, model logins)

    threads = [threading.Thread(target=worker, args=(lane,))
               for lane, n in LANES.items() if lane in a.lanes.split(",")
               for _ in range(min(n, a.concurrency) if a.concurrency else n)]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(15)
    for t in threads:
        t.join()
    print(f"round {a.round} done: {log}")


if __name__ == "__main__":
    main()
