#!/usr/bin/env python3
"""Live run of a generated task with built-in players (docs/framework_zh.md §8 step ④, "实跑"): the task's own sim
directory (the core's launcher: metering proxies + match service), its solution/ (the reference player over HTTP)
on every `oracle` seat, nothing on a `nop` seat, then the session end (SIGUSR1, as Harbor's verifier.collect) and
the task's own grader (tests/) -> reward.json. No Harbor, no model, no network: one container of the sport's image.

    python3 tools/live.py TASK_DIR OUT_DIR oracle|nop [oracle|nop ...] [--env K=V ...] [--skill SEAT=DIR] [--harbor]

A low task with the skill in the loop (sport.toml [modes.low.inloop]): each seat's skill host runs as its own process
in the same container (skilleval's skillhost.py from environment/skill/, the same Unix-socket protocol as the task's
skill containers; the toolkit at the seat's paths; the evaluation image, [modes.low] eval_pip), on --skill SEAT=DIR or,
for an oracle seat, the task's solution/skill (our reference skill); every call lands in OUT_DIR/data/inloop_<seat>.jsonl.
The isolation of the real task (no network, read-only, nobody, no simulator files) is the compose file's, not this.

--harbor runs the same match through Harbor instead (`harbor run` with the built-in agents oracle / nop per seat;
needs Harbor with multi-seat support, see README; $HARBOR names the binary). Prints reward.json; exit 0 when it was written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402
import sportlib  # noqa: E402

INNER = r"""
set -u
cd /srv
for s in ${SKILL_SEATS:-}; do                    # the skill in the loop: one skill host per seat (skillhost.py)
  mkdir -p /tmp/skillipc/$s
  python3 /opt/skilleval/skillhost.py serve --socket /tmp/skillipc/$s/skill.sock --skill /skill_$s > /data/skillhost_$s.log 2>&1 &
done
python3 /srv/core/launch.py > /data/server.log 2>&1 &
LP=$!
for i in $(seq 1 120); do [ -f /data/server.pid ] && break; sleep 1; done
pids=""
for s in $SEATS_ORACLE; do
  (cd /solution && SEAT=$s SIM_URL=http://127.0.0.1:8000 timeout ${PLAYER_TIMEOUT_S:-3000} python3 oracle_play.py > /data/player_$s.log 2>&1) &
  pids="$pids $!"
done
for p in $pids; do wait $p; done
[ -n "$SEATS_ORACLE" ] || sleep ${NOP_WAIT_S:-5}
kill -USR1 $(cat /data/server.pid)
for i in $(seq 1 $RO_REPLAY_TIMEOUT_S); do grep -q '"graded_ready": true' /data/final.json 2>/dev/null && break; sleep 1; done
[ "$RCB_RENDER_VIDEO" = 1 ] && for i in $(seq 1 $RCB_VIDEO_TIMEOUT_S); do [ -f /data/video.json ] && break; sleep 1; done
kill $LP 2>/dev/null; sleep 1
mkdir -p /data/verifier
cd /tests && RCB_DATA=/data RCB_LOG=/data/verifier python3 /tests/grade.py > /data/grade.log 2>&1
"""


def resolve(v: str, over: dict) -> str:
    """${VAR:-default} (nested) with the overrides, else the default."""
    while True:
        m = re.search(r"\$\{([A-Z0-9_]+):-([^${}]*)\}", v)
        if not m:
            return v
        v = v[:m.start()] + over.get(m.group(1), m.group(2)) + v[m.end():]


def task_entry(task: Path):
    meta = tomllib.loads((task / "task.toml").read_text())["metadata"]
    fam = meta["family"]                 # "<sport>_<format>"; a format name may itself hold "_" (soccer match_overhead)
    for sport in sorted(sportlib.names(), key=len, reverse=True):
        if fam.startswith(sport + "_"):
            for tid, t in generate.tasks_of(sport):
                if tid == task.name:
                    return sport, t
    raise SystemExit(f"{task.name} is not a task of any sports/*/sport.toml (family {fam})")


def run_offline(task: Path, out: Path, players: list[str], over: dict, timeout: float | None = None,
                skills: dict | None = None) -> dict:
    sport, t = task_entry(task)
    d = sportlib.decl(sport)
    # a long realtime match (e.g. x20 foosball, ~4200 s) needs more than the default: sport.toml [check]
    # live_player_timeout_s, else $PLAYER_TIMEOUT_S, else 3000; the container also gets the inner script's other waits:
    # the sim's start (120 s), the replay and the video (the sim's limits, sportlib.post_match_timeouts)
    player_timeout = float((d.get("check") or {}).get("live_player_timeout_s") or os.environ.get("PLAYER_TIMEOUT_S") or 3000)
    replay, video = sportlib.post_match_timeouts(d)
    timeout = timeout or player_timeout + 120 + replay + video
    seats = d["formats"][t["format"]]["seats"]
    if len(players) != len(seats):
        raise SystemExit(f"{len(seats)} seat(s): give one player per seat")
    over = {**{f"PLAYER_MODE_{s}": "oracle" for s in seats}, **over}
    env = {k: resolve(v, over) for k, v in generate.sim_env(d, t).items()}
    env.update({k: v for k, v in over.items() if not k.startswith("PLAYER_MODE_")})
    env["PLAYER_TIMEOUT_S"] = str(int(player_timeout))
    env.update({"RCB_RENDER_VIDEO": over.get("RCB_RENDER_VIDEO", "0"), "RCB_BROADCAST": over.get("RCB_BROADCAST", "0"),
                "HOME": "/tmp", "SEAT_HOSTS": "",   # localhost seats
                "SEATS_ORACLE": " ".join(s for s, p in zip(seats, players) if p == "oracle")})
    if all(p == "nop" for p in players):
        env.setdefault("START_TIMEOUT_S", "20")
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for f in data.iterdir():
        if f.is_file():
            f.unlink()
    (out / "inner.sh").write_text(INNER)
    name = f"ro-live-{task.name}-{os.getpid()}"
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", os.environ.get("RO_LIVE_CPUS", "4"),
           "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{task / 'environment' / 'sim'}:/srv:ro",
           "-v", f"{task / 'solution'}:/solution:ro", "-v", f"{task / 'tests'}:/tests:ro", "-v", f"{data}:/data",
           "-v", f"{out / 'inner.sh'}:/inner.sh:ro"]
    image = d["sport"]["image"]
    if generate.inloop(d, t) is not None:            # the skill in the loop: the skill hosts in this container
        import skill_eval
        ctx = task / "environment" / "skill"
        cmd += ["-v", f"{ctx / 'skilleval'}:/opt/skilleval:ro"]
        for top in sorted((ctx / "files").iterdir()) if (ctx / "files").is_dir() else []:
            for sub in sorted(top.iterdir()):        # the toolkit at the seat's paths (e.g. /opt/robot)
                cmd += ["-v", f"{sub}:/{top.name}/{sub.name}:ro"]
        for s, p in zip(seats, players):
            src = (skills or {}).get(s) or (task / "solution" / "skill" if p == "oracle" else None)
            if src is None:                          # a nop seat without a skill: an empty directory (no hit.py)
                src = out / f"no_skill_{s}"
                src.mkdir(parents=True, exist_ok=True)
            cmd += ["-v", f"{Path(src).resolve()}:/skill_{s}:ro"]
            env[f"RCB_SKILL_INLOOP_{s}"] = f"unix:/tmp/skillipc/{s}/skill.sock"
        env["RCB_SKILL_INLOOP_LOG"] = "/data"
        env["SKILL_SEATS"] = " ".join(seats)
        image = skill_eval.eval_image(sport, d["modes"]["low"])
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image, "bash", "/inner.sh"]
    try:
        subprocess.run(cmd, check=False, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise
    reward = data / "verifier" / "reward.json"
    return json.loads(reward.read_text()) if reward.exists() else {"error": "no reward.json", "dir": str(data)}


def run_harbor(task: Path, out: Path, players: list[str], over: dict) -> dict:
    """The same match through Harbor (multi-seat fork, $HARBOR or `harbor`): built-in agents oracle / nop per seat."""
    sport, t = task_entry(task)
    seats = sportlib.decl(sport)["formats"][t["format"]]["seats"]
    env = {**os.environ, "RCB_PLAYER_MODE": "oracle", "RCB_RENDER_VIDEO": "0", **over,
           **{f"PLAYER_MODE_{s}": "oracle" for s in seats}}
    job = f"live-{task.name}-{'-'.join(players)}-{time.strftime('%Y%m%dT%H%M%S')}"
    cmd = [os.environ.get("HARBOR", "harbor"), "run", "-p", str(task)]
    for s, p in zip(seats, players):
        cmd += ["--seat", f"{s}={p}"]
    cmd += ["-k", "1", "-n", "1", "-r", "0", "-o", str(out / "jobs"), "--job-name", job]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (out / "harbor.log").write_text(r.stdout + r.stderr)
    found = sorted((out / "jobs" / job).glob("*/verifier/reward.json"))
    return json.loads(found[0].read_text()) if found else {"error": "no reward.json", "log": str(out / "harbor.log")}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("players", nargs="+", choices=("oracle", "nop"))
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--skill", action="append", default=[], help="SEAT=DIR: the seat's skill (skill in the loop)")
    ap.add_argument("--harbor", action="store_true")
    a = ap.parse_args(argv)
    task, out = Path(a.task).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    over = dict(kv.split("=", 1) for kv in a.env)
    skills = dict(kv.split("=", 1) for kv in a.skill)
    r = run_harbor(task, out, a.players, over) if a.harbor else run_offline(task, out, a.players, over, skills=skills)
    print(json.dumps(r))
    return 0 if "error" not in r else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
