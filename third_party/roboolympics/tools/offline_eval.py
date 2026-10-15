#!/usr/bin/env python3
"""Offline evaluation of a program snapshot (C2, capability measurement framework §3, §7; from the C2 pilot's
scripts/c2_offline/): rerun ONE frozen version of a player's program, never a model, on chosen seeds against the
task's scripted level (format match) or a fixed opponent (format duel: the built-in reference or another snapshot),
and write the per-match and pooled results. Use held-out seeds (never the hidden official ones).

    python3 tools/offline_eval.py --task TASK_DIR --snapshot SNAP --seeds 121,131 [--level L] --out DIR
        [--cmd "python3 play.py 0.42"] [--cwd /app/output] [--slowdown S] [--repeat N]
        [--opponent builtin|SNAP2 [--opponent-cmd ...]] [--relaunch review|never] [--player-image IMG]
        [--timeout-s T] [--diagnose]

SNAP: a snapshot of tools/snapshot_watch.py (snap_NNN.tar.gz, or its seat directory + --n NNN; the files relative to /:
app/..., tmp/...) or a plain directory that becomes /app/output. Without --cmd the program's command line and working
directory are taken from the seat's procs.jsonl (the last process set recorded at or before the snapshot): the newest
process whose working directory is under /app or /tmp and that runs a python / shell script (not a model harness).

Per match: a private internal network, the task's sim (environment/sim read-only, the compose environment with every
seat in `oracle` mode: no model meter, the program acts by itself; video off) and one player container per seat (the
task's player image, environment/Dockerfile, built once as ro-offline-<task>:<hash>), each running a small driver
(stdlib): launch the program -> `start` -> at every review: stop it, relaunch it, `ready` with a fixed WHY/PLAN (what
the models did; --relaunch never: keep it running) -> relaunch once if it exited before the world resumed -> stop at
the end. Then the session end (SIGUSR1, as Harbor's verifier.collect: final.json with the replay check).

DIR/s<seed>[_r<i>]/data/       episode.jsonl, final.json, ... (python -m core.diagnose runs on it)
DIR/s<seed>[_r<i>]/run.json    command, seed, containers, driver log tail, timings, CPU seconds per container
DIR/metrics.json               per match: valid, reason, scores, won, score share, the core's act linkage and C1
                               description (core/diagnose.py), with --diagnose the sport's metrics; pooled: matches,
                               valid, wins, win rate, mean score share (own / (own + opponent))
Needs docker and python >= 3.11 (tomllib).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402
import sportlib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HARNESS = re.compile(r"(claude|codex|node|npm|/usr/local/bin/game|think_proxy|core\.serve|core/launch|sleep |tail |"
                     r"inner\.sh)")
PROGRAM = re.compile(r"^(\S*/)?(python3?(\.\d+)?|bash|sh)\s+(-u\s+)?\S+\.(py|sh)\b")

DRIVER = r'''
import json, os, resource, shlex, signal, subprocess, sys, time, urllib.error, urllib.request
BASE = os.environ["SIM_URL"].rstrip("/") + "/" + os.environ["SEAT"] + "/"
CMD, CWD, RELAUNCH, TIMEOUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
LOG = open(os.environ.get("DRIVER_LOG", "/tmp/offline_driver.jsonl"), "a", buffering=1)
def log(ev, **kw): LOG.write(json.dumps({"t": round(time.time(), 3), "ev": ev, **kw}) + "\n")
def call(ep, body=None, timeout=30):
    req = urllib.request.Request(BASE + ep, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try: return e.code, json.load(e)
        except Exception: return e.code, {}
    except Exception as e:
        return 0, {"error": repr(e)}
class Program:
    def __init__(self): self.p, self.n = None, 0
    def start(self):
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        out = open(f"/tmp/offline_program_{self.n:02d}.out", "w")
        self.p = subprocess.Popen(shlex.split(CMD), cwd=CWD, env=env, stdout=out, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        self.n += 1
        return self.p.pid
    def alive(self): return self.p is not None and self.p.poll() is None
    def stop(self):
        if self.p is None: return None
        for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
            if self.p.poll() is not None: break
            try: os.killpg(self.p.pid, sig)
            except ProcessLookupError: break
            t0 = time.time()
            while self.p.poll() is None and time.time() - t0 < wait: time.sleep(0.05)
        return self.p.returncode
def state(st): return st.get("state", st) if isinstance(st, dict) else {}
t0 = time.time()
while call("status")[0] != 200:
    if time.time() - t0 > 300: log("no_service"); sys.exit(2)
    time.sleep(1)
prog = Program()
log("launch", pid=prog.start(), why="initial")
time.sleep(1.0)
code, _ = call("start", {}, timeout=600)
log("start", code=code)
checked = False
while time.time() - t0 < TIMEOUT:
    code, st = call("status")
    st = state(st)
    if code != 200:
        time.sleep(1); continue
    phase = st.get("phase")
    if st.get("done") or phase == "over":
        rc = prog.stop(); ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        log("over", rc=rc, reason=st.get("reason"), program_cpu_s=round(ru.ru_utime + ru.ru_stime, 1)); sys.exit(0)
    if phase == "review" and not (st.get("ready") or {}).get(os.environ["SEAT"]):
        if RELAUNCH == "review":
            rc = prog.stop(); log("review", step=st.get("step"), prev_rc=rc)
            log("launch", pid=prog.start(), why="review"); time.sleep(1.0)
        code, r = call("ready", {"why": "offline replay of a frozen program version",
                                 "plan": "same program, relaunched by the offline driver"}, timeout=600)
        log("ready", code=code, err=(r or {}).get("error")); checked = False; continue
    if phase == "running" and not checked:
        time.sleep(1.0); checked = True
        if not prog.alive():
            log("launch", pid=prog.start(), why="exited_before_unit", prev_rc=prog.p.returncode if prog.p else None)
        continue
    time.sleep(0.5)
log("timeout", rc=prog.stop()); sys.exit(3)
'''


def sh(args, check=False, timeout=None):
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=timeout)


def task_info(task: Path):
    meta = tomllib.loads((task / "task.toml").read_text())["metadata"]
    sport = meta["family"].rsplit("_", 1)[0]
    for tid, t in generate.tasks_of(sport):
        if tid == task.name:
            return sport, sportlib.decl(sport), t
    raise SystemExit(f"{task.name} is not a task of sports/{sport}/sport.toml")


def player_image(task: Path, given: str | None) -> str:
    if given:
        return given
    ctx = task / "environment"
    h = hashlib.sha256()
    for p in sorted(x for x in ctx.iterdir() if x.is_file()):
        h.update(p.name.encode() + p.read_bytes())
    tag = f"ro-offline-{task.name}:{h.hexdigest()[:12]}"
    if sh(["docker", "image", "inspect", tag]).returncode != 0:
        r = sh(["docker", "build", "-q", "-t", tag, "-f", str(ctx / "Dockerfile"), str(ctx)], timeout=3600)
        if r.returncode != 0:
            raise SystemExit(f"building the player image failed:\n{r.stderr[-2000:]}")
    return tag


def snapshot_files(snap: str, n: int | None, dest: Path) -> dict:
    """Unpack a snapshot into dest/ (dest/app, dest/tmp); {"tar", "ts", "procs"} of its record when known."""
    p = Path(snap)
    info: dict = {"tar": None, "ts": None, "procs": None}
    if p.is_dir() and (p / "snapshots.jsonl").exists():
        rows = [json.loads(x) for x in (p / "snapshots.jsonl").read_text().splitlines() if x.strip()]
        saved = [r for r in rows if r.get("tar")]
        row = next((r for r in saved if r["n"] == n), None) if n else (saved[-1] if saved else None)
        if row is None:
            raise SystemExit(f"{p}: no saved snapshot{' ' + str(n) if n else ''}")
        p = p / row["tar"]
        info["ts"] = row["ts"]
    if p.is_file():
        info["tar"] = str(p)
        if info["ts"] is None:
            rows = [json.loads(x) for x in (p.parent / "snapshots.jsonl").read_text().splitlines()
                    if x.strip()] if (p.parent / "snapshots.jsonl").exists() else []
            info["ts"] = next((r["ts"] for r in rows if r.get("tar") == p.name), None)
        with tarfile.open(p) as tf:
            tf.extractall(dest, filter="data")
        procs_file = p.parent / "procs.jsonl"
        if procs_file.exists():
            recs = [json.loads(x) for x in procs_file.read_text().splitlines() if x.strip()]
            before = [r for r in recs if info["ts"] is None or r["ts"] <= info["ts"] + 1]
            info["procs"] = (before or recs or [None])[-1]
    else:                                        # a plain directory: /app/output
        shutil.copytree(p, dest / "app" / "output", ignore=shutil.ignore_patterns("__pycache__", "*.pid"))
    for x in dest.rglob("*.pid"):
        x.unlink()
    return info


def pick_program(procs_rec) -> tuple[str, str] | None:
    """(command line, cwd) of the player's program in a procs.jsonl record (the newest matching process)."""
    if not procs_rec:
        return None
    cand = [p for p in procs_rec.get("procs", []) if p.get("cwd", "").startswith(("/app", "/tmp"))
            and PROGRAM.match(p.get("args", "")) and not HARNESS.search(p.get("args", ""))]
    if not cand:
        return None
    p = max(cand, key=lambda x: x["pid"])
    return p["args"].strip(), p["cwd"]


def run_match(task, sport, d, t, a, seed, out: Path, image, seats_prog: dict) -> dict:
    """One offline match; seats_prog: seat -> {"root": unpacked snapshot dir or None, "cmd", "cwd", "solution"}."""
    (out / "data").mkdir(parents=True, exist_ok=True)
    for f in (out / "data").iterdir():
        if f.is_file():
            f.unlink()
    over = {"RCB_SEED": str(seed), "DUEL_SEED": str(seed), "RCB_RENDER_VIDEO": "0", "RCB_BROADCAST": "0",
            **({"RCB_LEVEL": str(a.level)} if a.level is not None else {}),
            **({"RCB_SLOWDOWN": str(a.slowdown)} if a.slowdown else {}),
            **({"FIRST_PLAYER": a.first} if a.first else {})}
    env = {k: _resolve(v, over) for k, v in generate.sim_env(d, t).items()}
    seats = d["formats"][t["format"]]["seats"]
    env.update({f"PLAYER_MODE_{s}": "oracle" for s in seats})
    env["HOME"] = "/tmp"
    hosts = dict(x.split("=", 1) for x in env["SEAT_HOSTS"].split(","))
    tag = f"ro-offline-{os.getpid()}-{int(time.time() * 1000) % 10 ** 8}"
    net, sim = f"{tag}-net", f"{tag}-sim"
    uid = f"{os.getuid()}:{os.getgid()}"
    run = {"seed": seed, "level": a.level, "slowdown": env.get("RO_SLOWDOWN"), "task": task.name,
           "seats": {s: {k: v for k, v in p.items() if k != "root"} for s, p in seats_prog.items()},
           "containers": [sim], "t_start": time.time()}
    players = {}
    try:
        sh(["docker", "network", "create", "--internal", net], check=True)
        cmd = ["docker", "run", "-d", "--name", sim, "--network", net, "--network-alias", "sim", "--cpus", "6",
               "--memory", "8g", "--pids-limit", "256", "--cap-drop", "NET_RAW", "--user", uid,
               "-v", f"{task / 'environment' / 'sim'}:/srv:ro", "-v", f"{out / 'data'}:/data", "-w", "/srv"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        sh(cmd + [d["sport"]["image"], "python3", "/srv/core/launch.py"], check=True)
        for _ in range(180):
            if (out / "data" / "server.pid").exists():
                break
            time.sleep(1)
        run["t_sim_up"] = time.time()
        (out / "driver.py").write_text(DRIVER)
        for s, p in seats_prog.items():
            name = f"{tag}-{s.lower()}"
            players[s] = name
            vols = ["-v", f"{out / 'driver.py'}:/offline/driver.py:ro", "-v", f"{out}:/offline/out"]
            if p.get("solution"):
                vols += ["-v", f"{task / 'solution'}:/solution:ro"]
            else:
                root = Path(p["root"])
                for sub in ("app", "tmp"):
                    (root / sub).mkdir(parents=True, exist_ok=True)
                    os.chmod(root / sub, 0o777)
                vols += ["-v", f"{root / 'app'}:/app", "-v", f"{root / 'tmp'}:/tmp"]
            sh(["docker", "run", "-d", "--name", name, "--hostname", hosts.get(s, "player"), "--network", net,
                "--network-alias", hosts.get(s, "player"), "--cpus", "2", "--memory", "4g", "--pids-limit", "128",
                "--cap-drop", "NET_RAW", "--user", uid, "-e", f"SEAT={s}", "-e", "HOME=/tmp",
                "-e", "SIM_URL=http://sim:8000", "-e", f"DRIVER_LOG=/offline/out/driver_{s}.jsonl", *vols,
                "-w", "/app", image, "python3", "/offline/driver.py", p["cmd"], p["cwd"], a.relaunch,
                str(a.timeout_s)], check=True)
            run["containers"].append(name)
        t_end = time.time() + a.timeout_s + 120
        while time.time() < t_end and any(_running(n) for n in players.values()):
            time.sleep(3)
        run["t_match_end"] = time.time()
        for s, n in players.items():
            r = sh(["docker", "logs", "--tail", "30", n])
            run.setdefault("driver_out", {})[s] = (r.stdout + r.stderr)[-1500:]
            run.setdefault("driver_rc", {})[s] = sh(["docker", "inspect", "-f", "{{.State.ExitCode}}", n]).stdout.strip()
        sh(["docker", "exec", sim, "sh", "-c", "kill -USR1 $(cat /data/server.pid)"])
        for _ in range(int(env.get("RO_REPLAY_TIMEOUT_S", "600")) + 60):
            try:
                if json.loads((out / "data" / "final.json").read_text()).get("graded_ready"):
                    break
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(1)
        run["t_graded"] = time.time()
    finally:
        run["cpu_s"] = {}
        for c in run["containers"]:
            cid = sh(["docker", "inspect", "-f", "{{.Id}}", c]).stdout.strip()
            try:
                st = Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/cpu.stat").read_text()
                run["cpu_s"][c] = round(int(st.split()[1]) / 1e6, 1)
            except (OSError, IndexError, ValueError):
                run["cpu_s"][c] = None
            sh(["docker", "rm", "-f", c])
        sh(["docker", "network", "rm", net])
    run["wall_s"] = round(time.time() - run["t_start"], 1)
    (out / "run.json").write_text(json.dumps(run, indent=1))
    return run


def _running(name):
    r = sh(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return r.returncode == 0 and r.stdout.strip() == "true"


def _resolve(v: str, over: dict) -> str:
    while True:
        m = re.search(r"\$\{([A-Z0-9_]+):-([^${}]*)\}", v)
        if not m:
            return v
        v = v[:m.start()] + over.get(m.group(1), m.group(2)) + v[m.end():]


def match_metrics(task, sport, d, out: Path, seat: str, image_sim: str, diag: bool) -> dict:
    fin = {}
    try:
        fin = json.loads((out / "data" / "final.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"dir": str(out), "valid": False, "error": "no final.json"}
    side = (fin.get("sides") or {}).get(seat, seat)
    scores = fin.get("scores") or {}
    own = scores.get(side)
    opp = sum(v for k, v in scores.items() if k != side and isinstance(v, (int, float)))
    m = {"dir": str(out), "valid": bool(fin.get("valid_trial")), "reason": fin.get("reason"), "scores": scores,
         "won": fin.get("winner") == side if fin.get("winner") is not None else None,
         "share": own / (own + opp) if isinstance(own, (int, float)) and (own + opp) > 0 else None,
         "steps": fin.get("steps"), "reaction": (fin.get("reaction") or {}).get("by_seat", {}).get(seat)}
    args = ["python3", "-m", "core.diagnose", "/srv/sport/plugin.py", "/d"]
    vols = ["-v", f"{task / 'environment' / 'sim'}:/srv:ro", "-v", f"{out / 'data'}:/d"]
    if diag and (REPO / "sports" / sport / "diagnose.py").exists():
        vols += ["-v", f"{REPO / 'sports' / sport / 'diagnose.py'}:/diag/diagnose.py:ro"]
        args += ["--diagnose", "/diag/diagnose.py"]
    else:
        args += ["--no-sport"]
    r = sh(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            "-e", "MUJOCO_GL=osmesa", "-e", "PYOPENGL_PLATFORM=osmesa", *vols, "-w", "/srv", image_sim, *args],
           timeout=3600)
    try:
        s = json.loads((out / "data" / "diag" / "summary.json").read_text())
        m["linkage"] = s["linkage"].get(seat)
        m["c1"] = s["c1"].get(seat)
        m["diag_metrics"] = s["metrics"].get(seat) if s.get("metrics") else None
        m["diag_errors"] = s.get("errors")
    except (OSError, json.JSONDecodeError, KeyError):
        m["diag_error"] = (r.stdout + r.stderr)[-800:]
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--n", type=int, default=None, help="with a snapshot directory: snapshot number (default: last)")
    ap.add_argument("--cmd")
    ap.add_argument("--cwd")
    ap.add_argument("--seat", default="A")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--slowdown", type=float, default=None)
    ap.add_argument("--first", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--opponent", default="builtin", help="format duel: builtin (the task's reference) or a snapshot")
    ap.add_argument("--opponent-cmd")
    ap.add_argument("--opponent-cwd")
    ap.add_argument("--relaunch", choices=("review", "never"), default="review")
    ap.add_argument("--player-image")
    ap.add_argument("--timeout-s", type=float, default=7200)
    ap.add_argument("--diagnose", action="store_true", help="also the sport's diagnose.py metrics per match")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    task, out = Path(a.task).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sport, d, t = task_info(task)
    seats = d["formats"][t["format"]]["seats"]
    if a.seat not in seats:
        raise SystemExit(f"seat {a.seat} is not a seat of format {t['format']} ({seats})")
    image = player_image(task, a.player_image)

    def prog(snap, n, cmd, cwd, label):
        root = out / f"snap_{label}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        info = snapshot_files(snap, n, root)
        picked = pick_program(info["procs"])
        if cmd is None and picked is None:
            raise SystemExit(f"{label}: no program command in the snapshot's procs.jsonl: give --cmd / --cwd")
        return {"root": str(root), "snapshot": info["tar"] or str(snap), "cmd": cmd or picked[0],
                "cwd": cwd or (picked[1] if picked and cmd is None else "/app/output"),
                "cmd_source": "given" if cmd else "procs.jsonl"}

    seats_prog = {a.seat: prog(a.snapshot, a.n, a.cmd, a.cwd, a.seat)}
    for s in seats:
        if s == a.seat:
            continue
        if a.opponent == "builtin":
            seats_prog[s] = {"solution": True, "cmd": "python3 oracle_play.py", "cwd": "/solution",
                             "cmd_source": "the task's reference player"}
        else:
            seats_prog[s] = prog(a.opponent, None, a.opponent_cmd, a.opponent_cwd, s)
    (out / "plan.json").write_text(json.dumps({"task": task.name, "sport": sport, "seats": {
        s: {k: v for k, v in p.items() if k != "root"} for s, p in seats_prog.items()}, "seeds": a.seeds,
        "level": a.level, "repeat": a.repeat, "image": image}, indent=1))
    rows = []
    for seed in [int(x) for x in a.seeds.split(",") if x]:
        for i in range(a.repeat):
            mdir = out / (f"s{seed}" + (f"_r{i}" if a.repeat > 1 else ""))
            for s, p in seats_prog.items():           # a fresh copy of the snapshot per match
                if p.get("root"):
                    fresh = mdir / f"root_{s}"
                    if fresh.exists():
                        shutil.rmtree(fresh)
                    shutil.copytree(p["root"], fresh, symlinks=True)
            run_match(task, sport, d, t, a, seed, mdir, image,
                      {s: ({**p, "root": str(mdir / f"root_{s}")} if p.get("root") else p)
                       for s, p in seats_prog.items()})
            row = match_metrics(task, sport, d, mdir, a.seat, d["sport"]["image"], a.diagnose)
            row.update(seed=seed, repeat=i)
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("seed", "repeat", "valid", "scores", "won", "share")}),
                  flush=True)
    valid = [r for r in rows if r.get("valid")]
    shares = [r["share"] for r in valid if r.get("share") is not None]
    pooled = {"matches": len(rows), "valid": len(valid), "wins": sum(1 for r in valid if r.get("won")),
              "win_rate": (sum(1 for r in valid if r.get("won")) / len(valid)) if valid else None,
              "mean_share": sum(shares) / len(shares) if shares else None}
    (out / "metrics.json").write_text(json.dumps({"task": task.name, "sport": sport, "seat": a.seat,
                                                   "program": {k: v for k, v in seats_prog[a.seat].items()
                                                               if k != "root"},
                                                   "level": a.level, "pooled": pooled, "matches": rows}, indent=1))
    print(json.dumps(pooled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
