#!/usr/bin/env python3
"""Run a grid of screening matches (tools/screen/play.py) in the sport's image, a few containers at a time.

    python3 tools/screen/batch.py SPORT --variants reference,fixed --levels 1-4 --seeds 1-10 --data DIR \
        [--jobs 4] [--cpus 2] [--max-load 40] [--delay D] [--opt K=V ...] [--fresh-sim]

DIR/sim          the simulator directory (tools/sportlib.assemble_sim, as a generated task gets it), assembled once
                 (DIR/sim/STAMP holds the commit it came from; --fresh-sim re-assembles it). Every match of a batch
                 runs the same frozen copy, so editing the checkout while a batch runs changes nothing.
DIR/code         a copy of tools/screen/play.py and sports/SPORT/screen/ (the hooks)
DIR/runs/<variant>/L<level>_s<seed>/result.json   one match (skipped when it exists; a run.log without a result
                 means a match is running or failed: delete the directory to run it again)

Containers are named screen-<sport>-<variant>-L<level>-s<seed> (no network, one user, read-only code). When the
host's 1-minute load average exceeds --max-load, at most 2 run at once; a number in DIR/JOBS overrides --jobs while
the batch runs. Matches are started seed-major (every variant
of a seed before the next seed), so a partial batch stays paired.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sportlib  # noqa: E402

REPO = sportlib.REPO


def ints(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def head() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "sports", "core", "bodies", "tools"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def stage(sport: str, data: Path, fresh: bool) -> None:
    sim = data / "sim"
    if fresh or not (sim / "STAMP").exists():
        sportlib.assemble_sim(sport, sim)
        (sim / "STAMP").write_text(head() + "\n")
    code = data / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)
    shutil.copy2(REPO / "tools" / "screen" / "play.py", code / "play.py")
    shutil.copytree(REPO / "sports" / sport / "screen", code / "hooks", ignore=sportlib.IGNORE)


def load1() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sport")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--levels", default="")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cpus", type=float, default=2)
    ap.add_argument("--max-load", type=float, default=40.0)
    ap.add_argument("--delay", type=int, default=None, help="an experiment's effect delay (default: the sport's "
                    "declared reaction delay, sport.toml [timing] reaction_delay_steps; play.py)")
    ap.add_argument("--action-level", default=None, help="the action level (default: the hooks' ACTION_LEVEL)")
    ap.add_argument("--name-prefix", default="screen", help="container names: <prefix>-<sport>-<variant>-L<l>-s<seed>")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--fresh-sim", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--format", default=None, help="the format (default: the hooks' FORMAT); duel: both seats play")
    ap.add_argument("--firsts", default="", help="format duel: first movers to play, e.g. A,B (the fairness check)")
    ap.add_argument("--every", type=int, default=None, help="observation period without a delay (play.py --every)")
    ap.add_argument("--episode", action="store_true", help="each match also writes episode.jsonl (play.py --episode; "
                    "core 0.2.25: replay + diagnostic rows for python -m core.diagnose)")
    a = ap.parse_args(argv)
    data = Path(a.data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    stage(a.sport, data, a.fresh_sim)
    image = a.image or sportlib.image(a.sport)
    levels = ints(a.levels) or [None]
    todo = []
    firsts = [f for f in a.firsts.split(",") if f] or [None]
    for s in ints(a.seeds):
        for lv in levels:
            for fi in firsts:
                for v in a.variants.split(","):
                    tag = (f"L{lv}_" if lv is not None else "") + (f"F{fi}_" if fi else "") + f"s{s}"
                    d = data / "runs" / v / tag
                    if not (d / "result.json").exists() and not (d / "run.log").exists():
                        todo.append((v, lv, s, d, fi))
    print(f"{len(todo)} matches to run ({a.sport}, image {image}, sim {(data / 'sim' / 'STAMP').read_text().strip()})",
          flush=True)
    running: list = []
    t0, done = time.time(), 0
    while todo or running:
        jobs = a.jobs
        try:                                       # DIR/JOBS (a number) changes the limit while the batch runs
            jobs = int((data / "JOBS").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            pass
        cap = jobs if load1() <= a.max_load else min(2, jobs)
        while todo and len(running) < cap:
            v, lv, s, d, fi = todo.pop(0)
            d.mkdir(parents=True, exist_ok=True)
            name = f"{a.name_prefix}-{a.sport}-{v.replace('_', '')}-L{lv}-" + (f"F{fi}-" if fi else "") + f"s{s}"
            args = ["--hooks", "/code/hooks/hooks.py", "--variant", v, "--seed", str(s), "--out", "/out"]
            if lv is not None:
                args += ["--level", str(lv)]
            if a.delay is not None:
                args += ["--delay", str(a.delay)]
            if a.format:
                args += ["--format", a.format]
            if fi:
                args += ["--first", fi]
            if a.every is not None:
                args += ["--every", str(a.every)]
            if a.action_level:
                args += ["--action-level", a.action_level]
            if a.episode:
                args += ["--episode"]
            for o in a.opt:
                args += ["--opt", o]
            cmd = ["docker", "run", "--rm", "--name", name, "--cpus", str(a.cpus), "--network", "none",
                   "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=1",
                   "-e", "LP_NUM_THREADS=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-v", f"{data / 'sim'}:/srv:ro", "-v", f"{data / 'code'}:/code:ro", "-v", f"{d}:/out",
                   "-w", "/srv", image, "python3", "/code/play.py", *args]
            log = open(d / "run.log", "w")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), v, lv, s, d, log))
        time.sleep(3)
        for item in list(running):
            p, v, lv, s, d, log = item
            if p.poll() is None:
                continue
            running.remove(item)
            log.close()
            done += 1
            tail = [x for x in (d / "run.log").read_text().splitlines() if x.startswith("{")][-1:] or ["FAILED"]
            print(f"[{time.time() - t0:7.0f}s] {done} rc={p.returncode} {v} L{lv} s{s}: {tail[0][:150]}", flush=True)


if __name__ == "__main__":
    main()
