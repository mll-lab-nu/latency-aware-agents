#!/usr/bin/env python3
"""Program snapshots of a running match, taken from outside by the runner (capability measurement framework §7, C2):
whenever a watched program file of a seat changes (its content hash), and at every `ready` / the end, copy the seat's
files out of its container (a program file: --watch-ext, e.g. .py .sh .toml .yaml; logs and state files
do not trigger, but every small file is in the snapshot); record the seat's processes (command line + working directory) whenever they
change. Nothing in the match changes: the watcher only reads the player containers (`docker exec` find / tar / /proc)
and the sim's episode log, never calls the game, and writes only under its output directory. Snapshots are never
part of a match record, a replay or a hash; offline evaluation (tools/offline_eval.py) reruns them.

    snapshot_watch.py --job-dir JOBS/JOB [--runner-pid PID]          a Harbor job (run_single / run_duel / run_team):
        the trial's containers <trial>__env-{sim,player,seat_<s>}-1; out = <trial>/snapshots/<seat>/
    snapshot_watch.py --container NAME --seat S --out DIR [--episode FILE | --sim-container NAME]
        one container (e.g. tools/live.py's), the seat's files under --roots
    common: [--roots /app,/tmp] [--poll 2] [--max-file-kb 2000] [--timeout-s 21600]

OUT/<seat>/snapshots.jsonl   one row per check that found a change or a ready / start / end:
    {n, ts, k (the sim's latest step, None if unknown), why (start | files | ready | end), tar (snap_NNN.tar.gz, or
     None when the files are those of the previous saved snapshot), files_sha, program_sha, n_files, n_program_files,
     bytes, changed [program files changed since the previous row], files {path: sha256[:16]}, procs_sha}
OUT/<seat>/snap_NNN.tar.gz   the files (relative to /: app/..., tmp/...), each < --max-file-kb, without images,
                             arrays, videos, caches, .git, pycache, Codex / Node caches
OUT/<seat>/procs.jsonl       {ts, k, procs: [{pid, ppid, cwd, args}]} whenever the seat's process set changes (the
                             model's launch command: arguments such as aiming constants live here, not in files)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

SKIP = ("! -path '*/__pycache__/*' ! -path '*/.git/*' ! -path 'tmp/codex*' ! -path 'tmp/node-compile-cache/*' "
        "! -path 'tmp/.cache/*' ! -path 'tmp/.claude*' ! -path 'tmp/.codex*' ! -path 'tmp/.npm/*' ! -path 'tmp/.config/*' ! -path '*/node_modules/*' ! -path 'app/output/frames/*' ! -path 'tmp/claude*' "
        "! -iname '*.png' ! -iname '*.jpg' ! -iname '*.jpeg' ! -name '*.npy' ! -name '*.npz' ! -name '*.mp4' "
        "! -name '*.pyc' ! -name '*.so' ! -name '*.pid' ! -name '*.sock'")
# program files: a change of one of these is a new program version (logs, images and state files are not)
PROG_EXT = ("py", "sh", "bash", "js", "mjs", "ts", "toml", "yaml", "yml", "cfg", "ini", "c", "cc", "cpp", "h", "hpp",
            "rs", "go", "jl", "r", "lua", "m")
PROG_NAMES = ("Makefile",)
# the processes of the container, without this watcher's own shell and its children
PROCS = r"""me=$$
for d in /proc/[0-9]*; do
  pid=${d#/proc/}; [ "$pid" = "$me" ] && continue
  ppid=$(cut -d' ' -f4 "$d/stat" 2>/dev/null); [ "$ppid" = "$me" ] && continue
  args=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null | cut -c1-2000); [ -n "$args" ] || continue
  printf '%s\t%s\t%s\t%s\n' "$pid" "$ppid" "$(readlink "$d/cwd" 2>/dev/null)" "$args"
done"""


def sh(args, timeout=60, binary=False):
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, b"" if binary else ""
    return r.returncode, (r.stdout if binary else r.stdout.decode(errors="replace"))


def running(name):
    rc, out = sh(["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=20)
    return rc == 0 and out.strip() == "true"


class Seat:
    def __init__(self, seat, container, out: Path, roots, max_kb, ep_seats=None, exts=None):
        self.seat, self.c, self.out = seat, container, out
        self.ep_seats = ep_seats                # the episode's seats whose ready rows count (None: every seat)
        out.mkdir(parents=True, exist_ok=True)
        rel = " ".join(r.strip("/") for r in roots)
        self.find = f"cd / && find {rel} -type f -size -{int(max_kb)}k {SKIP} 2>/dev/null"
        self.n, self.prog_manifest, self.saved_sha, self.saved_prog, self.prog, self.procs_key = 0, None, None, None, {}, None
        self.exts = set(exts or PROG_EXT)
        self.rows = (out / "snapshots.jsonl").open("a")
        self.plog = (out / "procs.jsonl").open("a")

    def check(self, k, why=None) -> bool:
        """One poll: the process set, then the files (fetched only when their size / mtime list changed or a
        ready / end asks for a row). False when the container is gone."""
        rc, txt = sh(["docker", "exec", self.c, "sh", "-c", PROCS], timeout=30)
        if rc not in (0, 1) and not running(self.c):
            return False
        procs = []
        for line in txt.splitlines():
            p = line.split("\t", 3)
            if len(p) == 4:
                procs.append({"pid": int(p[0]), "ppid": int(p[1]) if p[1].isdigit() else None, "cwd": p[2],
                              "args": p[3].strip()})
        key = hashlib.sha256(json.dumps(sorted((x["pid"], x["cwd"], x["args"]) for x in procs)).encode()).hexdigest()[:16]
        if key != self.procs_key:
            self.procs_key = key
            self.plog.write(json.dumps({"ts": round(time.time(), 3), "k": k, "procs_sha": key, "procs": procs}) + "\n")
            self.plog.flush()
        rc, man = sh(["docker", "exec", self.c, "sh", "-c", self.find + " -printf '%p\\t%s\\t%T@\\n' | sort"],
                     timeout=60)
        prog_man = "\n".join(x for x in man.splitlines() if self.is_prog(x.split("\t")[0]))
        if prog_man == self.prog_manifest and why is None:
            return True                      # no program file touched (logs and state files do not count)
        self.prog_manifest = prog_man
        t0 = time.time()
        rc, blob = sh(["docker", "exec", self.c, "sh", "-c", self.find + " -print0 | tar --null -T - -cf - 2>/dev/null"],
                      timeout=120, binary=True)
        files = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        files[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()[:16]
        except tarfile.TarError:
            pass
        prog = {p: h for p, h in files.items() if self.is_prog(p)}
        sha = digest(files)
        prog_sha = digest(prog)
        if prog_sha == self.saved_prog and why is None:
            return True                      # only mtimes moved (a program file rewritten with the same bytes)
        changed = sorted(p for p in set(prog) | set(self.prog) if prog.get(p) != self.prog.get(p))
        self.n += 1
        tar = None
        if sha != self.saved_sha:            # a program file changed, or (ready / end) any file since the last save
            tar = f"snap_{self.n:03d}.tar.gz"
            (self.out / tar).write_bytes(gzip.compress(blob))
            self.saved_sha = sha
        self.saved_prog = prog_sha
        self.rows.write(json.dumps({"n": self.n, "ts": round(t0, 3), "k": k,
                                    "why": why or ("start" if self.n == 1 else "files"), "tar": tar,
                                    "files_sha": sha, "program_sha": prog_sha, "n_files": len(files),
                                    "n_program_files": len(prog), "bytes": len(blob),
                                    "changed": changed if self.n > 1 else [], "files": files,
                                    "procs_sha": self.procs_key}) + "\n")
        self.rows.flush()
        self.prog = prog
        return True

    def is_prog(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return name in PROG_NAMES or ("." in name and name.rsplit(".", 1)[1].lower() in self.exts)


def digest(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]


class Episode:
    """The sim's episode log, read from a file or from the sim container: the latest step, new ready / end rows."""

    def __init__(self, path=None, container=None):
        self.path, self.c, self.off, self.k, self.buf = path, container, 0, None, b""

    def poll(self):
        if self.path:
            try:
                with open(self.path, "rb") as f:
                    f.seek(self.off)
                    data = f.read()
            except OSError:
                return []
        elif self.c:
            rc, data = sh(["docker", "exec", self.c, "sh", "-c", f"tail -c +{self.off + 1} /data/episode.jsonl"],
                          timeout=30, binary=True)
            if rc != 0:
                return []
        else:
            return []
        self.off += len(data)
        self.buf += data
        lines = self.buf.split(b"\n")
        self.buf = lines.pop()
        out = []
        for ln in lines:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "k" in r and isinstance(r["k"], int):
                self.k = r["k"]
            if r.get("ev") in ("ready", "review", "end"):
                out.append(r)
        return out


def find_trial(job_dir: Path):
    for d in sorted(job_dir.glob("*/")) if job_dir.exists() else []:
        if "__" in d.name or (d / "config.json").exists():
            return d
    return None


def harbor_seats(trial: Path) -> tuple[str, dict]:
    proj = trial.name.lower() + "__env"
    rc, names = sh(["docker", "ps", "--format", "{{.Names}}"], timeout=20)
    names = set(names.split())
    seats = {}
    if f"{proj}-player-1" in names:
        seats["A"] = f"{proj}-player-1"
    for n in sorted(names):
        if n.startswith(f"{proj}-seat_") and n.endswith("-1"):
            seats[n[len(f"{proj}-seat_"):-2].upper()] = n
    return f"{proj}-sim-1", seats


def watch(seats: list[Seat], ep: Episode, poll: float, runner_pid: int, timeout: float):
    t0 = time.time()
    for s in seats:
        s.check(ep.k, "start")
    ended = False
    while time.time() - t0 < timeout and not ended:
        time.sleep(poll)
        rows = ep.poll()
        alive = []
        for s in seats:
            why = None
            mine = [r for r in rows if r.get("ev") == "end" or s.ep_seats is None or r.get("seat") in s.ep_seats]
            if any(r.get("ev") == "end" for r in mine):
                why, ended = "end", True
            elif mine:
                why = "ready"
            alive.append(s.check(ep.k, why))
        if not any(alive):
            break
        if runner_pid and not Path(f"/proc/{runner_pid}").exists():
            for s in seats:
                s.check(ep.k, "end")
            break


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir")
    ap.add_argument("--container")
    ap.add_argument("--seat", default="A")
    ap.add_argument("--out")
    ap.add_argument("--episode", help="the sim's episode.jsonl on this host")
    ap.add_argument("--sim-container", help="read /data/episode.jsonl in this container")
    ap.add_argument("--episode-seats", default="", help="--container: the seats whose ready rows trigger a snapshot "
                    "(default: every seat)")
    ap.add_argument("--roots", default="/app,/tmp")
    ap.add_argument("--watch-ext", default=",".join(PROG_EXT), help="extensions of the program files whose change "
                    "triggers a snapshot (every file under --roots below --max-file-kb is saved)")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--max-file-kb", type=int, default=2000)
    ap.add_argument("--runner-pid", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=6 * 3600)
    a = ap.parse_args(argv)
    roots = [r for r in a.roots.split(",") if r]
    exts = [x.strip(".").lower() for x in a.watch_ext.split(",") if x]
    if a.container:
        if not a.out:
            ap.error("--container needs --out")
        for _ in range(int(600 / a.poll)):
            if running(a.container):
                break
            time.sleep(a.poll)
        seats = [Seat(a.seat, a.container, Path(a.out) / a.seat, roots, a.max_file_kb,
                      [x for x in a.episode_seats.split(",") if x] or None, exts)]
        watch(seats, Episode(a.episode, a.sim_container), a.poll, a.runner_pid, a.timeout_s)
        return 0
    if not a.job_dir:
        ap.error("--job-dir or --container")
    job, t0 = Path(a.job_dir), time.time()
    while True:                                  # the trial directory, then its containers
        trial = find_trial(job)
        if trial is not None:
            sim, found = harbor_seats(trial)
            if found and running(sim):
                break
        if (a.runner_pid and not Path(f"/proc/{a.runner_pid}").exists()) or time.time() - t0 > 3600:
            return 1
        time.sleep(a.poll)
    out = Path(a.out) if a.out else trial / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    (out / "containers.json").write_text(json.dumps({"trial": str(trial), "sim": sim, "seats": found}))
    seats = [Seat(s, c, out / s, roots, a.max_file_kb, [s], exts) for s, c in sorted(found.items())]
    watch(seats, Episode(container=sim), a.poll, a.runner_pid, a.timeout_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
