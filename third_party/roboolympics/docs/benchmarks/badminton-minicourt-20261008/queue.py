"""Run job lines ("KEY=val ... script.py args") through run.sh, at most P at a time (shared host). Host-side, stdlib.
    python3 queue.py jobs.txt [P]      logs: results/logs/<jobfile>-<n>.log"""
import os, shlex, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
jobs = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip() and not l.startswith("#")]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 5
(HERE / "results" / "logs").mkdir(parents=True, exist_ok=True)
run, stem = [], Path(sys.argv[1]).stem
for n, line in enumerate(jobs):
    while len([p for p in run if p.poll() is None]) >= P:
        time.sleep(5)
    toks = shlex.split(line)
    env = dict(os.environ)
    i = 0
    while "=" in toks[i] and not toks[i].endswith(".py"):
        k, v = toks[i].split("=", 1); env[k] = v; i += 1
    env["NAME"] = f"bm-mini-{stem}-{n}-{os.getpid()}"
    log = open(HERE / "results" / "logs" / f"{stem}-{n}.log", "w")
    log.write(line + "\n"); log.flush()
    run.append(subprocess.Popen(["./run.sh"] + toks[i:], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT))
for p in run:
    p.wait()
print("done", len(run))
