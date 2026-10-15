"""Job queue: jobs.jsonl lines {"tag", "script", "repo", "env": {...}}; at most MAXC containers of ours at once (2 when
the host's 1-min load > 40); a job is done when out/<tag>.done exists. Re-reads jobs.jsonl every loop, so jobs can be
appended. Stops when STOP exists and nothing runs."""
import json, os, subprocess, time
W = os.path.dirname(os.path.abspath(__file__))
MAXC = int(os.environ.get("MAXC", "4"))
running = {}
while True:
    for tag, p in list(running.items()):
        if p.poll() is not None:
            if p.returncode == 0:
                open(f"{W}/out/{tag}.done", "w").write("ok")
            else:
                open(f"{W}/out/{tag}.fail", "a").write(f"rc {p.returncode}\n")
            del running[tag]
    jobs = [json.loads(l) for l in open(f"{W}/jobs.jsonl") if l.strip()]
    load = os.getloadavg()[0]
    nb9 = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
    others = sum(1 for n in nb9 if (n.startswith("bmscreen-") and n[9:] not in running) or n.startswith("bmcheck"))
    try:
        maxc = int(open(f"{W}/MAXC").read().strip())
    except (OSError, ValueError):
        maxc = MAXC
    cap = (min(2, maxc) if load > 40 else maxc) - others
    for j in jobs:
        if len(running) >= cap:
            break
        t = j["tag"]
        if t in running or os.path.exists(f"{W}/out/{t}.done") or os.path.exists(f"{W}/out/{t}.fail"):
            continue
        if f"bmscreen-{t}" in nb9:          # still running from an earlier queue process
            continue
        if os.path.exists(f"{W}/out/{t}.jsonl") and "SEEDS" in j.get("env", {}):
            n_have = len({json.loads(l)["seed"] for l in open(f"{W}/out/{t}.jsonl") if l.strip()})
            if n_have >= len(str(j["env"]["SEEDS"]).split(",")):
                open(f"{W}/out/{t}.done", "w").write("ok (adopted)")
                continue
        env = [f"{k}={v}" for k, v in j.get("env", {}).items()] + [f"TAG={t}"]
        log = open(f"{W}/out/{t}.log", "a")
        running[t] = subprocess.Popen([f"{W}/run.sh", f"bmscreen-{t}", j["repo"],
                                       j.get("script", "scr.py")] + env, stdout=log, stderr=subprocess.STDOUT)
    if not running and os.path.exists(f"{W}/STOP"):
        break
    time.sleep(15)
