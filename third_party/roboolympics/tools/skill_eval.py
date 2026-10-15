#!/usr/bin/env python3
"""Post-match evaluation of a low-level skill in an isolated container (docs/framework_zh.md §6.3.7, §9 robot and mode
rule 10; skilleval/README.md). Host side, standard library only (python >= 3.11) + docker; the evaluation itself is
skilleval/evaluate.py inside the sport's image.

    python3 tools/skill_eval.py SPORT SKILL [--out DIR] [--set hidden|public|all] [--no-competition] [--checks]
                                [--label NAME] [--cpus 2] [--memory 4g] [--timeout 1800] [--inloop-log FILE]

SKILL: a skill directory (a trial's artifact .../app/output/skill holding the deliverable, e.g. aim.py), or
  platform   our implementation of the skill API (the high level's), in process: the reference row
  reference  sport.toml [modes.low] reference: our skill from the toolkit only (rule 4)
  boundary   sport.toml [modes.low] boundary: the toolkit without skill development (rule 5)
  probe      skilleval/probe/: tries to read the simulator and the hidden requests, to write, to reach the network

The container: the sport's image (or [modes.low] eval_image), no network, a read-only root file system with a small
/tmp, CPU / memory / process limits, all capabilities dropped but the three the evaluator needs to read the simulator
directory (owner-only on the host) and to start the skill as `nobody` (DAC_OVERRIDE, SETUID, SETGID; no new
privileges). Mounted read-only: the simulator directory with the hidden requests at /srv/sim (tools/sportlib.py
assemble_sim), the skill at /app/output/skill (regular files only, copied first: no symlink leaves the copy), and the
toolkit at the seat's paths ([seat.low] files). Nothing is mounted writable: the report comes back on stdout and is
written to OUT/report.json. A time-out kills the container by its exact name.

--inloop-log: the match's skill-in-the-loop log (the trial's artifact inloop_<seat>.jsonl, skilleval/inloop.py): the
report's `inloop` says which versions of the skill played the match (sha256, calls) and whether the evaluated skill
(skill_sha256, the same hash) is the one that played its last call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sportlib  # noqa: E402

MAX_SKILL_BYTES = 256 << 20
CAPS = ("DAC_OVERRIDE", "SETUID", "SETGID")


def copy_skill(src: Path, dst: Path) -> int:
    """Regular files and directories only (symlinks and special files are skipped), world-readable, size-capped."""
    total = 0
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        for d in dirs:
            p = Path(root) / d
            if not p.is_symlink() and d != "__pycache__":
                (dst / rel / d).mkdir()
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink() and d != "__pycache__"]
        for f in files:
            p = Path(root) / f
            st = p.lstat()
            if not stat.S_ISREG(st.st_mode):
                continue
            total += st.st_size
            if total > MAX_SKILL_BYTES:
                raise SystemExit(f"the skill directory is larger than {MAX_SKILL_BYTES >> 20} MB")
            shutil.copyfile(p, dst / rel / f)
    for p in [dst, *dst.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    return total


def eval_image(sport: str, lo: dict) -> str:
    """The sport's image, or [modes.low] eval_image; with [modes.low] eval_pip a derived image built once (the
    packages the seat promises a skill, e.g. scipy, on top of the simulator's image)."""
    base = lo.get("eval_image") or sportlib.image(sport)
    pip = list(lo.get("eval_pip") or [])
    if not pip:
        return base
    tag = "ro-skill-eval-" + hashlib.sha256(json.dumps([base, pip]).encode()).hexdigest()[:12] + ":1"
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Dockerfile").write_text(f"FROM {base}\nRUN pip install --no-cache-dir {' '.join(pip)}\n")
            subprocess.run(["docker", "build", "-q", "-t", tag, tmp], check=True, capture_output=True)
    return tag


def run(
    sport: str,
    skill: str,
    out: Path | None = None,
    which: str = "hidden",
    competition: bool = True,
    checks: bool = False,
    label: str | None = None,
    cpus: str = "2",
    memory: str = "4g",
    timeout: float = 1800.0,
) -> dict:
    d = sportlib.decl(sport)
    lo = (d.get("modes") or {}).get("low")
    if not isinstance(lo, dict):
        raise SystemExit(f"{sport}: no [modes.low] in sport.toml (nothing to evaluate)")
    src = sportlib.SPORTS / sport
    named = {
        "reference": src / lo["reference"],
        "boundary": src / lo["boundary"],
        "probe": sportlib.REPO / "skilleval" / "probe",
    }
    impl = "platform" if skill == "platform" else "skill"
    label = label or (skill if skill in (*named, "platform") else "player")
    with tempfile.TemporaryDirectory(prefix=f"skill-eval-{sport}-") as tmp:
        tmp = Path(tmp)
        sim = sportlib.assemble_sim(sport, tmp / "sim", hidden=True)
        sim.chmod(0o700)  # root only inside the container (the skill runs as nobody)
        mounts = ["-v", f"{sim}:/srv/sim:ro"]
        if impl == "skill":
            sdir = named.get(skill, Path(skill))
            if not sdir.is_dir():
                raise SystemExit(f"no skill directory {sdir}")
            copy_skill(sdir.resolve(), tmp / "skill")
            mounts += ["-v", f"{tmp / 'skill'}:/app/output/skill:ro"]
        for dst, rel in sorted((((d.get("seat") or {}).get("low") or {}).get("files") or {}).items()):
            mounts += ["-v", f"{(src / rel).resolve()}:{dst}:ro"]
        name = f"ro-skill-eval-{sport}-{os.getpid()}-{int(time.time())}"
        iso = (
            f"docker {name}: network none, read-only root, {cpus} CPUs, {memory}, pids 256, caps "
            f"{'+'.join(CAPS)} only, no-new-privileges; the skill as nobody in its own process"
        )
        args = [
            "--sport-dir",
            "/srv/sim/sport",
            "--impl",
            impl,
            "--skill",
            "/app/output/skill",
            "--set",
            which,
            "--label",
            label,
            "--isolation",
            iso,
        ]
        args += ["--probe"] if skill == "probe" else []
        args += [] if competition else ["--no-competition"]
        args += ["--checks"] if checks else []
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=512m,mode=1777",
            "--cpus",
            cpus,
            "--memory",
            memory,
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            *[x for c in CAPS for x in ("--cap-add", c)],
            "--security-opt",
            "no-new-privileges",
            "--user",
            "0:0",
            "-e",
            "HOME=/tmp",
            "-e",
            "MUJOCO_GL=osmesa",
            "-e",
            "PYOPENGL_PLATFORM=osmesa",
            *mounts,
            "-w",
            "/srv/sim",
            eval_image(sport, lo),
            "python3",
            "-B",
            "-m",
            "skilleval.evaluate",
            *args,
        ]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            report = {"valid": False, "reason": f"the evaluation exceeded {timeout} s", "sport": sport, "label": label}
        else:
            text = r.stdout.split("\nREPORT\n")[-1].strip() if "\nREPORT\n" in r.stdout else ""
            try:
                report = json.loads(text)
            except ValueError:
                report = {
                    "valid": False,
                    "sport": sport,
                    "label": label,
                    "reason": f"the evaluator gave no report (exit {r.returncode}): {r.stderr[-1500:]}",
                }
        report["wall_s"] = round(time.time() - t0, 1)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    return report


def inloop_check(report: dict, log_path) -> dict:
    """The versions of the skill that played the match (skilleval/inloop.py log) against the evaluated one."""
    rows = [json.loads(x) for x in Path(log_path).read_text().splitlines() if x.strip()]
    calls = [r for r in rows if r.get("op") == "call"]
    versions: dict = {}
    for r in calls:
        if r.get("sha256"):
            v = versions.setdefault(r["sha256"], {"sha256": r["sha256"], "calls": 0, "first_unit": r.get("unit")})
            v["calls"] += 1
    last = calls[-1].get("sha256") if calls else None
    ev = report.get("skill_sha256")
    return {"log": str(log_path), "calls": len(calls), "errors": sum(1 for r in calls if not r.get("ok")),
            "versions": list(versions.values()), "last_sha256": last, "evaluated_sha256": ev,
            "evaluated_is_last": bool(ev) and ev == last, "evaluated_played": bool(ev) and ev in versions}


def summary(r: dict) -> str:
    if "probe" in r:
        return f"probe: {'ok' if r['probe']['ok'] else 'LEAKS ' + str(r['probe']['leaks'])} {r['probe']['results']}"
    if not r.get("valid"):
        return f"{r.get('label')}: not valid: {r.get('reason')}"
    a, c = r.get("api_test", {}), r.get("competition_test", {})
    out = f"{r.get('label')}: API test {a.get('success')}/{a.get('n')} ({a.get('success_rate')})"
    if c:
        out += f"; competition test {c.get('success')}/{c.get('n')} ({c.get('success_rate')})"
    out += f"; deterministic {r.get('determinism', {}).get('deterministic')}; {r.get('wall_s')} s"
    if r.get("inloop"):
        il = r["inloop"]
        out += (f"; in the loop: {il['calls']} calls, {len(il['versions'])} version(s), evaluated = last played: "
                f"{il['evaluated_is_last']}")
    return out


def main(argv):
    ap = argparse.ArgumentParser(prog="tools/skill_eval.py")
    ap.add_argument("sport")
    ap.add_argument("skill")
    ap.add_argument("--out")
    ap.add_argument("--set", default="hidden", choices=["hidden", "public", "all"])
    ap.add_argument("--no-competition", action="store_true")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--inloop-log", help="the match's inloop_<seat>.jsonl (the skill in the loop)")
    ap.add_argument(
        "--record",
        action="store_true",
        help="platform / reference / boundary: store the summary in the " "sport's [modes.low] reference_results file",
    )
    a = ap.parse_args(argv)
    r = run(
        a.sport,
        a.skill,
        Path(a.out) if a.out else None,
        a.set,
        not a.no_competition,
        a.checks,
        a.label,
        a.cpus,
        a.memory,
        a.timeout,
    )
    if a.inloop_log:
        r["inloop"] = inloop_check(r, a.inloop_log)
        if a.out:
            (Path(a.out) / "report.json").write_text(json.dumps(r, indent=1, sort_keys=True, default=float) + "\n")
    print(summary(r))
    if a.record and r.get("valid") and a.skill in ("platform", "reference", "boundary"):
        print("recorded in", record(a.sport, a.skill, r))
    return 0 if r.get("valid") else 1


def record(sport: str, label: str, r: dict) -> Path:
    """The reference rows the report of every evaluation carries (skilleval/evaluate.py reference_results)."""
    lo = sportlib.decl(sport)["modes"]["low"]
    p = sportlib.SPORTS / sport / lo["reference_results"]
    rows = json.loads(p.read_text()) if p.is_file() else {}
    rows[label] = {
        k: r.get(k) for k in ("api_test", "competition_test", "determinism", "revision", "toolkit", "skill_sha256")
    }
    rows[label]["date"] = time.strftime("%Y-%m-%d")
    p.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    return p


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
