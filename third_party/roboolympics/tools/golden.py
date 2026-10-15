#!/usr/bin/env python3
"""Golden replays (docs/framework_zh.md §10; core rule 4): every sport keeps recorded matches whose replay through the
CURRENT core and sport code must reproduce the recorded result exactly (state hash, scores, winner, steps).

    python3 tools/golden.py replay SPORT [--external DIR]    replay sports/SPORT/golden/*/ (+ converted episodes of
                                                              the golden-data store DIR, if the sport has a converter)
    python3 tools/golden.py record SPORT CASE --format F --seed N [--first S] [--level L] [--players a,b]
                                                              record a new golden case with built-in players

A case is sports/SPORT/golden/CASE/episode.jsonl[.gz] + expected.json ({"expect": {state_sha, scores, winner, k}}).
Each replay runs in a fresh process in the sport's simulator image, without network. For a migrated sport, a
converter sports/SPORT/golden/convert.py (convert(old_dir, out_dir) -> expected dict) turns the matches recorded by
the legacy task (the golden-data store, RO_GOLDEN_DATA or --external) into core episodes, which must replay to the
legacy result. Needs python >= 3.11 and docker.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sportlib  # noqa: E402

KEYS = ("state_sha", "scores", "winner", "k")


def docker(sport, sim, case, cmd, name):
    return subprocess.run(["docker", "run", "--rm", "--name", name, "--network", "none", "-v", f"{sim}:/srv:ro",
                           "-v", f"{case}:/ep", "-w", "/srv", "-e", "MUJOCO_GL=osmesa", "-e", "PYOPENGL_PLATFORM=osmesa",
                           "--user", f"{os.getuid()}:{os.getgid()}", sportlib.image(sport), "sh", "-c", cmd],
                          capture_output=True, text=True, timeout=sportlib.stage_timeout(sport))


def replay_case(sport, sim, src: Path, work: Path) -> dict:
    case = work / src.name
    case.mkdir(parents=True, exist_ok=True)
    ep = src / "episode.jsonl"
    if ep.exists():
        shutil.copy2(ep, case / "episode.jsonl")
    else:
        with gzip.open(src / "episode.jsonl.gz", "rb") as f:
            (case / "episode.jsonl").write_bytes(f.read())
    want = json.loads((src / "expected.json").read_text())["expect"]
    r = docker(sport, sim, case, "python3 -m core.episode /srv/sport/plugin.py /ep",
               f"golden-{sport}-{src.name}-{os.getpid()}")
    if r.returncode != 0:
        return {"case": src.name, "ok": False, "error": (r.stderr or r.stdout)[-600:]}
    got = json.loads((case / "replay_check.json").read_text())
    diff = {k: {"want": want[k], "got": got.get(k)} for k in want if got.get(k) != want[k]}
    return {"case": src.name, "ok": not diff, "diff": diff, "state_sha": got["state_sha"][:16],
            "scores": got["scores"], "winner": got["winner"], "k": got["k"]}


def replay(sport: str, external: str | None = None) -> list[dict]:
    gdir = sportlib.SPORTS / sport / sportlib.decl(sport).get("golden", {}).get("episodes", "golden")
    with tempfile.TemporaryDirectory(prefix=f"golden-{sport}-") as tmp:
        sim = sportlib.assemble_sim(sport, Path(tmp) / "sim")
        cases = sorted(p for p in gdir.glob("*/") if (p / "expected.json").exists()) if gdir.is_dir() else []
        out = [replay_case(sport, sim, c, Path(tmp) / "cases") for c in cases]
        conv = gdir / "convert.py"
        ext = external or os.environ.get("RO_GOLDEN_DATA")
        if conv.exists() and ext and Path(ext).is_dir():
            spec = importlib.util.spec_from_file_location("convert", conv)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for old in mod.cases(Path(ext)):
                dst = Path(tmp) / "external" / f"{old.parent.name}--{old.name}"
                dst.mkdir(parents=True)
                want = mod.convert(old, dst)
                (dst / "expected.json").write_text(json.dumps({"expect": want}))
                out.append({**replay_case(sport, sim, dst, Path(tmp) / "ext-cases"), "external": True})
    return out


def record(sport, case, fmt, seed, first, level, players):
    """Record a golden case with built-in players in-process (core/local.py), then replay it once to fix expected."""
    f = sportlib.decl(sport)["formats"][fmt]
    seats = f["seats"]
    names = dict(zip(seats, (players.split(",") if players else ["reference"] * len(seats))))
    dst = sportlib.SPORTS / sport / "golden" / case
    dst.mkdir(parents=True, exist_ok=True)
    cfg = dict(sport=sport, format=fmt, seats=seats, seed=seed, first=first, action_level=level, sides=f.get("sides"))
    code = ("import json, sys\nsys.path.insert(0, '/srv')\nfrom core.episode import load_plugin\n"
            "from core.sport import MatchConfig\nfrom core import local\nP = '/srv/sport/plugin.py'\n"
            f"S = load_plugin(P)\ncfg = MatchConfig.from_json({cfg!r})\n"
            f"r = local.play(S, cfg, local.make_players(S, cfg, {names!r}, plugin=P), data='/ep', plugin=P)\n"
            "print(json.dumps({k: r[k] for k in ('scores', 'winner', 'k', 'digest')}))\n")
    with tempfile.TemporaryDirectory(prefix=f"golden-{sport}-") as tmp:
        sim = sportlib.assemble_sim(sport, Path(tmp) / "sim")
        work = Path(tmp) / "rec"
        work.mkdir()
        (work / "record.py").write_text(code)
        r = docker(sport, sim, work, "python3 /ep/record.py", f"golden-rec-{sport}-{os.getpid()}")
        if r.returncode != 0:
            raise SystemExit(r.stderr[-2000:])
        live = json.loads(r.stdout.strip().splitlines()[-1])
        shutil.copy2(work / "episode.jsonl", dst / "episode.jsonl")
        r = docker(sport, sim, work, "python3 -m core.episode /srv/sport/plugin.py /ep",
                   f"golden-rep-{sport}-{os.getpid()}")
        got = json.loads((work / "replay_check.json").read_text())
    if got["state_sha"] != live["digest"] or got["scores"] != live["scores"] or got["winner"] != live["winner"]:
        raise SystemExit(f"the fresh-process replay disagrees with the recording: {got} vs {live}")
    exp = {"expect": {k: got[k] for k in KEYS}, "source": f"core/local.py, players {names}, cfg {cfg}"}
    (dst / "expected.json").write_text(json.dumps(exp, indent=1) + "\n")
    return exp


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("replay", "record"))
    ap.add_argument("sport")
    ap.add_argument("case", nargs="?")
    ap.add_argument("--external")
    ap.add_argument("--format", default="duel")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--first", default="A")
    ap.add_argument("--level", default="high")
    ap.add_argument("--players", default="")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(json.dumps(record(a.sport, a.case, a.format, a.seed, a.first, a.level, a.players)))
        return 0
    res = replay(a.sport, a.external)
    for r in res:
        print(json.dumps(r))
    bad = [r for r in res if not r["ok"]]
    print(f"golden {a.sport}: {len(res) - len(bad)}/{len(res)} identical")
    return 1 if bad or not res else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
