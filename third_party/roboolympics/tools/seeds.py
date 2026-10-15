#!/usr/bin/env python3
"""Hidden official seeds of a sport (core 0.2.14, core/seeds.py; docs/framework_zh.md §6.4; CONTRIBUTING.md "Freeze a
sport"). The set is partitioned into cells, one per format x level (cells(): "<format>/L<level>" for a format with
levels, else "<format>"), each with its own seeds.

    python3 tools/seeds.py new SPORT --count N [--counts CELL=N,...] [--from FILE] [--force]
        a fresh private set (N seeds per cell; --counts overrides single cells), or the seeds of FILE: a JSON object
        {"<format>": [seeds] | {"<level>": [seeds]}} (fresh salts). Written OUTSIDE the repository to
        $RO_SEEDS_DIR/<sport>.hidden_seeds.json (mode 0600; official runs: the restricted directory of the official run
        host, CONTRIBUTING.md); its public part to sports/<sport>/seed_commitments.json; the registration (sha256,
        counts per cell) to sport.toml [official.hidden_seeds]
    python3 tools/seeds.py check SPORT [--private FILE]
        registration = public commitments; with the private set on this host (or FILE), also private = registration
    python3 tools/seeds.py resolve TASK_DIR INDEX [--level L]
    python3 tools/seeds.py resolve SPORT INDEX --cell CELL
        "SEED SALT" of entry INDEX of the match's cell (a generated task: its format from tests/contract.json, the level
        from --level), after checking that the private set is the registered one. The run scripts use it for
        --seed hidden:INDEX (scripts/olympiad/v02_run.sh run_seed); an offline replay of a hidden-seed episode takes
        the pair as RO_SEED / RO_SEED_SALT
    python3 tools/seeds.py show SPORT
        the public registration (no seeds)

The private set never enters git (.gitignore *.hidden_seeds.json; ci_checks/check-hidden-seeds.sh fails on any file of
the private schema in the repository). Python >= 3.10 (tomllib, else tomli), standard library otherwise.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
from core import seeds  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10 (the run hosts' system python)
    import tomli as tomllib


def seeds_dir() -> Path:
    return Path(os.environ.get("RO_SEEDS_DIR") or Path.home() / ".roboolympics" / "seeds")


def private_path(sport: str) -> Path:
    return seeds_dir() / f"{sport}.hidden_seeds.json"


def toml_path(sport: str) -> Path:
    return REPO / "sports" / sport / "sport.toml"


def decl(sport: str) -> dict:
    return tomllib.loads(toml_path(sport).read_text())


def cells(d: dict) -> list[str]:
    """The cells of a sport: per format (a preview format has none), "<format>/L<level>" for each level of a one-side format with levels (its
    own `levels`, else [official] levels when it has a scripted opponent; only those in [official] levels: a warm-up
    level has no official seeds), else "<format>" (as scripts/olympiad/plan.py levels_of)."""
    out = []
    off = (d.get("official") or {}).get("levels")
    for name, f in (d.get("formats") or {}).items():
        if f.get("preview"):                   # a preview format has no official seeds
            continue
        lv = None
        one_side = len(set((f.get("sides") or {s: s for s in f.get("seats", [])}).values())) == 1
        if len(f.get("seats", [])) == 1 or (one_side and f.get("opponent")):   # one side (several seats: team_match)
            lv = f.get("levels") or (off if f.get("opponent") else None)
            if lv and off:
                lv = [x for x in lv if x in off]
        out += [f"{name}/L{int(x)}" for x in lv] if lv else [name]
    return out


def registration(sport: str) -> dict | None:
    """sport.toml [official.hidden_seeds] + the public commitments, checked (ValueError if they disagree)."""
    hs = (decl(sport).get("official") or {}).get("hidden_seeds")
    if not hs:
        return None
    pub = json.loads((REPO / "sports" / sport / seeds.COMMITMENTS).read_text())
    reg = {"sha256": hs.get("sha256"), "counts": dict(hs.get("counts") or {}), "cells": pub.get("cells")}
    seeds.registered(sport, reg)
    if pub.get("sha256") != reg["sha256"] or pub.get("sport") != sport:
        raise ValueError(f"{seeds.COMMITMENTS}: sport or sha256 differs from sport.toml")
    return reg


def load_private(sport: str, path: str | None = None) -> dict:
    p = Path(path) if path else private_path(sport)
    if not p.is_file():
        raise FileNotFoundError(f"the private seed set of {sport} is not on this host ({p}; set RO_SEEDS_DIR)")
    d = json.loads(p.read_text())
    seeds.check_private(d)
    if d["sport"] != sport:
        raise ValueError(f"{p} is the set of {d['sport']}, not {sport}")
    return d


def register(sport: str, pub: dict) -> None:
    """Write [official.hidden_seeds] (sha256, counts) into sport.toml, right after [official] (replacing an old one)."""
    path = toml_path(sport)
    text = path.read_text()
    counts = ", ".join(f'"{c}" = {n}' for c, n in pub["counts"].items())
    block = (f"[official.hidden_seeds]                 # registered hidden seed set (tools/seeds.py; public commitments: "
             f"{seeds.COMMITMENTS})\nsha256 = \"{pub['sha256']}\"\ncounts = {{ {counts} }}\n")
    old = re.search(r"^\[official\.hidden_seeds\][^\n]*\n(?:(?!\[)[^\n]*\n)*", text, re.M)
    if old:
        text = text[:old.start()] + block + "\n" + text[old.end():].lstrip("\n")
    else:
        off = re.search(r"^\[official\][^\n]*\n(?:(?!\[)[^\n]*\n)*", text, re.M)
        if off is None:
            text = text.rstrip("\n") + "\n\n[official]\n\n" + block
        else:
            body = off.group(0).rstrip("\n") + "\n"
            gap = off.group(0)[len(body):]
            text = text[:off.start()] + body + "\n" + block + gap + text[off.end():]
    path.write_text(text)
    tomllib.loads(path.read_text())          # still valid TOML


def from_file(sport: str, data, want: list[str]) -> dict:
    """{cell: [seeds]} from {"<format>": [seeds] | {"<level>": [seeds]}}."""
    out = {}
    for fmt, v in data.items():
        if isinstance(v, dict):
            for lv, ss in v.items():
                out[f"{fmt}/L{int(lv)}"] = [int(x) for x in ss]
        else:
            out[fmt] = [int(x) for x in v]
    extra = sorted(set(out) - set(want))
    if extra:
        raise ValueError(f"cells {extra} are not cells of {sport} ({want})")
    return out


def cmd_new(a) -> int:
    sport = a.sport
    if not toml_path(sport).is_file():
        sys.exit(f"no sports/{sport}/sport.toml")
    dest = private_path(sport)
    if dest.exists() and not a.force:
        sys.exit(f"{dest} exists (a registered set must not be replaced by accident); --force to replace it")
    d = decl(sport)
    want = cells(d)
    created = datetime.date.today().isoformat()
    if a.from_file:
        priv = seeds.from_seeds(sport, from_file(sport, json.loads(Path(a.from_file).read_text()), want), created)
    else:
        if not a.count and not a.counts:
            sys.exit("--count N (per cell), --counts CELL=N,... or --from FILE")
        counts = {c: a.count for c in want} if a.count else {}
        for item in filter(None, (a.counts or "").split(",")):
            c, _, n = item.partition("=")
            if c not in want:
                sys.exit(f"{c} is not a cell of {sport}: {want}")
            counts[c] = int(n)
        priv = seeds.new(sport, counts, avoid=(d.get("official") or {}).get("seeds", []), created=created)
    missing = sorted(set(want) - set(priv["cells"]))
    pub = seeds.public(priv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(dest.parent, 0o700)
    tmp = dest.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(priv, f, indent=1)
        f.write("\n")
    tmp.replace(dest)
    (REPO / "sports" / sport / seeds.COMMITMENTS).write_text(json.dumps(pub, indent=1) + "\n")
    register(sport, pub)
    print(f"{sport}: hidden seeds per cell {pub['counts']}" + (f" (no hidden seeds for {missing})" if missing else ""))
    print(f"  private (keep it on the official run host, back it up, never commit it): {dest}")
    print(f"  public: sports/{sport}/{seeds.COMMITMENTS}, sport.toml [official.hidden_seeds] sha256 = {pub['sha256']}")
    print(f"  next: python3 tools/generate.py {sport}  (the tasks' verifier gets the registration)")
    return 0


def cmd_check(a) -> int:
    reg = registration(a.sport)
    if reg is None:
        print(f"{a.sport}: no hidden seed set registered (sport.toml [official.hidden_seeds])")
        return 0
    print(f"{a.sport}: registration ok ({reg['counts']}, sha256 {reg['sha256']})")
    try:
        priv = load_private(a.sport, a.private)
    except FileNotFoundError as e:
        print(f"  private set: {e}")
        return 0 if not a.private else 1
    ok = seeds.public(priv)["sha256"] == reg["sha256"]
    print(f"  private set: {'the registered one' if ok else 'NOT the registered one (sha256 differs)'}")
    return 0 if ok else 1


def task_cell(task: Path, level) -> tuple[str, str]:
    """(sport, cell) of a generated task's match at a level."""
    c = json.loads((task / "tests" / "contract.json").read_text())
    reg = registration(c["sport"])
    if reg is None:
        raise ValueError(f"{c['sport']}: no hidden seed set registered")
    cell = seeds.cell_of(c["format"], level, reg["cells"])
    if cell is None:
        raise ValueError(f"{c['sport']}: no hidden seeds for format {c['format']} at level {level}")
    return c["sport"], cell


def cmd_resolve(a) -> int:
    if Path(a.target, "tests", "contract.json").is_file():
        sport, cell = task_cell(Path(a.target), a.level)
    else:
        sport, cell = a.target, a.cell
        if not cell:
            sys.exit("resolve SPORT INDEX needs --cell (or give a task directory)")
    reg = registration(sport)
    if reg is None:
        sys.exit(f"{sport}: no hidden seed set registered")
    seed, salt = seeds.resolve(load_private(sport), cell, a.index, reg)
    print(seed, salt)
    return 0


def cmd_show(a) -> int:
    reg = registration(a.sport)
    print(json.dumps({k: reg[k] for k in ("sha256", "counts")} if reg else None))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("sport")
    n.add_argument("--count", type=int)
    n.add_argument("--counts")
    n.add_argument("--from", dest="from_file")
    n.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("sport")
    c.add_argument("--private")
    r = sub.add_parser("resolve")
    r.add_argument("target")
    r.add_argument("index", type=int)
    r.add_argument("--level", type=int)
    r.add_argument("--cell")
    s = sub.add_parser("show")
    s.add_argument("sport")
    a = ap.parse_args(argv)
    try:
        return {"new": cmd_new, "check": cmd_check, "resolve": cmd_resolve, "show": cmd_show}[a.cmd](a)
    except (ValueError, FileNotFoundError, IndexError, KeyError) as e:
        print(f"tools/seeds.py {a.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
