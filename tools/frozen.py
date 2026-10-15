#!/usr/bin/env python3
"""Which sports are frozen (official): a sport is frozen iff its sport.toml has a registered hidden official seed set,
[official.hidden_seeds] (sha256 + counts per cell), and the public commitments sports/<sport>/seed_commitments.json
hash to that sha256 with those counts (core/seeds.py registered(); CONTRIBUTING.md "Freeze a sport"). The official part
of a release takes only frozen sports; the others are in screening. Nothing here reads a private seed set.

    python3 tools/frozen.py [SPORT ...]            a table: status, publish, frozen, revision, the registered cells
    python3 tools/frozen.py --json [SPORT ...]     the same as JSON (schema ro-frozen/1), for release tooling
    python3 tools/frozen.py --names                the frozen sports' names, one per line

Per sport: [sport] status, publish and revision (the version string; the generated tasks' task.toml [metadata]
revision next to it, which `tools/generate.py check` keeps equal), the declared formats and [official] levels, and for
a frozen sport the registered cells ("<format>/L<level>" or "<format>") with their seed counts, also grouped as
format -> levels. A registration that does not verify (commitments missing, another sha256, other counts) is not
frozen: the row says why and the exit status is 1. Python >= 3.10 (tomllib, else tomli), standard library otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
from core import seeds  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

SCHEMA = "ro-frozen/1"


def by_format(cells) -> dict:
    """{"match/L2": n, "duel": n} -> {"match": [2], "duel": None}: the formats of the cells, with their levels."""
    out: dict = {}
    for c in sorted(cells):
        fmt, _, lv = c.partition("/L")
        if lv:
            out[fmt] = sorted((out.get(fmt) or []) + [int(lv)])
        else:
            out.setdefault(fmt, None)
    return out


def registration(root: Path, sport: str, d: dict) -> tuple[dict | None, str | None]:
    """(the verified registration {sha256, counts}, None), (None, None) when none is declared, or (None, why not)."""
    hs = (d.get("official") or {}).get("hidden_seeds")
    if not hs:
        return None, None
    path = root / "sports" / sport / seeds.COMMITMENTS
    if not path.is_file():
        return None, f"[official.hidden_seeds] declared but sports/{sport}/{seeds.COMMITMENTS} is missing"
    try:
        pub = json.loads(path.read_text())
        reg = {"sha256": hs.get("sha256"), "counts": dict(hs.get("counts") or {}), "cells": pub.get("cells")}
        seeds.registered(sport, reg)          # the commitments hash to the registered sha256, counts per cell agree
    except (ValueError, TypeError, AttributeError) as e:
        return None, str(e)
    if pub.get("sport") != sport or pub.get("sha256") != reg["sha256"]:
        return None, f"{seeds.COMMITMENTS}: sport or sha256 differs from sport.toml"
    return {"sha256": reg["sha256"], "counts": {c: int(n) for c, n in sorted(reg["counts"].items())}}, None


def task_revision(root: Path, tid: str) -> str | None:
    p = root / "tasks" / tid / "task.toml"
    if not p.is_file():
        return None
    return (tomllib.loads(p.read_text()).get("metadata") or {}).get("revision")


def status(root: Path, sport: str) -> dict:
    d = tomllib.loads((root / "sports" / sport / "sport.toml").read_text())
    s = d.get("sport") or {}
    reg, problem = registration(root, sport, d)
    publish = bool(s.get("publish", True))
    return {
        "sport": sport,
        "title": s.get("title"),
        "status": s.get("status", "sport"),
        "publish": publish,
        "frozen": reg is not None,
        "revision": s.get("revision"),
        "formats": list(d.get("formats") or {}),
        "official_levels": (d.get("official") or {}).get("levels"),
        "tasks": [{"id": t.get("id"), "format": t.get("format"),
                   "revision": task_revision(root, t.get("id", "")) if publish else None}
                  for t in d.get("tasks") or []],
        "hidden_seeds": None if reg is None else {**reg, "formats": by_format(reg["counts"])},
        "problem": problem,
    }


def sports(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / "sports").glob("*/sport.toml"))


def commit(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def report(root: Path, names=None) -> dict:
    rows = [status(root, s) for s in (names or sports(root))]
    return {"schema": SCHEMA, "commit": commit(root), "frozen": [r["sport"] for r in rows if r["frozen"]],
            "screening": [r["sport"] for r in rows if not r["frozen"] and r["publish"]], "sports": rows}


def cells_text(r: dict) -> str:
    if r["problem"]:
        return f"NOT FROZEN, registration invalid: {r['problem']}"
    if not r["frozen"]:
        return "-"
    return ", ".join(f"{c}={n}" for c, n in r["hidden_seeds"]["counts"].items())


def table(rep: dict) -> str:
    rows = [("sport", "status", "publish", "frozen", "revision", "hidden seed cells")]
    rows += [(r["sport"], r["status"], "yes" if r["publish"] else "no", "yes" if r["frozen"] else "no",
              r["revision"] or "?", cells_text(r)) for r in rep["sports"]]
    w = [max(len(x[i]) for x in rows) for i in range(4)]
    lines = ["  ".join(x[i].ljust(w[i]) for i in range(4)) + "  " + x[4] + "  " + x[5] for x in rows]
    lines.append(f"frozen ({len(rep['frozen'])}): {' '.join(rep['frozen']) or '-'}")
    lines.append(f"screening ({len(rep['screening'])}): {' '.join(rep['screening']) or '-'}")
    return "\n".join(lines)


def main(argv, root: Path = REPO) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sport", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--names", action="store_true")
    a = ap.parse_args(argv)
    missing = [s for s in a.sport if not (root / "sports" / s / "sport.toml").is_file()]
    if missing:
        print(f"tools/frozen.py: no sports/<sport>/sport.toml for {missing}", file=sys.stderr)
        return 2
    rep = report(root, a.sport)
    if a.json:
        print(json.dumps(rep, indent=1))
    elif a.names:
        print("\n".join(rep["frozen"]))
    else:
        print(table(rep))
    return 1 if any(r["problem"] for r in rep["sports"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
