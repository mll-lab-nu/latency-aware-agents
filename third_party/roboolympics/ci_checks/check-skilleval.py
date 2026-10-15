#!/usr/bin/env python3
"""The two modes (docs/framework_zh.md §6.3, §9 robot and mode rules; skilleval/README.md). Static, standard library only.

M1  every sport that declares [modes.high] / [modes.low] passes skilleval/declare.py: the skill API and its contract
    module, the toolkit manifest (schema and sha256 of every tool), [seat.low] ships exactly the manifest's tools
    and the public practice requests (no forbidden module, no platform implementation, no hidden request), the
    deliverable (path under /app/output/skill/, signature, time limits), the evaluator hooks, the reference and
    boundary skills, the thresholds, hidden and public requests disjoint
M2  a generated task's simulator directory carries skilleval/ and leaves out the hidden requests (tools/sportlib.py)
M3  skilleval/ names no sport (it is shared, like the core) and its host-side modules (declare, toolkit, contract,
    cases, host, and sandbox / skillhost, which the skill containers run) import only the standard library and each
    other

  python3 ci_checks/check-skilleval.py        exit 1 on a failure
"""

from __future__ import annotations

import ast
import re
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "skilleval"), str(ROOT / "tools")]
import declare  # noqa: E402
import sportlib  # noqa: E402

STDLIB_ONLY = ("declare", "toolkit", "contract", "cases", "host", "sandbox", "skillhost")   # skillhost + sandbox +
# host run in the seat's skill container (the skill in the loop) without the simulator or the core


def main() -> int:
    bad, n = [], 0
    sports = sorted(p.parent for p in (ROOT / "sports").glob("*/sport.toml"))
    for sd in sports:
        d = tomllib.loads((sd / "sport.toml").read_text())
        if not declare.declared(d):
            continue
        n += 1
        bad += [f"M1 {sd.name}: {name}: {det}" for name, ok, det in declare.check(sd.name, d, sd) if not ok]
        with tempfile.TemporaryDirectory() as tmp:
            sim = sportlib.assemble_sim(sd.name, Path(tmp) / "sim")
            if not (sim / "skilleval" / "contract.py").is_file():
                bad.append(f"M2 {sd.name}: the simulator directory has no skilleval/")
            hc = sportlib.hidden_cases(d)
            if hc and (sim / "sport" / hc).exists():
                bad.append(f"M2 {sd.name}: the generated simulator directory holds the hidden requests {hc}")
    names = {p.name for p in sports}
    words = names | {"table tennis", "air hockey", "disc golf", "table-tennis", "air-hockey"}
    pat = re.compile(
        r"(?<![A-Za-z_])(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")(?![A-Za-z_])", re.I
    )
    for p in sorted((ROOT / "skilleval").rglob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            m = pat.search(line)
            if m:
                bad.append(f"M3 {p.relative_to(ROOT)}:{i}: names the sport {m.group(1)!r}")
        if p.stem in STDLIB_ONLY and p.parent.name == "skilleval":
            tree = ast.parse(p.read_text())
            mods = {a.name.split(".")[0] for x in ast.walk(tree) if isinstance(x, ast.Import) for a in x.names}
            mods |= {
                x.module.split(".")[0]
                for x in ast.walk(tree)
                if isinstance(x, ast.ImportFrom) and x.module and x.level == 0
            }
            extra = sorted(mods - set(sys.stdlib_module_names) - {"skilleval", *STDLIB_ONLY})
            if extra:
                bad.append(f"M3 {p.relative_to(ROOT)}: imports {extra} (standard library only)")
    for b in bad:
        print(f"FAIL [skilleval] {b}")
    print(f"skilleval: {n} sport(s) declare the two modes, {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
