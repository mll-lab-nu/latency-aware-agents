#!/usr/bin/env bash
# Sport rule 10 (docs/framework_zh.md §9): a sport imports only the core's public modules, the renderers' interface,
# its own files and third-party libraries; never another sport, the players, the tools or a legacy sport's sources.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$root" <<'PY'
import ast, sys
from pathlib import Path
root = Path(sys.argv[1])
sports = sorted(p.parent for p in root.glob("sports/*/sport.toml"))
names = {p.name for p in sports}
banned_top = {"sports", "players", "tools", "scripts", "harbor_agents", "tasks", "ci_checks"}
bad = []
for sd in sports:
    own = {p.stem for p in sd.rglob("*.py")}
    for p in sorted(sd.rglob("*.py")):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for m in mods:
                top = m.split(".")[0]
                if top in banned_top or (top in names and top not in own) or (top == "renderers" and m not in (
                        "renderers", "renderers.base")):
                    bad.append(f"{p.relative_to(root)}:{n.lineno}: import {m}")
for b in bad:
    print(f"FAIL [sport isolation] {b}")
print(f"sport isolation: {len(sports)} sport(s), {len(bad)} forbidden import(s)")
sys.exit(1 if bad else 0)
PY
