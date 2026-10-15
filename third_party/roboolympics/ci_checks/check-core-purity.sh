#!/usr/bin/env bash
# Core rule 1 (docs/framework_zh.md §9): core/ names no sport and no harness. Sport names: every sports/<sport>/ and
# the names written apart ("table tennis", "air hockey", ...);
# harness names: the `harness` values of players/players.toml and the harness words below.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$root" <<'PY'
import re, sys, tomllib
from pathlib import Path
root = Path(sys.argv[1])
sports = {p.parent.name for p in root.glob("sports/*/sport.toml")}
words = set(sports)
for s in sports:                                       # tabletennis -> "table tennis", "table-tennis", "table_tennis"
    for a, b in (("table", "tennis"), ("air", "hockey"), ("disc", "golf")):
        if s == a + b:
            words |= {f"{a} {b}", f"{a}-{b}", f"{a}_{b}"}
plist = tomllib.loads((root / "players/players.toml").read_text()).get("player", [])
harness = {p["harness"] for p in plist if p.get("harness") not in (None, "builtin")}
words |= harness | {h.replace("-", " ") for h in harness} | {"claude", "codex", "gemini", "opencode", "aider", "cursor"}
pat = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")(?![A-Za-z])", re.I)
bad = []
for p in sorted((root / "core").rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".md", ".toml", ".json", ".txt", ""):
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for m in pat.finditer(line):
                bad.append(f"{p.relative_to(root)}:{i}: {m.group(1)!r}")
for b in bad:
    print(f"FAIL [core purity] {b}")
print(f"core purity: {len(bad)} sport/harness name(s) in core/ ({len(words)} names checked)")
sys.exit(1 if bad else 0)
PY
