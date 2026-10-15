#!/usr/bin/env bash
# Hidden official seeds (core 0.2.14, core/seeds.py, tools/seeds.py; docs/framework_zh.md §6.4): no private seed set in
# the repository (any JSON file of the private schema fails, wherever it is), and every sport that registers a set
# (sport.toml [official.hidden_seeds]) has public commitments (sports/<sport>/seed_commitments.json) that hash to the
# registered sha256 and count.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$root" <<'PY'
import json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
bad = []
files = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"], capture_output=True,
                       text=True).stdout.split()
for f in files:
    p = root / f
    if p.suffix == ".json" and p.is_file() and p.stat().st_size < 50_000_000:
        try:
            if json.loads(p.read_text(errors="replace")).get("schema") == "ro-hidden-seeds-private/1":
                bad.append(f"{f}: a private seed set (keep it outside the repository: tools/seeds.py)")
        except (ValueError, AttributeError):
            pass
n = 0
for t in sorted(root.glob("sports/*/sport.toml")):
    if "[official.hidden_seeds]" in t.read_text():
        n += 1
        r = subprocess.run([sys.executable, str(root / "tools/seeds.py"), "show", t.parent.name], capture_output=True,
                           text=True)
        if r.returncode != 0:
            bad.append(f"sports/{t.parent.name}: {r.stderr.strip()[-300:]}")
for b in bad:
    print(f"FAIL [hidden seeds] {b}")
print(f"hidden seeds: {n} sport(s) with a registered set, {len(bad)} problem(s)")
sys.exit(1 if bad else 0)
PY
