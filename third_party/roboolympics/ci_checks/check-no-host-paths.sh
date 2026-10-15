#!/usr/bin/env bash
# Tool rule 2 and player rule 4 (docs/framework_zh.md §9): no host paths, machine names or secrets in the v0.2 parts
# (core/, sports/, players/, renderers/, tools/, seasons/). Container paths (/srv, /data, /app, /tests, /solution) are
# fine; a home directory, a user's data directory or a key-shaped string is not. Configuration of this machine lives
# outside the repository (environment variables).
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$root" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
HOST = re.compile(r"(/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|/mnt/[A-Za-z0-9_.-]+/|/scratch/|/data/(?!verifier\b|views\b)[a-z][a-z0-9_-]*/[A-Za-z0-9_.-]+/)")
KEY = re.compile(r"(sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[bp]-[A-Za-z0-9-]{20,}|eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{20,}\.)")
bad = []
for part in ("core", "sports", "players", "renderers", "tools", "seasons"):
    for p in sorted((root / part).rglob("*")) if (root / part).is_dir() else []:
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".gz", ".npz", ".png", ".jpg", ".mp4"):
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for rx, what in ((HOST, "host path"), (KEY, "secret-shaped string")):
                m = rx.search(line)
                if m and not (what.startswith("secret") and "placeholder" in line):   # documented dummy values
                    bad.append(f"{p.relative_to(root)}:{i}: {what} {m.group(0)[:24]!r}")
for b in bad:
    print(f"FAIL [no host paths/secrets] {b}")
print(f"no host paths/secrets: {len(bad)} finding(s) in the v0.2 parts")
sys.exit(1 if bad else 0)
PY
