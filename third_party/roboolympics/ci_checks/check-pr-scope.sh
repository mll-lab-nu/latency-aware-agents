#!/usr/bin/env bash
# One kind of change touches one part (docs/framework_zh.md §8, five contribution paths): a pull request changes ONE
# of core/, players/, renderers/, tools/, seasons/, sports/<one sport>/ (with its generated tasks/<sport>-* and
# tasks/dataset.toml), or the legacy tree (scripts/, tasks/, images/, harbor_agents/, ci_checks/). Documentation
# (docs/, README.md, CONTRIBUTING.md) may come with any part. The base is $RO_PR_BASE (the PR's target, e.g.
# origin/main); without it the check is skipped (integration branches touch many parts on purpose). A migration PR
# (one sports/<s>/ part) may also DELETE that sport's legacy code under scripts/olympiad/ (docs §10 step 6).
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
base="${1:-${RO_PR_BASE:-}}"
if [ -z "$base" ]; then echo "pr scope: skipped (set RO_PR_BASE=<target ref> in pull-request CI)"; exit 0; fi
git -C "$root" diff --name-status --no-renames "$(git -C "$root" merge-base "$base" HEAD)" HEAD | python3 -c '
import sys, re
parts, deleted = {}, set()
for line in (l.rstrip("\n") for l in sys.stdin if l.strip()):
    status, f = line.split("\t", 1)
    if status.startswith("D"):
        deleted.add(f)
    top = f.split("/")[0]
    if top in ("docs",) or f in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md"):
        continue
    if top == "sports":
        key = "sports/" + f.split("/")[1]
    elif top in ("core", "players", "renderers", "tools", "seasons"):
        key = top
    elif top == "tasks":
        m = re.match(r"tasks/([a-z0-9]+)-", f)
        key = "sports/" + m.group(1) if m else "legacy"
    else:
        key = "legacy"
    parts.setdefault(key, []).append(f)
sport_tasks = {k for k in parts if k.startswith("sports/")}
own = [k for k in sport_tasks if not any(f.startswith(k + "/") for f in parts[k])]
for k in own:                       # tasks/<x>-* of a sport without sports/<x>/ changes: legacy-generated tasks
    parts.setdefault("legacy", []).extend(parts.pop(k))
sports_parts = [k for k in parts if k.startswith("sports/")]
if len(sports_parts) == 1 and "legacy" in parts:   # a migration: the sport may delete its own legacy code
    name = sports_parts[0].split("/")[1]
    own_legacy = lambda f: f in deleted and f.startswith("scripts/olympiad/") and re.search(rf"(^|/){name}([/._-]|$)", f[len("scripts/olympiad/"):])
    moved = [f for f in parts["legacy"] if own_legacy(f) or f == "tasks/dataset.toml"]
    parts["legacy"] = [f for f in parts["legacy"] if f not in moved]
    parts[sports_parts[0]] += moved
    if not parts["legacy"]:
        parts.pop("legacy")
print("pr scope: parts touched:", ", ".join(f"{k} ({len(v)})" for k, v in sorted(parts.items())) or "none")
if len(parts) > 1:
    print("FAIL [pr scope] a pull request changes one part only (docs may come with it); split it")
    sys.exit(1)
'
