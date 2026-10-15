#!/usr/bin/env python3
"""Table tennis: where the points come from in a tools/screen batch (stdlib only).

    python3 sports/tabletennis/screen/analyze.py DIR [--variants a,b] [--json out.json]

Per variant and level: matches, wins, points won / lost split by how they ended (the referee's `how` text):
  won  serve_short    the opponent met our service too close to the net ("too short")
       wide / fast    our ball passed out of its reach ("too wide") / too fast for it
       opp_fault      the opponent's return was out / into the net / it could not play a legal return
  lost own_fault      our ball out, into our own half, into the net
       miss           no paddle contact
       serve_fault    our failed service (serve_fault:*)
and the self-collision record (events, force warnings above 20 N, the pairs).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def kind(p: dict) -> str:
    how = (p.get("how") or "").lower()
    if p.get("winner") == "you":
        if "too short" in how:
            return "won:serve_short"
        if "too wide" in how:
            return "won:wide"
        if "too fast" in how:
            return "won:fast"
        if "serve_fault" in how:
            return "won:opp_serve_fault"
        return "won:opp_fault"
    if "serve_fault" in how:
        return "lost:serve_fault"
    if "expedite" in how:
        return "lost:expedite"
    if "missed" in how or "no paddle contact" in how:
        return "lost:miss"
    if "after your hit" in how or "your" in how:
        return "lost:own_fault"
    return "lost:other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    tab = collections.defaultdict(lambda: collections.Counter())
    sc = collections.defaultdict(lambda: collections.Counter())
    pairs = collections.defaultdict(collections.Counter)
    for f in sorted(Path(a.data, "runs").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        v, lv = r["variant"], r.get("level")
        if a.variants and v not in a.variants.split(","):
            continue
        ff = r["final_fields"]
        t = tab[(v, lv)]
        t["matches"] += 1
        t["wins"] += bool(ff.get("won"))
        for p in ff.get("points") or []:
            t[kind(p)] += 1
        s = (ff.get("self_collision") or {}).get("A") or {}
        sc[v]["matches"] += 1
        sc[v]["events"] += int(s.get("events") or 0)
        sc[v]["warnings"] += int(s.get("warnings") or 0)
        sc[v]["matches_with_event"] += bool(s.get("events"))
        for w in s.get("worst_pairs") or []:
            pairs[v]["x".join(w["geoms"])] += int(w["events"])
    cols = ["matches", "wins", "won:serve_short", "won:wide", "won:fast", "won:opp_fault", "won:opp_serve_fault",
            "lost:own_fault", "lost:miss", "lost:serve_fault", "lost:expedite", "lost:other"]
    print("| variant | level | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for (v, lv), t in sorted(tab.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        print(f"| {v} | L{lv} | " + " | ".join(str(t[c]) for c in cols) + " |")
    print("\n| variant | matches | matches with a self-collision event | events | warnings (> 20 N) | pairs |\n|---|---|---|---|---|---|")
    for v, s in sorted(sc.items()):
        print(f"| {v} | {s['matches']} | {s['matches_with_event']} | {s['events']} | {s['warnings']} | "
              + ", ".join(f"{k} {n}" for k, n in pairs[v].most_common(4)) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps({"points": {f"{v}@L{lv}": dict(t) for (v, lv), t in tab.items()},
                                            "self_collision": {v: dict(s) for v, s in sc.items()},
                                            "pairs": {v: dict(p) for v, p in pairs.items()}}, indent=1))


if __name__ == "__main__":
    main()
