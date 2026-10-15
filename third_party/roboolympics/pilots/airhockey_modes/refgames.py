"""Reference-vs-reference air hockey duels (the camera reference on both seats, core/local.py as the fairness check
plays them): goals per game, 0:0 games, own goals. Measures the reference fixes and the real limits (pilot doc §1).

    python3 refgames.py ROOT OUT.jsonl --seeds 1 10 [--first A B] [--workers 1]

ROOT is a checkout (core/, sports/airhockey/, bodies/, renderers/): the code version to measure."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def one(root, seed, first):
    sys.path.insert(0, root)
    from core import local
    from core.episode import load_plugin
    from core.sport import MatchConfig
    plugin = f"{root}/sports/airhockey/plugin.py"
    sport = load_plugin(plugin)
    cfg = MatchConfig(sport=sport.name, format="duel", seats=("A", "B"), seed=seed, first=first,
                      action_level=sport_level(plugin))
    t0 = time.time()
    r = local.play(sport, cfg, local.make_players(sport, cfg, {"A": "reference", "B": "reference"}, plugin=plugin),
                   plugin=plugin)
    ff = r["final_fields"]
    res = ff["results"]
    goals = [x for x in res if x["how"] == "goal"]
    return {"seed": seed, "first": first, "revision": ff.get("hockey_revision"), "scores": r["scores"],
            "winner": r["winner"], "decided_by": r["decided_by"], "goals": len(goals),
            "own_goals": sum(bool(x.get("own_goal")) for x in goals), "rallies": len(res),
            "fouls": sum(x["how"] == "seven_second_foul" for x in res),
            "dead": sum(x["how"] == "dead_puck" for x in res), "t_play": r["k"] * 0.01,
            "zero_zero": len(goals) == 0, "cpu_s": round(time.time() - t0, 1), "digest": r["digest"]}


def sport_level(plugin):
    import tomllib
    d = tomllib.loads((Path(plugin).parent / "sport.toml").read_text())
    return (d.get("modes") or {}).get("action_levels", ["low"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--seeds", nargs=2, type=int, default=[1, 10])
    ap.add_argument("--first", nargs="+", default=["A", "B"])
    a = ap.parse_args()
    done = set()
    if Path(a.out).exists():
        for line in Path(a.out).read_text().splitlines():
            d = json.loads(line)
            done.add((d["seed"], d["first"]))
    for seed in range(a.seeds[0], a.seeds[1] + 1):
        for first in a.first:
            if (seed, first) in done:
                continue
            row = one(a.root, seed, first)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
