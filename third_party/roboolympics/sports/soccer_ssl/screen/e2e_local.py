"""End-to-end check of the camera reference (step 4, "端到端检查"): the built-in reference player (builtin_players
PLAYERS["reference"]: own robots' state + the overhead camera images only) plays a match against a scripted level
through the real plugin and core's in-process loop (core/local.py: the whitelisted observation, the camera tiles drawn
by core's renderer, the action layer). Halves can be shortened for the check (--half).

    MUJOCO_GL=osmesa python3 sports/soccer_ssl/screen/e2e_local.py --level 2 --seed 1 --half 60 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPORT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(SPORT))
for d in (ROOT, SPORT):
    if d not in sys.path:
        sys.path.insert(0, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--half", type=float, default=60.0)
    ap.add_argument("--player", default="reference")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from core import local
    from core.episode import load_plugin
    from core.sport import MatchConfig

    plugin = os.path.join(SPORT, "plugin.py")
    sport = load_plugin(plugin)
    cfg = MatchConfig(sport=sport.name, format="match", seats=("A",), seed=a.seed, level=a.level, action_level="high")
    build = sport.build

    def build_short(c):
        m = build(c)
        m.w.referee.half_s = a.half
        return m

    sport.build = build_short
    spec = local.seat_spec(sport, cfg, "A", plugin=plugin)
    player = sport.players()[a.player](spec, "A", a.seed)
    t0 = time.time()
    res = local.play(sport, cfg, {"A": player}, data=a.out, render=True, plugin=plugin, every=a.every)
    out = dict(level=a.level, seed=a.seed, half=a.half, wall_s=round(time.time() - t0, 1),
               **{k: res.get(k) for k in ("scores", "winner", "decided_by", "k")},
               stats=dict(player.team.stats))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "e2e.json"), "w") as f:
        json.dump(out, f, default=str)
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
