"""Tables of a batch_local.py directory: per variant and level, wins / draws / losses of the blue team (the variant),
goals, goal difference, fouls, passes."""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*", "L*_s*.json")):
        try:
            rows.append(json.load(open(f)))
        except Exception:  # noqa: BLE001
            pass
    return rows


def table(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["variant"], r["level"])].append(r)
    lines = ["variant level n W-D-L goals_for goals_against mean_gd fouls_for fouls_against passes(ok) redirects"]
    for (v, lv), rs in sorted(g.items()):
        W = sum(r["score"]["blue"] > r["score"]["yellow"] for r in rs)
        L = sum(r["score"]["blue"] < r["score"]["yellow"] for r in rs)
        D = len(rs) - W - L
        gf = sum(r["score"]["blue"] for r in rs)
        ga = sum(r["score"]["yellow"] for r in rs)
        ff = sum(r["fouls_total"]["blue"] for r in rs)
        fa = sum(r["fouls_total"]["yellow"] for r in rs)
        ps = sum(r["stats"]["blue"].get("passes", 0) for r in rs)
        po = sum(r["stats"]["blue"].get("pass_ok", 0) for r in rs)
        rd = sum(r["stats"]["blue"].get("redirects", 0) for r in rs)
        lines.append(f"{v} L{lv} {len(rs)} {W}-{D}-{L} {gf} {ga} {(gf - ga) / len(rs):+.2f} {ff} {fa} {ps}({po}) {rd}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(load(sys.argv[1])))
