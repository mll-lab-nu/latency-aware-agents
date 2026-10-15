"""Format match-wallclock: the reference (perception-free) against the scripted opponent at a level: score, hits per
point, how points end.   run.sh match_run.py   (LEVEL=1|2, SEEDS=...; results/<TAG>/match.jsonl)"""
from __future__ import annotations

import json
import os
import time

import study as S

PL, BM = S.PL, S.BM
from core.sport import MatchConfig  # noqa: E402


def run(level, seed):
    cfg = MatchConfig(sport="badminton", format="match-wallclock", seats=("A",), seed=int(seed), level=int(level),
                      slowdown=20)
    m = PL.SPORT.build(cfg)
    layer = PL.SPORT.action_layer("high")
    pl = {"A": S.PerfectRef(S.seat_spec(cfg, m, "A"), "A", seed)}
    t0 = time.time()
    S.play(cfg, m, layer, pl, max_steps=80000)
    g = m.g
    return {"level": level, "seed": seed, "score": list(g.score), "won": g.winner == 0, "n_points": len(g.points),
            "hits": g.hits, "how": [p["how"][:90] for p in g.points], "wall_s": round(time.time() - t0, 1)}


def main():
    out = S.OUT / (os.environ.get("TAG") or "match")
    out.mkdir(parents=True, exist_ok=True)
    level = int(os.environ.get("LEVEL", "1"))
    with (out / f"match_L{level}_{os.environ.get('SHARD', '0')}.jsonl").open("a") as fh:
        for s in [int(x) for x in os.environ.get("SEEDS", "101").split(",")]:
            r = run(level, s)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["level"], r["seed"], r["score"], r["won"], r["n_points"], r["hits"], r["wall_s"], flush=True)


if __name__ == "__main__":
    main()
