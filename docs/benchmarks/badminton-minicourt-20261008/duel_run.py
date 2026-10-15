"""Is the game playable? Full duel matches, the built-in reference on both seats (perception-free, PerfectRef), under the
variant's physics: points, hits per rally, how points end, serve faults, the depth of the shots (how far from the net
each hit's shuttle lands).   run.sh duel_run.py   (SEEDS=1,2,..; output results/<TAG>/duel.jsonl)"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import study as S

BM, BD, KN = S.BM, S.BD, S.KN


ACTS = []
_orig_apply = S.act_apply


def _apply(layer, m, seat, body):
    r, info = _orig_apply(layer, m, seat, body)
    st = (info or {}).get("stroke") or {}
    if body.get("type") == "act":
        ACTS.append({"k": m.k, "seat": seat, "t": body["t"], "z": body["pos"][2], "head": float(np.linalg.norm(body["vel"])),
                     "err": st.get("blade_error_m"), "nerr": st.get("normal_error_deg"),
                     "head_deliv": None if st.get("racket_vel_delivered") is None else
                     float(np.linalg.norm(st["racket_vel_delivered"])), "stroke": (st.get("stroke") or "")[:120]})
    return r, info


S.act_apply = _apply


def run(seed, first="A"):
    ACTS.clear()
    cfg, m, layer = S.make(seed, first)
    pl = {s: S.PerfectRef(S.seat_spec(cfg, m, s), s, seed) for s in ("A", "B")}
    hits, seen = [], set()

    def tap(mm):
        for e in mm.g.events:
            if e["kind"] == "hit":
                key = (e["k"], e["side"])
                if key not in seen:
                    seen.add(key)
                    L = S.landing(np.array(e["pos"]), np.array(e["v_out"]))
                    # depth of the landing beyond the net in the hitter's frame (A: y > 0 side)
                    depth = None if L is None or L[2] != "floor" else (L[1] if e["side"] == "A" else -L[1])
                    hits.append({"k": e["k"], "side": e["side"], "speed": e["speed"], "racket_speed": e["racket_speed"],
                                 "z": e["pos"][2], "depth": depth, "point": len(mm.g.points)})
        return False
    t0 = time.time()
    S.play(cfg, m, layer, pl, stop=tap, max_steps=60000)
    pts = m.g.points
    return {"seed": seed, "score": m.g.score, "winner": m.g.winner, "n_points": len(pts),
            "hits_per_point": [p["hits"] for p in pts], "how": [p["how"][:90] for p in pts], "hits": hits,
            "k": m.k, "wall_s": round(time.time() - t0, 1), "knobs": KN.describe(), "first": first, "acts": list(ACTS),
            "servers": [p["server"] for p in pts]}


def main():
    tag = os.environ.get("TAG") or KN.VARIANT
    out = S.OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5,6").split(",")]
    firsts = os.environ.get("FIRST", "A").split(",")
    with (out / f"duel_{os.environ.get('SHARD', '0')}.jsonl").open("a") as fh:
        for s, first in [(s, f) for s in seeds for f in firsts]:
            r = run(s, first)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            print(s, r["score"], r["n_points"], np.mean(r["hits_per_point"]) if r["hits_per_point"] else None,
                  r["wall_s"], flush=True)


if __name__ == "__main__":
    main()
