"""Offline: the opponent's decision on each logged shot of ours (opplog: shuttle state and the opponent's position
reaction_s after our hit) under the current rule ('main') or the time-margin rule ('robot': the opponent is a robot
that needs `keep` s of swing after its base arrives; base travel = max(0, distance - reach) at the gantry's speed).
usage: oppsim.py OUT_GLOB... ; prints, per policy/level file, the outcome shares under several parameter sets."""
import glob, json, math, sys
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "sports/badminton/sim"))
import numpy as np
import bm as BM


def decide(e, lv, rule):
    P, V = BM.predict(np.array(e["p"]), np.array(e["v"]))
    last = P[-1]
    if e["kind"] == "our_serve":
        return "serve"
    if not (last[2] <= 1e-9 and BM.in_court(last[0], last[1], +1)):
        return "out"
    o = np.array(e["o"])
    idx = np.arange(9, len(P), 10); j = (idx + 1) // 10
    Pc, sp = P[idx], np.linalg.norm(V[idx], axis=1)
    t = j * BM.CONTROL_DT
    ok = (Pc[:, 1] > 0.3) & (Pc[:, 2] >= BM.OPP_Z[0]) & (Pc[:, 2] <= BM.OPP_Z[1])
    dist = np.hypot(Pc[:, 0] - o[0], Pc[:, 1] - o[1])
    ps = (sp - BM.SPEED_FREE) / (lv["max_speed"] - BM.SPEED_FREE)
    if rule == "main":
        ok &= t + lv["reaction_s"] >= lv["min_t"]
        if not ok.any():
            return "quick"
        pp = np.maximum(0.0, dist - lv["chase"] * t) / lv["reach"]
    else:
        ok &= t + lv["reaction_s"] >= lv.get("min_t", 0.0)
        if not ok.any():
            return "quick"
        travel = np.maximum(0.0, dist - lv["reach"])
        slack = t - travel / lv["chase"] - lv["swing"]
        pp = np.maximum(0.0, lv["window"] - slack) / lv["window"]
    pr = np.where(ok, np.maximum(pp, ps), np.inf)
    pmin = float(pr.min())
    if pmin > 1.0:
        return "miss"
    return "weak" if pmin > BM.PRESSURE_WEAK else ("attack" if pmin < BM.PRESSURE_PASSIVE else "normal")


if __name__ == "__main__":
    sets = {"main1": ("main", BM.LEVELS[1]), "main2": ("main", BM.LEVELS[2])}
    import itertools
    for sw, ch, re_, W in itertools.product((0.4, 0.55, 0.7), (1.5, 2.2), (0.5, 0.8), (0.3,)):
        sets[f"s{sw}c{ch}r{re_}w{W}"] = ("robot", dict(BM.LEVELS[1], swing=sw, chase=ch, reach=re_, window=W, min_t=0.0))
    for f in sys.argv[1:]:
        recs = [json.loads(l) for l in open(f)]
        es = [e for r in recs for e in r["opplog"] if e["kind"] != "our_serve"]
        print(f"== {f.split('/')[-1]} shots {len(es)}")
        for name, (rule, lv) in sets.items():
            c = Counter(decide(e, lv, rule) for e in es)
            n = len(es)
            print(f"  {name:18s} " + " ".join(f"{k}={c[k]/n:.2f}" for k in ("out", "quick", "miss", "weak", "normal", "attack")))
