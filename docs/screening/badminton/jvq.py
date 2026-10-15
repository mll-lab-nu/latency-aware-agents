#!/usr/bin/env python3
"""Summarise pulled jv runs: python3 jvq.py RUNDIR [RUNDIR ...] (matches ref_L*_s*.jsonl, B9 b9_*.jsonl)."""
import glob, json, os, sys
import numpy as np


def boot(x, n=3000):
    x = np.asarray(x, float); rng = np.random.default_rng(0)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


for d in sys.argv[1:]:
    print(f"== {d}")
    M = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))) if not os.path.basename(f).startswith("b9")
         for l in open(f) if l.strip()]
    tot = {"steps": 0, "over": 0, "over105": 0, "eps": [], "max": 0, "byj": np.zeros(7), "rf": 0, "n": 0, "ob5": 0}
    for lv in sorted({r["level"] for r in M}):
        R = [r for r in M if r["level"] == lv]
        sh = [r["score"][0] / max(1, sum(r["score"])) for r in R]
        lo, hi = boot(sh)
        st = sum(r["jv"]["steps"] for r in R); ov = sum(r["jv"]["over"] for r in R); o5 = sum(r["jv"]["over105"] for r in R)
        rf = sum(r["jv"].get("n_reflex", 0) for r in R)
        caps = sum(r["hits"] >= 400 for r in R)
        print(f"L{lv}: n {len(R)} wins {sum(r['won'] for r in R)} pts {sum(r['score'][0] for r in R)}:{sum(r['score'][1] for r in R)} "
              f"share {np.mean(sh):.2f} [{lo:.2f}, {hi:.2f}] | over {ov / st:.3%} >1.05x {o5 / st:.3%} max "
              f"{max(r['jv']['max'] for r in R):.2f} | reflex {rf} ({rf / len(R):.2f}/match, matches with any "
              f"{sum(r['jv'].get('n_reflex', 0) > 0 for r in R)}) | speed-limited notes {sum(r['n_speed_lim'] for r in R)}/"
              f"{sum(r['n_acts'] for r in R)} | rally returned {sum(r['rally_returned'] for r in R)}/{sum(r['rally_legs'] for r in R)} "
              f"| hits/pt {sum(r['hits'] for r in R) / max(1, sum(r['n_points'] for r in R)):.1f} caps {caps}")
        for r in R:
            j = r["jv"]; tot["steps"] += j["steps"]; tot["over"] += j["over"]; tot["over105"] += j["over105"]
            tot["eps"] += j["eps"]; tot["max"] = max(tot["max"], j["max"]); tot["byj"] += np.array(j["byj"])
            tot["rf"] += j.get("n_reflex", 0); tot["n"] += 1; tot["ob5"] += j["over_base5"]
    if tot["n"]:
        e = np.array(tot["eps"] or [0])
        print(f"ALL: arm-s {tot['steps'] / 1000:.0f}; over {tot['over'] / tot['steps']:.3%}; >1.05x {tot['over105'] / tot['steps']:.3%}; "
              f"episodes {len(tot['eps'])} (median {np.median(e):.0f} ms, p90 {np.percentile(e, 90):.0f} ms); max {tot['max']:.2f}x; "
              f"by joint {np.round(tot['byj'] / max(1, tot['byj'].sum()), 2).tolist()}; with base acc > 5: "
              f"{tot['ob5'] / max(1, tot['over']):.0%}; reflexes {tot['rf']} in {tot['n']} matches")
    B = [json.loads(l) for f in glob.glob(os.path.join(d, "b9_*.jsonl")) for l in open(f) if l.strip()]
    if B:
        leg = [r for r in B if r.get("legal")]
        tries = [t for r in leg for t in ([r.get("success")] if r.get("success") else [])]
        print(f"B9: serves {len(B)}, legal {len(leg)}, returned {sum(r['returned'] for r in leg)} "
              f"({sum(r['returned'] for r in leg) / max(1, len(leg)):.1%}), clean {sum(r['returned_clean'] for r in leg)}; "
              f"tries {sum(r['n_tries'] for r in leg)}")
