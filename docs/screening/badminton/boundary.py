"""Core-skill boundary (low level, rule 5), analytic: a racket held still (IK to the contact pose, no swing) sends the
shuttle out at 0.15 max(0, -v_in.n) <= 0.15 x 18 m/s = 2.7 m/s (spec racket_contact; the opponent's shots are <= 18
m/s). For every request of the reference's matches: can ANY launch at <= 2.7 m/s from the contact point land in the
opponent's court? (best case over directions; the real v_in is slower when it arrives)."""
import json, sys, glob
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "sports/badminton/sim"))
import numpy as np, bm as BM
reqs = [s for f in sys.argv[1:] for l in open(f) for s in json.loads(l)["final"]["shots"]]
ok = 0
for s in reqs:
    p = np.array(s["racket_pos"]); good = False
    for th in range(-10, 80, 5):
        for az in range(-40, 41, 10):
            for sp in (1.5, 2.0, 2.7):
                t, a = np.radians(th), np.radians(az)
                v = sp * np.array([np.cos(t) * np.sin(a), np.cos(t) * np.cos(a), np.sin(t)])
                P, _ = BM.predict(p, v)
                L = P[-1]
                cross = np.nonzero((P[:-1, 1] < 0) & (P[1:, 1] >= 0))[0]
                if len(cross) and P[cross[0] + 1, 2] > BM.NET_TOP and L[2] <= 1e-9 and BM.in_court(L[0], L[1], +1):
                    good = True; break
            if good: break
        if good: break
    ok += good
print(f"requests {len(reqs)}; a still racket could land the shuttle in court from {ok} ({ok/len(reqs):.1%})")
