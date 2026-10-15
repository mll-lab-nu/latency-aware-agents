"""Fairness of results/<tag>/duel_*.jsonl (host side): first-server and seat advantage with 95 % intervals (normal
approximation, as tools/check stage 4), points won by the server, serve faults.   python3 fair_sum.py tag"""
import glob, json, math, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
tag = sys.argv[1]
recs = {}
for f in sorted(glob.glob(str(R / tag / "duel_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        recs[(r["seed"], r.get("first", "A"))] = r
n = len(recs)
win = {k: ("A" if r["score"]["A"] > r["score"]["B"] else "B") for k, r in recs.items()}
fs = sum(1 for (s, f), w in win.items() if w == f)
sa = sum(1 for w in win.values() if w == "A")
def ci(k, n):
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return round(100 * (p - 0.5), 1), round(100 * h, 1)
won_srv = tot = 0
for k, r in recs.items():
    sv = r.get("servers") or []
    for i in range(len(sv)):
        w = sv[i + 1] if i + 1 < len(sv) else win[k]          # rally scoring: the winner serves next
        if "let:" in r["how"][i]:
            continue
        tot += 1
        won_srv += w == sv[i]
mirror = sum(1 for s in {k[0] for k in recs} if (s, "A") in recs and (s, "B") in recs
             and recs[(s, "A")]["score"] == {"A": recs[(s, "B")]["score"]["B"], "B": recs[(s, "B")]["score"]["A"]})
out = {"N": n, "first_server_wins": fs, "first_server_adv_pp": ci(fs, n), "seat_A_wins": sa, "seat_adv_pp": ci(sa, n),
       "server_won_points": [won_srv, tot], "mirror_pairs": mirror}
print(json.dumps(out))
(R / f"fair_{tag}.json").write_text(json.dumps(out, indent=1))
