"""Summarise per-seed results: qsum.py RUNDIR [RUNDIR...] -> per (prefix, policy, level): wins, points, share
[bootstrap 95 %], matches ending at the hit cap (CAP, default 400), hits per point."""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
rows = defaultdict(list)
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        m = re.match(r"(.+)_L(\d)_s(\d+)\.jsonl$", os.path.basename(f))
        if not m:
            continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); rows[(m.group(1), int(m.group(2)))].append(r)
def boot(x, n=4000):
    x = np.asarray(x); g = np.random.default_rng(0)
    b = [x[g.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
for (k, lv), R in sorted(rows.items()):
    R.sort(key=lambda r: r["seed"])
    a = sum(r["score"][0] for r in R); b = sum(r["score"][1] for r in R)
    sh = [r["score"][0] / max(1, sum(r["score"])) for r in R]
    lo, hi = boot(sh)
    cap = sum(r["hits"] >= int(os.environ.get("CAP", "400")) for r in R)
    hpp = sum(r["hits"] for r in R) / max(1, a + b)
    print(f"{k:24s} L{lv}  wins {sum(r['won'] for r in R)}/{len(R)}  pts {a}:{b}  share {np.mean(sh):.2f} [{lo:.2f},{hi:.2f}]"
          f"  cap {cap}/{len(R)}  hits/pt {hpp:.1f}  scores " + " ".join(f"{r['score'][0]}:{r['score'][1]}" for r in R))

if __name__ == "__main__" and os.environ.get("KILLS"):
    print("\nkill attempts (our contacts >= 1.9 m), delivered, per match; serve faults")
    for (k, lv), R in sorted(rows.items()):
        att = sum(1 for r in R for s in r["final"]["shots"] if s.get("racket_pos") and s["racket_pos"][2] >= 1.9)
        ok = sum(1 for r in R for s in r["final"]["shots"] if s.get("racket_pos") and s["racket_pos"][2] >= 1.9 and s.get("stroke") == "ok")
        sf = sum(h.startswith("serve_fault") for r in R for h in r["how"])
        print(f"{k:24s} L{lv}  kills {att} ({att/len(R):.1f}/match), delivered {ok}; serve faults {sf}")
