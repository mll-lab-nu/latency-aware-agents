"""Markdown tables of the step-4 suite (one .jsonl per seed): tables_final.py RUNDIR [PREFIX]."""
import glob, json, os, re, sys
import numpy as np
D = sys.argv[1]; PRE = sys.argv[2] if len(sys.argv) > 2 else "s37"
R = {}
for f in glob.glob(os.path.join(D, "*.jsonl")):
    m = re.match(r"(.+)_L(\d)_s(\d+)\.jsonl$", os.path.basename(f))
    if not m:
        continue
    for l in open(f):
        if l.strip():
            r = json.loads(l); R.setdefault((m.group(1), int(m.group(2))), {})[r["seed"]] = r
def boot(x, n=4000):
    x = np.asarray(x); g = np.random.default_rng(0)
    b = [x[g.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
def share(r):
    return r["score"][0] / max(1, sum(r["score"]))
POLS = [("ref", "reference"), ("shortrand", "short-wide, random corner"), ("shortalt", "short-wide, alternating"),
        ("center", "always to the middle"), ("reflex", "return to sender"), ("fixedracket", "fixed racket"),
        ("flatserve", "reference + flat short serves"), ("refnowait", "reference, no waiting move"),
        ("fixed", "fixed act"), ("random", "random act"), ("serveonly", "serve only"), ("tossonly", "toss only"),
        ("nop", "nothing")]
print("| policy | " + " | ".join(f"L{l} wins | L{l} points | L{l} share [95 %] | L{l} cap, hits/pt" for l in (1, 2, 3)) + " |")
print("|---" * 13 + "|")
for p, name in POLS:
    cells = []
    for l in (1, 2, 3):
        rs = list(R.get((f"{PRE}_{p}", l), {}).values())
        if not rs:
            cells += ["-"] * 4; continue
        a = sum(r["score"][0] for r in rs); b = sum(r["score"][1] for r in rs)
        sh = [share(r) for r in rs]; lo, hi = boot(sh)
        cap = sum(r["hits"] >= 400 for r in rs); hpp = sum(r["hits"] for r in rs) / max(1, a + b)
        cells += [f"{sum(r['won'] for r in rs)}/{len(rs)}", f"{a}:{b}", f"{np.mean(sh):.2f} [{lo:.2f}, {hi:.2f}]", f"{cap}/{len(rs)}, {hpp:.1f}"]
    print(f"| {name} | " + " | ".join(cells) + " |")
ABL = [("a1_2", "A1 noise 2 cm"), ("a1_5", "A1 noise 5 cm"), ("a2_k05", "A2 drag x0.5"), ("a2_k0", "A2 no drag"),
       ("a4_half", "A4 half the lead"), ("a4_full", "A4 no lead"), ("b2_rand", "B2 random side"),
       ("b2_fixed", "B2 one fixed corner"), ("waitC3", "waits at (-0.5, -2.1)"), ("nowait", "no waiting move")]
print()
print("| ablation (paired with the reference, same seeds) | L1 share (diff) | L2 share (diff) | L3 share (diff) | pooled paired diff [95 %] |")
print("|---|---|---|---|---|")
for a, name in ABL:
    tag = f"{PRE}_refnowait" if a == "nowait" else f"{PRE}{a}_ref"
    cells = []; D_all = []
    for l in (1, 2, 3):
        ab = R.get((tag, l), {}); rf = R.get((f"{PRE}_ref", l), {})
        seeds = sorted(set(ab) & set(rf))
        if not seeds:
            cells.append("-"); continue
        d = [share(ab[s]) - share(rf[s]) for s in seeds]; D_all += d
        cells.append(f"{np.mean([share(ab[s]) for s in seeds]):.2f} ({np.mean(d):+.2f}, n {len(seeds)})")
    if D_all:
        lo, hi = boot(D_all)
        print(f"| {name} | " + " | ".join(cells) + f" | {np.mean(D_all):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
