"""Markdown tables of the step-4 runs of the final configuration (v12 = levels 1-2, v13 = level 3, same code for them)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summ

PRE = {1: "v16", 2: "v16", 3: "v16"}
POLS = ["ref", "shortalt", "center", "reflex", "fixedracket", "fixed", "random", "serveonly", "tossonly", "nop"]
ABLS = [("a1_2", "A1 noise 2 cm"), ("a1_5", "A1 noise 5 cm"), ("a2_k05", "A2 drag x0.5"), ("a2_k0", "A2 no drag"),
        ("a4_half", "A4 half the lead"), ("a4_full", "A4 no lead"), ("b2_rand", "B2 random side"),
        ("b2_fixed", "B2 one fixed corner")]


def tag(l, p, a=None):
    return f"{PRE[l]}_{p}" + (f"-{a}" if a else "") + f"_L{l}"


def share(t):
    return {json.loads(x)["seed"]: (lambda r: r["score"][0] / max(1, sum(r["score"])))(json.loads(x))
            for x in open(os.path.join(summ.OUT, t + ".jsonl"))}


def boot(d, n=5000):
    rng = np.random.default_rng(0)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


out = []
out.append("| policy | " + " | ".join(f"L{l} wins | L{l} points | L{l} share" for l in (1, 2, 3)) + " |")
out.append("|---|" + "---|---|---|" * 3)
for p in POLS:
    cells = []
    for l in (1, 2, 3):
        r = summ.row(tag(l, p))
        cells += ["-", "-", "-"] if r is None else [f"{r['wins']}/{r['n']}", f"{r['pf']}:{r['pa']}",
                                                     f"{r['share']:.2f} [{r['share_lo']:.2f}, {r['share_hi']:.2f}]"]
    out.append(f"| {p} | " + " | ".join(cells) + " |")
out.append("")
out.append("| ablation (two strengths) | " + " | ".join(f"L{l} share (diff vs ref)" for l in (1, 2, 3)) +
           " | pooled paired diff [95 % CI] |")
out.append("|---|---|---|---|---|")
for a, name in ABLS:
    cells, D = [], []
    for l in (1, 2, 3):
        try:
            base, abl = share(tag(l, "ref")), share(tag(l, "ref", a))
        except OSError:
            cells.append("-"); continue
        d = [abl[s] - base[s] for s in base if s in abl]
        D += d
        cells.append(f"{np.mean(list(abl.values())):.2f} ({np.mean(d):+.2f}, n {len(d)})")
    D = np.array(D)
    ci = boot(D) if len(D) else (np.nan, np.nan)
    out.append(f"| {name} | " + " | ".join(cells) + f" | {D.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
print("\n".join(out))
