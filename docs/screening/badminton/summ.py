"""Summaries of screening runs: per tag (file), wins / n with Wilson 95 % interval, points for:against, the point share
(points for / all points) with a bootstrap interval over matches, rally returns, act notes, self-collision.
usage: summ.py PATTERN... (globs under out/)"""
import glob, json, math, os, sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(tag):
    """The tag's matches; a tag v16_X also takes v16b_X (the second ten seeds)."""
    out = []
    for t in [tag] + ([tag.replace("v16_", "v16b_", 1)] if tag.startswith("v16_") else []):
        f = os.path.join(OUT, t + ".jsonl")
        if os.path.exists(f):
            out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def row(tag):
    R = load(tag)
    if not R:
        return None
    n = len(R); w = sum(r["won"] for r in R)
    pf = sum(r["score"][0] for r in R); pa = sum(r["score"][1] for r in R)
    share = np.array([r["score"][0] / max(1, sum(r["score"])) for r in R])
    rng = np.random.default_rng(0)
    bs = [share[rng.integers(0, n, n)].mean() for _ in range(2000)]
    lo, hi = wilson(w, n)
    rl = sum(r["rally_legs"] for r in R); rr = sum(r["rally_returned"] for r in R)
    sc = sum((r.get("selfcol") or {}).get("A", {}).get("events", 0) for r in R)
    return dict(tag=tag, n=n, wins=w, win_lo=round(lo, 2), win_hi=round(hi, 2), pf=pf, pa=pa,
                share=round(float(share.mean()), 3), share_lo=round(float(np.percentile(bs, 2.5)), 3),
                share_hi=round(float(np.percentile(bs, 97.5)), 3), rally_ret=f"{rr}/{rl}",
                hits_per_pt=round(sum(r["hits"] for r in R) / max(1, sum(r["n_points"] for r in R)), 2),
                base_short=sum(r["n_base_short"] for r in R), speed_lim=sum(r["n_speed_lim"] for r in R),
                acts=sum(r["n_acts"] for r in R), selfcol_events=sc, seeds=[r["seed"] for r in R])


if __name__ == "__main__":
    tags = sorted({os.path.basename(f)[:-6] for p in sys.argv[1:] for f in glob.glob(os.path.join(OUT, p + ".jsonl"))})
    rows = [r for r in (row(t) for t in tags) if r]
    if "--json" in sys.argv:
        print(json.dumps(rows))
    for r in rows:
        print(f"{r['tag']:28s} {r['wins']:2d}/{r['n']:<2d} [{r['win_lo']:.2f},{r['win_hi']:.2f}]  pts {r['pf']:3d}:{r['pa']:<3d} "
              f"share {r['share']:.2f} [{r['share_lo']:.2f},{r['share_hi']:.2f}]  ret {r['rally_ret']:7s} hpp {r['hits_per_pt']} "
              f"short {r['base_short']} lim {r['speed_lim']}/{r['acts']} sc {r['selfcol_events']}")
