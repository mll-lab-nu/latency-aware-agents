"""Paired comparison of two variants of a batch_local.py directory, per level: the per-seed goal-difference gap
(variant A minus variant B on the same seed) with a 95 % bootstrap interval, and the wins of each.

    python3 sports/soccer_ssl/screen/compare.py DIR reference nopass
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def gd_by_seed(d, variant, level):
    out = {}
    for f in glob.glob(os.path.join(d, variant, f"L{level}_s*.json")):
        r = json.load(open(f))
        out[int(r["seed"])] = r["score"]["blue"] - r["score"]["yellow"]
    return out


def main():
    d, a, b = sys.argv[1:4]
    rng = np.random.default_rng(0)
    for lv in (1, 2, 3):
        A, B = gd_by_seed(d, a, lv), gd_by_seed(d, b, lv)
        seeds = sorted(set(A) & set(B))
        if not seeds:
            continue
        diff = np.array([A[s] - B[s] for s in seeds], float)
        boots = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wa = sum(A[s] > 0 for s in seeds)
        wb = sum(B[s] > 0 for s in seeds)
        print(f"L{lv} n={len(seeds)}  {a} wins {wa}  {b} wins {wb}  mean gd {np.mean([A[s] for s in seeds]):+.2f} vs "
              f"{np.mean([B[s] for s in seeds]):+.2f}  paired gap {diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
