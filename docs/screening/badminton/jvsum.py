#!/usr/bin/env python3
"""Summarise jvrec npz files: over-speed share of arm time, episodes, max ratio, by joint, and attribution.

    python3 jvsum.py DIR [DIR ...]
"""
import sys
from pathlib import Path
import numpy as np

V = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
TAU = np.array([87, 87, 87, 87, 12, 12, 12.0])
DT = 0.001


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def summ(arrs, label):
    A = np.vstack(arrs)
    qd, sp = A[:, 1:8], A[:, 8:15]
    tin, tout = A[:, 15:22], A[:, 22:29]
    bacc, bq = A[:, 29:31], A[:, 31:33]
    cpl = A[:, 35:42]
    R = np.abs(qd) / V
    over = R.max(1) > 1.0
    over5 = R.max(1) > 1.05
    eps = []
    for a in arrs:
        r = (np.abs(a[:, 1:8]) / V).max(1)
        eps += [(j - i) * DT for i, j in runs(r > 1.0)]
    eps = np.array(eps) if eps else np.zeros(1)
    byj = np.bincount(np.argmax(R[over], 1), minlength=7) / max(1, over.sum())
    sat = np.abs(tout) >= TAU * 0.999
    ratel = np.abs(tin - tout) > 1e-6
    cplr = np.abs(cpl) / TAU
    ba = np.hypot(*bq.T)
    nrf = sum(int(a[:, 43].max() > 0) for a in arrs)
    print(f"{label}: arm-s {len(A) * DT:7.1f}; over limit {over.mean():6.2%}; over 1.05x {over5.mean():6.2%}; "
          f"episodes {len(eps)} (median {1e3 * np.median(eps):.0f} ms, p90 {1e3 * np.percentile(eps, 90):.0f} ms); "
          f"max {R.max():.2f}x; by joint {np.round(byj, 2).tolist()}")
    print(f"   setpoint max {np.max(np.abs(sp) / V):.3f}x; at over-speed: torque saturated {sat[over].any(1).mean():.0%}, "
          f"rate-limited {ratel[over].any(1).mean():.0%}, |base acc| median {np.median(ba[over]):.1f} "
          f"(all {np.median(ba):.1f}) m/s2, base acc > 5: {(ba[over] > 5).mean():.0%} (all {(ba > 5).mean():.0%}), "
          f"coupling/tau_max max-joint median {np.median(cplr[over].max(1)):.2f} (all {np.median(cplr.max(1)):.2f}), "
          f"resync {A[over, 42].mean():.0%}")
    return A


if __name__ == "__main__":
    for d in sys.argv[1:]:
        fs = sorted(Path(d).glob("*.npz"))
        allarrs = []
        for f in fs:
            z = np.load(f)
            arrs = [z[f"a{i}"] for i in range(int(z["n"]))]
            allarrs += arrs
            summ(arrs, f"{Path(d).name}/{f.stem}")
        if len(fs) > 1:
            summ(allarrs, f"{Path(d).name}/ALL")
