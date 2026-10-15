"""Run a grid of world-level matches (worldrun.py) with a few worker processes in this container (development and
the preliminary step-4 numbers before the plugin exists). Skips finished matches.

    python3 sports/soccer_ssl/screen/batch_local.py --variants reference,nopass --levels 1,2,3 --seeds 1-10 \
        --half 120 --out DIR [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def job(args):
    var, lvl, seed, half, out = args
    path = os.path.join(out, var, f"L{lvl}_s{seed}.json")
    if os.path.exists(path):
        return path, "skip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from worldrun import run_match

        res = run_match(var, f"L{lvl}" if not str(lvl).startswith("L") else lvl, seed, half_s=half)
        res.update(variant=var, level=lvl)
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, default=str)
        os.replace(path + ".tmp", path)
        return path, f"{res['score']['blue']}:{res['score']['yellow']} wall {res['wall_s']}"
    except Exception:  # noqa: BLE001
        return path, "ERROR " + traceback.format_exc()[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--levels", default="2,3")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--half", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    grid = [(v, int(l), s, a.half, a.out) for s in seeds(a.seeds) for l in a.levels.split(",")
            for v in a.variants.split(",")]
    t0 = time.time()
    with Pool(a.jobs) as p:
        for path, msg in p.imap_unordered(job, grid):
            print(f"[{time.time() - t0:7.0f}s] {os.path.relpath(path, a.out)} {msg}", flush=True)


if __name__ == "__main__":
    main()
