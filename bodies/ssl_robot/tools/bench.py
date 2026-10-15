#!/usr/bin/env python3
"""Speed of a 10-robot world (README "Speed"): 5 v 5 robots driving to random targets every second, dribblers on,
a ball; the wall-clock per simulated second. python bodies/ssl_robot/tools/bench.py [seconds] [n dribbling]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bodies.ssl_robot.tools.lab import Lab  # noqa: E402


def main(seconds=10.0, dribbling=10):
    robots = [(f"{t}_{k}", t, k, ((-1 if t == "blue" else 1) * (0.5 + 0.7 * k), 0.5 * k - 1.0, 0.0))
              for t in ("blue", "yellow") for k in range(5)]
    lab = Lab(robots, ball=(0.0, 0.0))
    rng = np.random.default_rng(0)
    for r in lab.rigs[:dribbling]:
        r.dribble(3000)
    t0 = time.perf_counter()
    n = int(seconds / 0.01)
    for i in range(n):
        if i % 100 == 0:
            for r in lab.rigs:
                r.set_base_target(rng.uniform(-4, 4), rng.uniform(-2.8, 2.8), rng.uniform(-3, 3))
        lab.step()
    wall = time.perf_counter() - t0
    print(f"{dribbling} dribbling: {seconds:.0f} s simulated in {wall:.2f} s wall: {seconds / wall:.2f} x real time")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0, int(sys.argv[2]) if len(sys.argv) > 2 else 10)
