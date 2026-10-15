#!/usr/bin/env python3
"""bodies/servo.py PlanarBase at 1 kHz (the library's gains KP 60 /s, KV 300 /s) against body.py RatePlanarBase
at 100 Hz (KP 10, KV 50): the same moves, the peaks of the setpoint's speed, acceleration (robot frame, on the
norm) and jerk against the limits, the time to within 1 cm / 0.01 rad of the goal, overshoot, and the yaw
acceleration's sign flips (chatter). Also the library's gains at 100 Hz (what goes wrong). README "Speed".

    python bodies/ssl_robot/tools/planarbase_rate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bodies import servo as SV  # noqa: E402
from bodies.ssl_robot import body as SSL  # noqa: E402

MOVES = [(1.5, 0.8, 2.5), (-1.0, 1.2, -1.0), (0.2, -0.3, 3.0), (4.0, 0.0, 0.0), (0.05, 0.0, 0.1), (-3.0, -2.0, 1.5)]


def run(cls, dt, goal, L, T=4.0):
    pb = cls(L, dt)
    pb.reset(np.zeros(3))
    r = np.array(goal, float)
    pv = pa = pj = pw = pwd = 0.0
    prev_a, prev_wd = np.zeros(2), 0.0
    reach, over, flips = None, 0.0, 0
    u = r[:2] / (np.linalg.norm(r[:2]) or 1.0)
    for i in range(int(T / dt)):
        q, qd, qdd = pb.step(r, np.zeros(3))
        pv = max(pv, np.linalg.norm(pb.v) / L.speed)
        pa = max(pa, np.linalg.norm(pb.a) / L.accel)
        pj = max(pj, np.linalg.norm(pb.a - prev_a) / dt / L.jerk)
        pw = max(pw, abs(pb.w) / L.yaw_rate)
        pwd = max(pwd, abs(pb.wd) / L.yaw_accel)
        if abs(pb.wd) > 1.0 and abs(prev_wd) > 1.0 and np.sign(pb.wd) != np.sign(prev_wd):
            flips += 1
        prev_a, prev_wd = pb.a.copy(), pb.wd
        over = max(over, float((q[:2] - r[:2]) @ u), abs(q[2]) - abs(r[2]) if np.sign(q[2]) == np.sign(r[2]) else 0)
        if reach is None and np.linalg.norm(q[:2] - r[:2]) < 0.01 and abs(q[2] - r[2]) < 0.01:
            reach = (i + 1) * dt
    return dict(speed=pv, accel=pa, jerk=pj, yaw_rate=pw, yaw_accel=pwd, reach=reach, over=over, flips=flips)


def main():
    L = SV.BaseLimits.of(SSL.load_decl()["limits"]["base"])

    class Lib100(SV.PlanarBase):
        pass
    rows = {"PlanarBase 1 kHz": (SV.PlanarBase, 0.001), "RatePlanarBase 100 Hz": (SSL.RatePlanarBase, 0.01),
            "PlanarBase gains at 100 Hz": (Lib100, 0.01)}
    for name, (cls, dt) in rows.items():
        res = [run(cls, dt, g, L) for g in MOVES]
        peak = {k: max(r[k] for r in res) for k in ("speed", "accel", "jerk", "yaw_rate", "yaw_accel", "over")}
        print(f"{name:28s} peaks/limit: speed {peak['speed']:.4f} accel {peak['accel']:.4f} jerk {peak['jerk']:.4f} "
              f"yaw rate {peak['yaw_rate']:.4f} yaw accel {peak['yaw_accel']:.4f}; overshoot {peak['over'] * 100:.1f} "
              f"cm/rad*100; yaw-accel flips {sum(r['flips'] for r in res)}; reach {[r['reach'] for r in res]}")


if __name__ == "__main__":
    main()
