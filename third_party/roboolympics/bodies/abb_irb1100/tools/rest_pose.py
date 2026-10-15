#!/usr/bin/env python3
"""The ABB IRB 1100's ready pose (body.toml [rest] arm): the hand's centre at (x, y, z) in the robot's frame (on the
floor under the gantry's home; the arm's base at the mount height), its face normal (the hand's +x) along +x
(forward, towards a table's net); the rotation about the normal (the handle's direction) is free (FREE_ROLL; `tilt`
is used only when it is off). Damped least squares on axes 1..6 from 61 seeds; the solution with the largest margin
to the ranges (up to 0.8 rad), then the closest to the elbow-up seed; prints the joints and their margins.

    python tools/rest_pose.py [x y z tilt]        (default 0.42 -0.05 1.02 20)
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(x=0.42, y=-0.05, z=1.02, tilt=20.0):
    import mujoco as mj
    from bodies import load_body
    B = load_body("abb_irb1100")
    m = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle"))
    d = mj.MjData(m)
    r = B.rig(m, "", hand="paddle")
    t = math.radians(tilt)
    hx = np.array([1.0, 0.0, 0.0])
    hz = np.array([0.0, -math.sin(t), math.cos(t)])
    hy = np.cross(hz, hx)
    Rt = np.column_stack([hx, hy, hz])
    pt = np.array([x, y, z])
    lo, hi = r.LO[5:11], r.HI[5:11]
    best = None
    rng = np.random.default_rng(0)
    seeds = [np.array([0.0, 0.3, 0.3, 0.0, 1.0, 0.0])] + [rng.uniform(np.maximum(lo, -2.5), np.minimum(hi, 2.5))
                                                         for _ in range(60)]
    for q0 in seeds:
        q, err = solve(m, d, r, q0, pt, Rt, lo, hi)
        margin = float(np.min(np.minimum(q - lo, hi - q)))
        key = (err > 1e-6, -min(margin, 0.8), float(np.sum(np.abs(q - seeds[0]))))
        if best is None or key < best[0]:
            best = (key, q, err)
    _, q, err = best
    print("arm =", [round(float(v), 4) for v in q])
    print("error", err, "margins to the ranges (rad)", np.round(np.minimum(q - lo, hi - q), 2))


FREE_ROLL = True        # the rotation about the face normal is free (the handle's direction follows the arm)


def solve(m, d, r, q, pt, Rt, lo, hi):
    import mujoco as mj
    q = q.copy()
    e = np.ones(6)
    for _ in range(300):
        u = r.rest.copy(); u[5:11] = q
        d.qpos[r.qadr] = r.joint_targets(u)
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        p, R = r.tool(d)
        e_p = pt - p
        e_r = np.cross(R[:, 0], Rt[:, 0]) + (0.0 if FREE_ROLL else 0.5 * (np.cross(R[:, 1], Rt[:, 1]) +
                                                                       np.cross(R[:, 2], Rt[:, 2])))
        e = np.r_[e_p, e_r]
        if np.linalg.norm(e) < 1e-9:
            break
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mj.mj_jacSite(m, d, jp, jr, r.tool_site)
        J = np.vstack([jp, jr])[:, r.vadr[2:]]
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lo, hi)
    return q, float(np.linalg.norm(e))


if __name__ == "__main__":
    main(*map(float, sys.argv[1:5]))
