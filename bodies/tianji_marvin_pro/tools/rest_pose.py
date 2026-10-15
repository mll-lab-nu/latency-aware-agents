#!/usr/bin/env python3
"""The Marvin Pro's poses in body.toml [rest] (and a sport's ready pose): damped least squares IK from random seeds on
the robot-only model, the solution whose joints keep the largest margin from their limits wins. Deterministic (seeded).

    python -I tools/rest_pose.py                      # prints the body's rest (hand arm) and other_arm
    python -I tools/rest_pose.py X Y Z NX NY NZ       # a hand-arm pose: the paddle centre at (X, Y, Z) with face
                                                      # normal N (mount frame: x forward, y left, z up from the floor)

rest (hand arm, right): `rest_pose.py 0.40 -0.50 1.00 1 0 0` (a hand's centre ahead and right, its face forward; the
default run below solves the mount pose with the hand axis forward, which the hands' face convention does not use);
other_arm (left): the free arm hanging beside the stand, its mount 0.05 m behind the centre, 0.40 m
left, 0.85 m high, pointing down (an earlier pose 0.25 m ahead touched a table's end edge in the table-tennis variant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def solve(m, frame, kind, joints, pos, axis=None, axis_col=2, seeds=64, rng=0, iters=300):
    import mujoco as mj
    d = mj.MjData(m)
    jid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j) for j in joints]
    qa = np.array([m.jnt_qposadr[j] for j in jid]); va = np.array([m.jnt_dofadr[j] for j in jid])
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    oid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE if kind == "site" else mj.mjtObj.mjOBJ_BODY, frame)
    R_ = np.random.default_rng(rng)
    best = None
    for _ in range(seeds):
        q = R_.uniform(lo + 0.2, hi - 0.2)
        for it in range(iters):
            d.qpos[qa] = q
            mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
            if kind == "site":
                p, Rm = d.site_xpos[oid], d.site_xmat[oid].reshape(3, 3)
            else:
                p, Rm = d.xpos[oid], d.xmat[oid].reshape(3, 3)
            e = [np.asarray(pos) - p]
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            if kind == "site":
                mj.mj_jacSite(m, d, jp, jr, oid)
            else:
                mj.mj_jacBody(m, d, jp, jr, oid)
            J = [jp[:, va]]
            if axis is not None:
                a = Rm[:, axis_col]
                e.append(np.cross(a, np.asarray(axis) / np.linalg.norm(axis)))
                J.append(jr[:, va])
            e, J = np.concatenate(e), np.vstack(J)
            if np.linalg.norm(e) < 1e-5:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
            null = np.eye(len(q)) - J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), J)
            if it < iters - 60:                           # the margin pull, then a polish without it
                dq += null @ (0.05 * ((lo + hi) / 2 - q))
            q = np.clip(q + dq, lo, hi)
        if np.linalg.norm(e) < 1e-4:
            margin = float(np.min(np.minimum(q - lo, hi - q)))
            if best is None or margin > best[0]:
                best = (margin, q.copy())
    return best


def main(argv):
    import mujoco as mj
    from bodies import load_body
    B = load_body("tianji_marvin_pro")
    m = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", arm="right"))
    R = [f"right_arm_{i}" for i in range(7)]
    L = [f"left_arm_{i}" for i in range(7)]
    if len(argv) >= 6:
        x, y, z, nx, ny, nz = map(float, argv[:6])
        best = solve(m, "hand_center", "site", R, (x, y, z), (nx, ny, nz), axis_col=0)
        print("ready (arm, right):", None if best is None else (round(best[0], 3), np.round(best[1], 4).tolist()))
        return
    best = solve(m, "right_mount", "body", R, (0.40, -0.30, 1.00), (1.0, 0.0, 0.0))
    print("rest arm (right):", None if best is None else (round(best[0], 3), np.round(best[1], 4).tolist()))
    best = solve(m, "left_mount", "body", L, (-0.05, 0.40, 0.85), (0.0, 0.0, -1.0))
    print("other_arm (left):", None if best is None else (round(best[0], 3), np.round(best[1], 4).tolist()))


if __name__ == "__main__":
    main(sys.argv[1:])
