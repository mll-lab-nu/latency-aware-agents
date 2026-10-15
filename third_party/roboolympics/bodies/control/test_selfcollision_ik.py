#!/usr/bin/env python3
"""The self-collision-avoiding IK of bodies/control/kinematics.py on the G1 and the H1-2 (needs mujoco: run in the
sports' image).

    python -m bodies.control.test_selfcollision_ik [--json out.json]

Per robot (the robot-only model with the paddle on the right wrist, floating base held at the stand, the legs at the
walking policy's stance): the paddle centre is sent to a grid of 60 targets in front of the robot, from its right side (free space) to across the chest (the
right arm reaching to the robot's left: the IK's shortest way often runs the forearm or the paddle through the
torso). Each target is solved twice from the rest pose, right arm only, 60 iterations: plain (avoid_self_collision=
False) and with the avoidance (the default when the model has pairs). Checks:
  (distances: the closest of the pairs the right arm moves; MuJoCo reads deeply intersecting hulls as 0.0)
  1 with the avoidance no solution penetrates: every pair keeps >= self_min_dist / 4 (the line search's hard floor;
    the projection aims at self_min_dist), and the median closest pair is >= self_min_dist - 5 mm
  2 the plain solver does penetrate on some targets (the test is not vacuous)
  3 targets the plain solver reaches in free space (its solution keeps >= self_min_dist): the avoidance reaches
    them as well (position error within 1 mm of the plain one)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from bodies import load_body  # noqa: E402
from bodies.control import kinematics as KIN  # noqa: E402

DMIN = 0.02


def rest_q(B, K):
    """the robot's joints (kinematics.py order) at its rest pose: legs at the policy's stance"""
    R = B.decl["rest"]
    q = np.zeros(K.nj)
    q[K.groups["legs"]] = B.decl["walk"]["default_angles"]
    if len(K.groups["waist"]) == 3:            # G1: yaw, roll, pitch; [rest] waist = [pitch, yaw]
        q[K.groups["waist"]] = [R["waist"][1], 0.0, R["waist"][0]]
    arms = K.groups["arms"]
    q[arms[:7]] = R["other_arm"]
    q[arms[7:]] = R["arm"]
    return q


def run(robot):
    B = load_body(robot)
    with tempfile.TemporaryDirectory(prefix="selfcol_ik_") as t:
        xml = Path(t) / "robot.xml"
        xml.write_text(B.robot_only_xml(hand="paddle", fixed_base=False))
        K = KIN.Kinematics(xml, robot=robot)
    base_pos, base_quat = np.array([0.0, 0.0, 0.793]), np.array([1.0, 0.0, 0.0, 0.0])
    q0 = rest_q(B, K)
    arm = K.groups["arms"][7:]                 # the right arm (the paddle's)
    K.set_state(base_pos, base_quat, q0)
    moving = K._moving_pairs(arm)
    rows = []
    for x in (0.22, 0.30, 0.38, 0.46):
        for y in (-0.25, -0.10, 0.05, 0.20, 0.35):
            for z in (0.20, 0.32, 0.44):
                tgt = base_pos + np.array([x, y, z])
                out = {"target": [round(float(v), 3) for v in tgt]}
                for name, avoid in (("plain", False), ("avoid", True)):
                    t0 = time.perf_counter()
                    q, err = K.solve([{"site": "hand_center", "pos": tgt}], base_pos, base_quat, q0, joints=arm,
                                     iters=60, avoid_self_collision=avoid, self_min_dist=DMIN)
                    K.min_self_distance(q, base_pos, base_quat)          # sets the pose
                    dist = K._min_dist(moving, KIN.DIST_CUT)                 # the pairs the right arm moves
                    pair = None
                    out[name] = {"err": round(err[0], 4), "min_dist": round(dist, 4), "pair": pair,
                                 "ms": round((time.perf_counter() - t0) * 1000, 1)}
                rows.append(out)
    pen_plain = [r for r in rows if r["plain"]["min_dist"] < 1e-6]       # two hulls in contact can read 0.0
    bad_avoid = [r for r in rows if r["avoid"]["min_dist"] < 0.25 * DMIN - 1e-4]
    med = float(np.median([r["avoid"]["min_dist"] for r in rows]))
    free = [r for r in rows if r["plain"]["min_dist"] >= DMIN]
    regress = [r for r in free if r["avoid"]["err"] > r["plain"]["err"] + 0.001]
    summary = {
        "robot": robot, "pairs": len(K.pairs), "pairs_moved_by_the_arm": len(moving), "targets": len(rows),
        "plain_penetrating": len(pen_plain),
        "plain_deepest_cm": round(-min(r["plain"]["min_dist"] for r in rows) * 100, 2),
        "avoid_min_dist_cm": round(min(r["avoid"]["min_dist"] for r in rows) * 100, 2),
        "avoid_below_floor": len(bad_avoid), "avoid_median_min_dist_cm": round(med * 100, 2),
        "free_targets": len(free), "free_regressions": len(regress),
        "free_err_mm_plain_max": round(max((r["plain"]["err"] for r in free), default=0) * 1000, 2),
        "free_err_mm_avoid_max": round(max((r["avoid"]["err"] for r in free), default=0) * 1000, 2),
        "blocked_reached_1cm": sum(r["avoid"]["err"] < 0.01 for r in pen_plain),
        "blocked_err_cm_max": round(max((r["avoid"]["err"] for r in pen_plain), default=0) * 100, 1),
        "ms_per_solve_plain": round(float(np.median([r["plain"]["ms"] for r in rows])), 1),
        "ms_per_solve_avoid": round(float(np.median([r["avoid"]["ms"] for r in rows])), 1),
        "blocked_err_cm_median": round(float(np.median([r["avoid"]["err"] for r in pen_plain])) * 100, 1)
        if pen_plain else None,
    }
    ok = not bad_avoid and med >= DMIN - 0.005 and len(pen_plain) > 0 and not regress
    return ok, summary, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("robots", nargs="*", default=["unitree_g1", "unitree_h1"])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    allok, out = True, {}
    for r in a.robots:
        ok, s, rows = run(r)
        out[r] = {"ok": ok, "summary": s, "rows": rows}
        print(("PASS" if ok else "FAIL"), json.dumps(s))
        allok &= ok
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
