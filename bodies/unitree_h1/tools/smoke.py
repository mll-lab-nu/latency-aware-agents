#!/usr/bin/env python3
"""Conformance smoke test of the Unitree H1 body (the shared body API, bodies/README.md), in the sim image:

    python bodies/unitree_h1/tools/smoke.py            (from the repository root; mujoco + numpy)

  1 the registry loads it; the walking policy's weights are Unitree's motion.pt (source hash)
  2 two robots mounted in one scene (A and B facing each other), standing for 3 s on the policy
  3 a servo reach: robot A, standing, takes its blade to three targets (IK on the planar copy, the joints moved within
    their per-step speed limits: every u passes rig.check); the tool's error at the end
  4 robot_only_xml: the planar, fixed-base and free copies compile; the planner (bodies/control/wholebody.py) builds on it
  5 every hand of bodies/hands mounts (on the hand arm and as other_hand) and the robot stands with it for 1 s
  6 walking: steady-state speeds for a set of commands (the numbers in body.toml and the README)
Exit code 0 when every check passes."""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import mujoco as mj  # noqa: E402

from bodies import load_body, planner  # noqa: E402

B = load_body("unitree_h1")
FAIL = []


def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def world(seats, hand="paddle", other_hand=None):
    w = ET.fromstring('<mujoco><compiler angle="radian" autolimits="true"/>'
                      '<option timestep="0.001" integrator="implicitfast"/><default/><asset/><worldbody>'
                      '<geom name="floor" type="plane" size="20 20 0.1" contype="1" conaffinity="5"/>'
                      '</worldbody></mujoco>')
    infos = {s: B.mount(w, f"{s}_", pos, yaw, hand=hand, other_hand=other_hand) for s, (pos, yaw) in seats.items()}
    return mj.MjModel.from_xml_string(ET.tostring(w, encoding="unicode")), infos


def run(m, d, rigs, us, n, cmd=None):
    """n control steps; us: seat -> u (held), cmd: seat -> walking command."""
    for _ in range(n):
        for s, r in rigs.items():
            if cmd and s in cmd:
                us[s] = us[s].copy()
                us[s][0:3] = np.clip(cmd[s], us[s][0:3] - r.STEP[0:3], us[s][0:3] + r.STEP[0:3])
            r.begin_step(d, us[s])
        for j in range(1, B.substeps + 1):
            for s, r in rigs.items():
                r.apply(d, us[s], us[s], j)
            mj.mj_step(m, d)


def upright(d, r):
    return float(d.xmat[r.base_body].reshape(3, 3)[2, 2]), float(d.xpos[r.base_body][2])


# 1 registry, policy
z = np.load(Path(sys.modules["bodies"].__file__).parent / "unitree_h1" / B.decl["walk"]["weights"])
ok("registry + policy hash", B.name == "unitree_h1" and str(z["source_sha256"]) == B.decl["walk"]["sha256"])

# 2 two robots
m, infos = world({"A": ((0, -1.5, 0), math.pi / 2), "B": ((0, 1.5, 0), -math.pi / 2)})
d = mj.MjData(m)
rigs = {s: B.rig(m, f"{s}_") for s in "AB"}
us = {s: r.rest.copy() for s, r in rigs.items()}
for s, r in rigs.items():
    r.place(d, us[s])
mj.mj_forward(m, d)
run(m, d, rigs, us, 300)
st = {s: upright(d, r) for s, r in rigs.items()}
ok("two robots stand 3 s", all(u > 0.98 and z > 0.95 for u, z in st.values()),
   ", ".join(f"{s}: up {u:.3f} pelvis z {z:.3f}" for s, (u, z) in st.items()))
ok("mount info", all(i["tool_site"] == f"{s}_hand_center" and i["contact_geoms"] == [f"{s}_hand_blade"]
                      for s, i in infos.items()))

# 3 servo reach (robot A): IK on the planar copy, then rate-limited joint motion on the standing robot
P = planner("unitree_h1").WholeBodyPlanner(B, "paddle")
rA = rigs["A"]
targets = [np.array([0.45, -0.30, 1.05]), np.array([0.40, -0.45, 1.25]), np.array([0.35, -0.45, 0.85])]
errs, refused, ik_res = [], 0, []
for tgt in targets:
    best = None
    for k in range(12):                                          # a few IK seeds: the rest pose, the forearm rolled
        seed = P.rest.copy()
        seed[9] = -2.5 + 0.45 * k
        u_goal, res = P.ik(tgt, np.array([1.0, 0.0, 0.0]), seed)
        if best is None or res < best[1]:
            best = (u_goal, res)
    u_goal, res = best
    ik_res.append(res)
    want = P.fk(u_goal)[0]                                       # where the planar copy puts the blade at u_goal
    goal = us["A"].copy()
    goal[3:] = u_goal[3:]
    for _ in range(400):                                          # at most 4 s
        u_prev = us["A"]
        u = u_prev + np.clip(goal - u_prev, -0.5 * rA.STEP, 0.5 * rA.STEP)
        u[0:3] = 0.0
        refused += rA.check(u, u_prev) is not None
        us["A"] = u
        run(m, d, {"A": rA}, us, 1)
        if np.max(np.abs(goal[3:] - u[3:])) < 1e-6:
            break
    run(m, d, {"A": rA}, us, 100)
    # the target is in A's mount frame (pelvis on the floor at the origin, facing +x); the robot sways and drifts
    pel, R = rA.pelvis(d)
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s_ = math.cos(yaw), math.sin(yaw)
    p_tool = rA.tool(d)[0]
    rel = p_tool - np.array([pel[0], pel[1], 0.0])
    local = np.array([c * rel[0] + s_ * rel[1], -s_ * rel[0] + c * rel[1], rel[2]])
    errs.append(float(np.linalg.norm(local - want)))
ok("servo reach within limits", refused == 0 and max(errs) < 0.05 and max(ik_res) < 0.01 and upright(d, rA)[0] > 0.98,
   f"blade vs the planar copy's {', '.join(f'{e * 100:.1f} cm' for e in errs)} (pelvis frame, the robot standing on "
   f"the policy; "
   f"IK residual {', '.join(f'{e * 100:.1f} cm' for e in ik_res)}), "
   f"{refused} refused steps")

# 4 robot_only_xml
for fb in ("planar", True, False):
    try:
        mm = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", fixed_base=fb))
        ok(f"robot_only_xml fixed_base={fb}", True, f"nq {mm.nq}, nu {mm.nu}, {mm.ngeom} geoms")
    except Exception as e:                                       # noqa: BLE001
        ok(f"robot_only_xml fixed_base={fb}", False, str(e)[:200])

# 5 every hand
import tomllib  # noqa: E402
hands = tomllib.loads((REPO / "bodies" / "hands" / "hands.toml").read_text())
for h in hands:
    try:
        mh, inf = world({"A": ((0, 0, 0), 0.0)}, hand=h, other_hand=h)
        dh = mj.MjData(mh)
        rh = {"A": B.rig(mh, "A_", hand=h)}
        uh = {"A": rh["A"].rest.copy()}
        rh["A"].place(dh, uh["A"])
        mj.mj_forward(mh, dh)
        run(mh, dh, rh, uh, 100)
        ok(f"hand {h} (both arms)", upright(dh, rh["A"])[0] > 0.98 and "tool_site2" in inf["A"],
           f"tool site {inf['A']['tool_site']}, contact {inf['A']['contact_geoms']}")
    except Exception as e:                                       # noqa: BLE001
        ok(f"hand {h}", False, str(e)[:200])

# 6 walking speeds (steady state: 3 s to settle, then 3 s measured)
mw, _ = world({"A": ((0, 0, 0), 0.0)})
rows = []
for cmd in ([0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, 1.0, 0], [0, 0, 0.5], [0, 0, 1.0]):
    dw = mj.MjData(mw)
    rw = {"A": B.rig(mw, "A_")}
    uw = {"A": rw["A"].rest.copy()}
    rw["A"].place(dw, uw["A"])
    mj.mj_forward(mw, dw)
    run(mw, dw, rw, uw, 300, {"A": np.array(cmd, float)})
    p0 = rw["A"].base_pose(dw)
    run(mw, dw, rw, uw, 300, {"A": np.array(cmd, float)})
    p1 = rw["A"].base_pose(dw)
    v = (p1[0:2] - p0[0:2]) / 3.0
    c, s_ = math.cos(p0[2]), math.sin(p0[2])
    rows.append((cmd, v @ [c, s_], v @ [-s_, c], (p1[2] - p0[2]) / 3.0, upright(dw, rw["A"])[0]))
for cmd, vx, vy, wz, up in rows:
    print(f"  walk {cmd}: forward {vx:+.2f} m/s, left {vy:+.2f} m/s, turn {wz:+.2f} rad/s, upright {up:.3f}")
ok("walking: no fall, forward at command 1.0 > 0.6 m/s", all(r[4] > 0.97 for r in rows) and rows[2][1] > 0.6)

print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
