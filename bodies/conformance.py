#!/usr/bin/env python3
"""Conformance of the robot library's robots (bodies/README.md "准入"): the checks every robot passes before it is
admitted, the same for every robot (generalised from the Marvin Pro's and the R1 Pro's tools/conformance.py).

    python -m bodies.conformance                       every robot of the library
    python -m bodies.conformance unitree_g1 --json out.json

Per robot (a robot with a model: bodies/__init__.py BODIES; the limits-only arm entries run 1, 4b and 7):
  1 declared      body.toml declares every limit with a source (bodies/sources.py); a LICENSE and, for a model, the
                  provenance tool tools/make_model.py; a test-use robot says so
  2 scene         two robots (seats A and B, team accents) mounted in one world compile; the robot-only export has
                  the same kinematics as the mounted robot; two robots placed into each other do not touch as the
                  library mounts them and do touch after bodies/contacts.py between() (its own contacts unchanged)
  3 servo         the servo reaches targets: random joint targets inside the ranges, given as a step (the setpoint
                  generator plans the way), are reached within 0.01 rad; a planar base reaches its goal within 1 cm;
                  a legged robot walks at the real walking command limits without falling
  3b walkers      a legged robot's optional walking policies (body.walkers after the default; the G1's FALCON and
                  AGILE), each: the port reproduces the reference outputs stored with its weights, the robot stands
                  still for 10 s, walks at the real command limits without falling and along the command (at least
                  40 % of the limit on the requested axis), and replays bit-identically; the same under the arbiter
                  (bodies/control/arbiter.py), plus a fast two-arm swing plan while walking and the legs planned and
                  handed back to the walker without falling
  4 limits        enforced: under aggressive references (steps across the range, the base driven back and forth)
                  every servo setpoint stays within the declared speed / acceleration limits (and its jerk within the
                  jerk limit), every motor torque within its torque limit and, where declared, its torque-speed curve;
                  check() refuses rows beyond the speed and acceleration limits; the actual joint speeds are reported
                  (physics, not a setpoint: recorded); 4b for the arm entries: limits.JerkLimiter on random references
  5 appearance    the appearance meshes (meshes_fine/, the presentation level) and the simulation meshes move with the
                  same kinematics: at random poses every mesh geom's vertices of the two levels coincide within the
                  decimation's error (skipped when meshes_fine/ is absent: it is not in git, tools/make_model.py)
  6 replay        a 3 s run of targets (and walking) replays bit-identically in a fresh MjData and a fresh rig
  7 assembly      the assembled configuration's hash is stable (twice the same; the same in a copy of the files as
                  tools/sportlib.copy_body makes it), and changes with the attachment and the controllers
  9 self-collision  the declaration (body.toml [self_collision]) fits the model: no enabled pair penetrates at the
                  rest pose, none overlaps in (nearly) every random legal pose, no exclusion is unused
                  (bodies/selfcollision.py check_body); and the contacts block: the hand arm driven at its speed
                  limits to a pose inside the body stops at the body (penetration < 1.5 cm; >= 3 cm with
                  self-collision off), and the rig's monitor records the event and its force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bodies import ARMS, BODIES, LEGS, assembly, load_body  # noqa: E402
from bodies import limits as AL  # noqa: E402
from bodies import sources  # noqa: E402

WORLD = """<mujoco model="conformance"><compiler angle="radian" autolimits="true"/>
<option timestep="0.001" integrator="implicitfast"/>
<worldbody><light pos="0 0 4" dir="0 0 -1"/><geom name="floor" type="plane" size="8 8 0.1" conaffinity="5"/>
</worldbody></mujoco>"""
TOL = 1e-6


def _mj():
    import mujoco
    return mujoco


class Report:
    def __init__(self, robot):
        self.robot, self.rows = robot, []

    def add(self, check, status, detail=""):
        self.rows.append(dict(check=check, status=status, detail=detail))
        print(f"  {status:4s} {check}: {detail}", flush=True)

    def ok(self):
        return all(r["status"] != "FAIL" for r in self.rows)


def _robot_only(B, **kw):
    mj = _mj()
    m = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", **kw))
    return m, mj.MjData(m)


def _legged(B):
    return B.kind == "legged"


def _step(r, m, d, u_prev, u_next, check_out=None):
    mj = _mj()
    r.begin_step(d, u_next)
    for j in range(1, r.substeps + 1):
        r.apply(d, u_prev, u_next, j)
        if check_out is not None:
            check_out(r, d)
        mj.mj_step(m, d)


def _joint_coords(r):
    """the commanded joint coordinates of u (those that move a joint), without a planar base: u[3:12], and u[0:3]
    where they are joints of their own (a gantry's linear axes: no planar base servo)"""
    moves = np.any(np.abs(r.M) > 0, axis=0)
    first = 3 if r.servo.base is not None else 0
    return [k for k in range(first, 12) if moves[k]]


# ------------------------------------------------------------------------------------------------ 1 declared
def check_declared(name, R: Report):
    d = HERE / name
    decl = tomllib.loads((d / "body.toml").read_text())
    probs = sources.source_problems(decl)
    n_est = sum(str(v).startswith("ESTIMATE") for L in decl.get("limits", {}).values() if isinstance(L, dict)
                for v in (L.get("source") or {}).values())
    R.add("1 limits declared with sources", "FAIL" if probs else "PASS",
          "; ".join(probs) or f"{len(decl.get('sources', {}))} sources, {n_est} labelled estimates")
    lic = (d / "LICENSE").exists() or decl["body"].get("kind") in ("arm", "legs")
    R.add("1 licence file", "PASS" if lic else "FAIL", "LICENSE" if (d / "LICENSE").exists() else
          "limits-only entry (the sports' scenes carry the model under their own NOTICE)")
    if name in BODIES:
        R.add("1 provenance tool", "PASS" if (d / "tools" / "make_model.py").exists() else "FAIL", "tools/make_model.py")
    st = decl["body"].get("status", "library")
    if st == "test-use":
        txt = (d / "README.md").read_text() if (d / "README.md").exists() else ""
        R.add("1 test-use flagged", "PASS" if "TEST USE" in txt else "FAIL", "status test-use, README banner")
    return decl


# ------------------------------------------------------------------------------------------------ 2 scene
def check_scene(B, R: Report):
    mj = _mj()
    root = ET.fromstring(WORLD)
    infos = {}
    for seat, (pos, yaw, acc) in {"A": ((0.0, -1.5, 0.0), math.pi / 2, (0.1, 0.35, 0.85, 1)),
                                  "B": ((0.0, 1.5, 0.0), -math.pi / 2, (0.85, 0.15, 0.12, 1))}.items():
        infos[seat] = B.mount(root, f"{seat}_", pos, yaw, hand="paddle", accent=acc)
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    rigs = {s: B.rig(m, f"{s}_", hand="paddle") for s in "AB"}
    for r in rigs.values():
        r.place(d, r.rest)
    mj.mj_forward(m, d)
    pa, pb = rigs["A"].tool(d)[0], rigs["B"].tool(d)[0]
    ok = pa[1] < 0 < pb[1] and all(mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, c) >= 0 for s in "AB"
                                   for c in infos[s]["cameras"])
    R.add("2 two robots in one scene", "PASS" if ok else "FAIL", f"nu {m.nu}, tools at y {pa[1]:.2f} / {pb[1]:.2f}")
    # the robot-only export: the same kinematics as the mounted robot
    kw = {"fixed_base": True} if _legged(B) else {}
    m1, d1 = _robot_only(B, **kw)
    r1 = B.rig(m1, "", hand="paddle")
    root = ET.fromstring(WORLD)
    B.mount(root, "X_", (0.0, 0.0, 0.0), 0.0, hand="paddle", **kw)
    m2 = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d2 = mj.MjData(m2)
    r2 = B.rig(m2, "X_", hand="paddle")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5):
        u = _random_u(r1, rng)
        r1.place(d1, u); r2.place(d2, u)
        mj.mj_kinematics(m1, d1); mj.mj_kinematics(m2, d2)
        p1, p2 = r1.tool(d1)[0], r2.tool(d2)[0]
        if _legged(B):                                   # the welded pelvis sits at the mount height in both
            p1, p2 = p1 - d1.xpos[r1.base_body], p2 - d2.xpos[r2.base_body]
        worst = max(worst, float(np.linalg.norm(p1 - p2)))
    R.add("2 robot-only export = mounted kinematics", "PASS" if worst < 1e-9 else "FAIL", f"max tool offset {worst:.1e} m")
    n0, n1, own0, own1 = _overlap_contacts(B)
    R.add("2 robots touch each other (bodies/contacts.py)", "PASS" if n0 == 0 and n1 > 0 and own0 == own1 else "FAIL",
          f"A x B contacts {n0} as mounted, {n1} after contacts.between; A's own contacts {own0} -> {own1}")


def _overlap_contacts(B):
    """Two robots, B 5 cm from A (same heading, rest pose): (A x B contacts as the library mounts them, after
    contacts.between, A's own contacts before and after). Hands off: their contact geoms touch each other anyway."""
    from bodies import contacts as CT
    mj = _mj()
    root = ET.fromstring(WORLD)
    kw = {"fixed_base": True} if _legged(B) else {}
    for seat, x in (("A", 0.0), ("B", 0.05)):
        B.mount(root, f"{seat}_", (x, 0.0, 0.0), 0.0, hand=None, **kw)
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    for s in "AB":
        r = B.rig(m, f"{s}_", hand=None)
        r.place(d, r.rest)

    def count():
        mj.mj_forward(m, d)
        ab = len(CT.contacts_between(m, d, "A_", "B_"))
        own = sum(1 for c in d.contact[:d.ncon]
                  if all((mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").startswith("A_")
                         for g in (c.geom1, c.geom2)))
        return ab, own
    n0, own0 = count()
    CT.between(m, ["A_", "B_"])
    n1, own1 = count()
    return n0, n1, own0, own1


def _random_u(r, rng, margin=0.15, spread=0.6):
    """a random target of the joint coordinates: within `spread` rad of the ready pose, `margin` inside the ranges"""
    u = r.rest.copy()
    for k in _joint_coords(r):
        lo, hi = r.LO[k] + margin, r.HI[k] - margin
        if hi <= lo:
            u[k] = 0.5 * (r.LO[k] + r.HI[k])
            continue
        u[k] = float(np.clip(r.rest[k] + rng.uniform(-spread, spread), lo, hi))
    return u


def _limits_u(r):
    """the servo's limits on u's joint coordinates: (indices, vel, acc or None, jerk or None)"""
    return r.servo.jix, r.servo.lim                       # the declared limits (not the generator's margins)


# ------------------------------------------------------------------------------------------------ 3 servo
def check_servo(B, R: Report):
    mj = _mj()
    kw = {"fixed_base": True} if _legged(B) else {}
    m, d = _robot_only(B, self_collision=False, **kw)   # the servo alone: random targets may lie inside the body,
    #                                                     where the arm now stops (check 9 tests the contacts)
    r = B.rig(m, "", hand="paddle")
    rng = np.random.default_rng(7)
    jix, lim = _limits_u(r)
    errs, times = [], []
    for _ in range(4):
        r.place(d, r.rest); mj.mj_forward(m, d)
        goal = _random_u(r, rng)
        T = AL.plan_line(r.rest[jix], goal[jix], lim).t_min if lim is not None else 0.0
        n = int(math.ceil((2.0 * T + 1.0) / r.dt))
        for _ in range(n):
            _step(r, m, d, goal, goal)
        err = float(np.max(np.abs(r.u_of(d)[jix] - goal[jix])))
        errs.append(round(err, 4)); times.append(round(T, 2))
    R.add("3 servo reaches joint targets", "PASS" if max(errs) < 0.01 else "FAIL",
          f"max |u - target| {errs} rad (S-curve times {times} s)")
    if r.servo.base is not None:
        r.place(d, r.rest); mj.mj_forward(m, d)
        goal = r.rest.copy(); goal[0:2] = (0.8, -0.6)
        for _ in range(500):
            _step(r, m, d, goal, goal)
        e = float(np.linalg.norm(r.u_of(d)[0:2] - goal[0:2]))
        R.add("3 base reaches its goal", "PASS" if e < 0.01 else "FAIL", f"1 m move, error {e * 1000:.1f} mm")
    if _legged(B):
        check_walk(B, R)
        check_walkers(B, R)


def check_walk(B, R: Report, seconds=4.0, walker=None):
    """the floating-base robot walks: requested beyond the real limits, it gets the limits; it stays up"""
    mj = _mj()
    root = ET.fromstring(WORLD)
    B.mount(root, "", (0.0, 0.0, 0.0), 0.0, hand=None)
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    r = B.rig(m, "", hand=None, **_walker(walker))
    out = []
    ok = True
    for req in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)):
        r.place(d, r.rest); mj.mj_forward(m, d)
        u = r.rest.copy(); u[0:3] = req
        p0, yaw0 = d.xpos[r.base_body].copy(), r.base_pose(d)[2]
        n = int(seconds / r.dt)
        zmin, cmds, yaws = 9.0, [], [yaw0]
        for k in range(n):
            _step(r, m, d, u, u)
            zmin = min(zmin, float(d.xpos[r.base_body][2]))
            cmds.append(r.cmd.copy())
            yaws.append(r.base_pose(d)[2])
        cm = np.max(np.abs(cmds), axis=0)
        lim = np.asarray(B.decl["limits"]["walk"]["command_max"], float)
        dv = (d.xpos[r.base_body] - p0)[:2] / seconds
        dyaw = (np.unwrap(yaws)[-1] - yaw0) / seconds
        ok &= bool(np.all(cm <= lim + 1e-9)) and zmin > 0.45 * B.decl["body"]["height_m"]
        if walker:                                # an optional walker also follows the command: along the requested
            axis = int(np.flatnonzero(req)[0])    # axis, at least 40 % of the limit
            ok &= bool(np.r_[dv, dyaw][axis] >= 0.4 * lim[axis])
        out.append(f"request {req} -> command max {np.round(cm, 2).tolist()}, mean speed "
                   f"{np.round(dv, 2).tolist()} m/s, {dyaw:.2f} rad/s, pelvis >= {zmin:.2f} m")
    R.add("3 walks within the real command limits" + (f" ({walker})" if walker else ""), "PASS" if ok else "FAIL",
          "; ".join(out))


def _walker(walker):
    return {"walker": walker} if walker else {}


def check_walkers(B, R: Report):
    """3b the optional walking policies of a legged robot (B.walkers[1:]): parity, standing, walking, replay"""
    mj = _mj()
    for w in getattr(B, "walkers", ())[1:]:
        root = ET.fromstring(WORLD)
        B.mount(root, "", (0.0, 0.0, 0.0), 0.0, hand="paddle")
        m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mj.MjData(m)
        r = B.rig(m, "", hand="paddle", walker=w)
        if hasattr(r.policy, "parity"):
            e = r.policy.parity()
            R.add(f"3b {w}: port = reference outputs", "PASS" if e < 1e-4 else "FAIL", f"max |difference| {e:.1e}")
        r.place(d, r.rest)
        mj.mj_forward(m, d)
        p0, zmin = d.xpos[r.base_body].copy(), 9.0
        for _ in range(int(10.0 / r.dt)):
            _step(r, m, d, r.rest, r.rest)
            zmin = min(zmin, float(d.xpos[r.base_body][2]))
        drift = float(np.linalg.norm((d.xpos[r.base_body] - p0)[:2]))
        ok = zmin > 0.45 * B.decl["body"]["height_m"] and drift < 0.1
        R.add(f"3b {w}: stands still", "PASS" if ok else "FAIL", f"10 s with the paddle: drift {drift * 100:.1f} cm, "
              f"pelvis >= {zmin:.2f} m; legs' motor curves clipped in {r.legs_act.stats['envelope_clipped']} of "
              f"{r.legs_act.stats['steps']} steps")
        check_walk(B, R, walker=w)
        a, b = _run(B, walker=w), _run(B, walker=w)
        R.add(f"3b {w}: deterministic replay", "PASS" if a == b else "FAIL", f"3 s, state hash {a[:16]} twice")
        (rows, a), (_, b) = _walker_arbiter(B, w), _walker_arbiter(B, w)
        for check, ok, detail in rows:
            R.add(f"3b {w} + arbiter: {check}", "PASS" if ok else "FAIL", detail)
        R.add(f"3b {w} + arbiter: deterministic replay", "PASS" if a == b else "FAIL", f"state hash {a[:16]} twice")


def _walker_arbiter(B, walker, seconds=4.0):
    """An optional walker under the arbiter: 10 s standing; requests beyond the command limits on each axis; a fast
    two-arm swing plan (shoulder pitches +-1.4 rad in 0.25 s and back, with velocities, arm kp 400) while walking at
    0.4 m/s; then the legs planned (held where they are) and handed back. Returns ([(check, ok, detail)], hash)."""
    from bodies.control.arbiter import CODE, minjerk
    m, d, r, arb, run = _arbiter(B, walker)
    up = 0.45 * B.decl["body"]["height_m"]
    h, rows = hashlib.sha256(), []

    def record():
        for a in (d.qpos, d.qvel, arb.memory()):
            h.update(a.tobytes())

    def place():                                              # standing at the mount, the arbiter reset
        r.place(d, r.rest)
        _mj().mj_forward(m, d)
        arb.reset(d)

    p0 = d.xpos[r.base_body].copy()
    zmin = run(int(10.0 / r.dt))
    drift = float(np.linalg.norm((d.xpos[r.base_body] - p0)[:2]))
    rows.append(("stands still", zmin > up and drift < 0.1, f"10 s, arms held: drift {drift * 100:.1f} cm, "
                 f"pelvis >= {zmin:.2f} m"))
    record()
    lim = arb.cmd_max
    out, ok = [], True
    for req in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)):
        place()
        arb.walk(d, req)
        p0, yaws, cm, zmin = d.xpos[r.base_body].copy(), [r.base_pose(d)[2]], np.zeros(3), 9.0
        for _ in range(int(seconds / r.dt)):
            zmin = min(zmin, run(1))
            cm = np.maximum(cm, np.abs(arb.cmd))
            yaws.append(r.base_pose(d)[2])
        speed = np.r_[(d.xpos[r.base_body] - p0)[:2], np.unwrap(yaws)[-1] - yaws[0]] / seconds
        axis = int(np.flatnonzero(req)[0])
        ok &= bool(np.all(cm <= lim + 1e-9) and zmin > up and speed[axis] >= 0.4 * lim[axis])
        out.append(f"request {req} -> command max {np.round(cm, 2).tolist()}, mean speed "
                   f"{np.round(speed, 2).tolist()}, pelvis >= {zmin:.2f} m")
        record()
    rows.append(("walks within the real command limits", ok, "; ".join(out)))
    place()
    arb.walk(d, (0.4, 0.0, 0.0))
    zmin = run(150)
    A = arb.GROUPS["arms"]
    q0 = arb.target()[A]
    swing = np.zeros(len(A))
    swing[[0, 7]] = -1.4, 1.4                                 # the shoulder pitches, one arm forward, one back
    s = [minjerk((k + 1) / 25) for k in range(25)]
    Q = np.array([q0 + sg * swing * c for sg in (1, -1, 1, -1) for c in (s + s[::-1][1:] + [0.0])])
    why = arb.plan(d, "arms", Q, QD=np.diff(np.vstack([q0, Q]), axis=0) / r.dt, kp=400.0)
    zmin = min(zmin, run(300))
    walked = all(o[0] == CODE["walk"] for o in arb.owner_log)
    rows.append(("two-arm swing plan while walking", why is None and walked and zmin > up,
                 f"{len(Q)} rows {'accepted' if why is None else 'refused: ' + why}; legs walking throughout: "
                 f"{walked}; pelvis >= {zmin:.2f} m"))
    record()
    arb.walk(d, (0.0, 0.0, 0.0))
    zmin = run(100)
    Lg = arb.GROUPS["legs"]
    why = arb.plan(d, "legs", np.repeat(d.qpos[arb.qadr[Lg]][None], 30, axis=0))
    zmin = min(zmin, run(30))
    arb.release(d, "legs")
    zmin = min(zmin, run(250))
    ev = [e[2] for e in arb.events]
    rows.append(("legs planned and handed back", why is None and "walk_reinit" in ev and arb.owner["legs"] == "walk"
                 and zmin > up, f"events {sorted(set(ev))}, pelvis >= {zmin:.2f} m"))
    record()
    h.update(repr(arb.events).encode())
    return rows, h.hexdigest()


# ------------------------------------------------------------------------------------------------ 4 limits
def check_limits(B, R: Report):
    mj = _mj()
    kw = {"fixed_base": True} if _legged(B) else {}
    m, d = _robot_only(B, **kw)
    r = B.rig(m, "", hand="paddle")
    jix, lim = _limits_u(r)
    act = r.servo.act
    bad_tau = [0.0]

    def torque_ok(rr, dd):
        along, against = act.limit(dd.qvel[act.vadr])
        t = dd.ctrl[act.aid]
        L = np.where(t * dd.qvel[act.vadr] > 0, along, against)
        bad_tau[0] = max(bad_tau[0], float(np.max(np.abs(t) / np.maximum(L, 1e-9))))

    r.place(d, r.rest); mj.mj_forward(m, d)
    r.servo.peak = dict(vel=np.zeros(12), acc=np.zeros(12))
    rng = np.random.default_rng(3)
    jerk_peak = np.zeros(12)
    qd_peak = np.zeros(len(r.vadr))
    prev_acc = None
    u = r.rest.copy()
    for seg in range(6):                                   # steps across the ranges: the setpoints must plan
        goal = _random_u(r, rng, margin=0.05, spread=1.5)
        if r.servo.base is not None:
            goal[0:3] = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-1.0, 1.0)
        for _ in range(60):
            r.begin_step(d, goal)
            for j in range(1, r.substeps + 1):
                r.apply(d, u, goal, j)
                torque_ok(r, d)
                a = r.servo.xdd.copy()
                if prev_acc is not None:
                    jerk_peak = np.maximum(jerk_peak, np.abs(a - prev_acc) / (r.dt / r.substeps))
                prev_acc = a
                mj.mj_step(m, d)
                qd_peak = np.maximum(qd_peak, np.abs(d.qvel[r.vadr]))
            u = goal
    pv, pa = r.servo.peak["vel"], r.servo.peak["acc"]
    vel_ratio = float(np.max(pv[jix] / lim.vel))
    acc_ratio = float(np.max(pa[jix] / lim.acc)) if lim.acc is not None else None
    jerk_ratio = float(np.max(jerk_peak[jix] / lim.jerk)) if lim.jerk is not None else None
    detail = f"setpoint speed {vel_ratio:.4f} x limit"
    ok = vel_ratio <= 1 + TOL
    if acc_ratio is not None:
        detail += f", acceleration {acc_ratio:.4f} x"
        ok &= acc_ratio <= 1 + TOL
    else:
        detail += ", acceleration not declared (NOT PUBLISHED: the motor curves bound it)"
    if jerk_ratio is not None:
        detail += f", jerk {jerk_ratio:.4f} x"
        ok &= jerk_ratio <= 1 + 1e-3
    if r.servo.base is not None:
        bl = r.servo.base.lim
        detail += f"; base speed {float(np.max(pv[0:2] / np.max(bl.speed))):.3f} x, yaw {pv[2] / bl.yaw_rate:.3f} x"
        ok &= check_base_limits(B, R)
    detail += f"; torques {bad_tau[0]:.4f} x their (curve) limit"
    ok &= bad_tau[0] <= 1 + TOL
    act_ratio = qd_peak
    R.add("4 setpoints and torques within the limits", "PASS" if ok else "FAIL", detail)
    # the actual joint speeds (physics; recorded, not enforced: audit D9)
    vel_j = np.abs(r.M[:, jix]) @ lim.vel                 # each joint's speed limit through the coordinate map
    sel = vel_j > 0
    R.add("4 actual joint speeds (recorded)", "INFO",
          f"max {float(np.max(act_ratio[sel] / vel_j[sel])):.3f} x the speed limit")
    # check(): rows beyond the limits are refused
    u0 = r.rest.copy()
    k = jix[-1]
    u1 = u0.copy(); u1[k] += 0.9 * lim.vel[-1] * r.dt
    u2 = u0.copy(); u2[k] += 1.1 * lim.vel[-1] * r.dt
    probs = []
    if r.check(u1, u0) is not None:
        probs.append(f"refused a row at 0.9 x speed: {r.check(u1, u0)}")
    if r.check(u2, u0) is None:
        probs.append("accepted a row at 1.1 x speed")
    if lim.acc is not None:
        a = lim.acc[-1] * r.dt * r.dt
        u3 = u0.copy(); u3[k] += 1.5 * a
        if r.check(u3, u0, u0) is None:
            probs.append("accepted a row at 1.5 x the acceleration")
    R.add("4 check() refuses rows beyond the limits", "FAIL" if probs else "PASS", "; ".join(probs) or
          "speed" + (" and acceleration" if lim.acc is not None else ""))
    if act.env is not None:
        R.add("4 motor torque-speed curves applied", "PASS", f"{len(act.aid)} motors; clipped in "
              f"{act.stats['envelope_clipped']} of {act.stats['steps']} steps")


def check_base_limits(B, R: Report):
    """the base's setpoints in the robot's frame: per axis or on the norm, within speed and acceleration"""
    mj = _mj()
    m, d = _robot_only(B)
    r = B.rig(m, "", hand="paddle")
    bl = r.servo.base.lim
    r.place(d, r.rest); mj.mj_forward(m, d)
    worst_v, worst_a = np.zeros(2), np.zeros(2)
    u = r.rest.copy()
    for goal_xyz in ((1.0, 0.0, 0.6), (1.0, 1.0, -0.4), (-0.5, 1.0, 0.0), (0.0, 0.0, 0.0)):
        goal = r.rest.copy(); goal[0:3] = goal_xyz
        for _ in range(250):
            r.begin_step(d, goal)
            for j in range(1, r.substeps + 1):
                r.apply(d, u, goal, j)
                pb = r.servo.base
                if bl.mode == "axes":
                    worst_v = np.maximum(worst_v, np.abs(pb.v) / bl.speed)
                    worst_a = np.maximum(worst_a, np.abs(pb.a) / bl.accel)
                else:
                    worst_v = np.maximum(worst_v, np.linalg.norm(pb.v) / bl.speed)
                    worst_a = np.maximum(worst_a, np.linalg.norm(pb.a) / bl.accel)
                mj.mj_step(m, d)
            u = goal
    ok = worst_v.max() <= 1 + 1e-3 and worst_a.max() <= 1 + 1e-3
    R.add("4 base setpoints within the limits (robot frame)", "PASS" if ok else "FAIL",
          f"{bl.mode}: speed {np.round(worst_v, 4).tolist()} x, acceleration {np.round(worst_a, 4).tolist()} x")
    return ok


def _limiter_peaks(gen, lim, rng, dt=0.001):
    pv, pa, pj, prev = np.zeros(lim.n), np.zeros(lim.n), np.zeros(lim.n), np.zeros(lim.n)
    for _ in range(8):
        r = rng.uniform(lim.range_lo + 0.1, lim.range_hi - 0.1)
        for _ in range(800):
            q, qd, qdd = gen.step(r)
            pv, pa = np.maximum(pv, np.abs(qd)), np.maximum(pa, np.abs(qdd))
            pj = np.maximum(pj, np.abs(qdd - prev) / dt)
            prev = qdd.copy()
    rv = float(np.max(pv / lim.vel))
    ra = float(np.max(pa / lim.acc)) if lim.acc is not None else None
    rj = float(np.max(pj / lim.jerk)) if lim.jerk is not None else None
    ok = rv <= 1 + TOL and (ra is None or ra <= 1 + TOL) and (rj is None or rj <= 1 + 1e-3)
    txt = f"speed {rv:.4f} x, acceleration {ra if ra is None else round(ra, 4)} x, jerk {rj if rj is None else round(rj, 4)} x"
    return ok, txt


def check_arm_limiter(name, R: Report):
    """4b the arm entries (limits only): the library's setpoint generator (servo.SetpointGenerator, what a sport
    adopting the arm uses) keeps random references within the limits; limits.JerkLimiter (fencing's primitive,
    unchanged) is reported"""
    from bodies import servo as SV
    lim = AL.load(name, HERE)
    ok, txt = _limiter_peaks(SV.SetpointGenerator(lim, 0.001, q0=np.zeros(lim.n)), lim, np.random.default_rng(5))
    R.add("4b SetpointGenerator setpoints within the limits", "PASS" if ok else "FAIL", txt)
    ok2, txt2 = _limiter_peaks(AL.JerkLimiter(lim, 0.001, q0=np.zeros(lim.n)), lim, np.random.default_rng(5))
    R.add("4b limits.JerkLimiter (fencing's primitive, recorded)", "INFO" if ok2 else "WARN",
          txt2 + ("" if ok2 else ": its velocity guard overshoots by up to acc x 1 ms in discrete steps; "
                  "servo.SetpointGenerator keeps that margin"))


def check_leg_output(name, R: Report):
    """4c the limits-only legged entries (LEGS): the library's output stage (servo.Actuators with the body's [motors]
    curves) never lets a motor drive a joint along its motion beyond the curve or exceed the torque, on random torques
    and joint speeds in a toy model of one leg; servo.WalkCommand keeps the range and the rate"""
    from bodies import servo as SV
    mj = _mj()
    decl = tomllib.loads((HERE / name / "body.toml").read_text())
    L, W = decl["limits"]["leg"], decl["limits"]["walk"]
    keys = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    joints = "".join(f'<body pos="0 0 {0.1 * (i + 1)}"><joint name="j{i}" axis="0 1 0"/><geom size="0.02" mass="1"/>'
                     for i in range(6)) + "</body>" * 6
    motors = "".join(f'<motor joint="j{i}" ctrlrange="-1000 1000"/>' for i in range(6))
    m = mj.MjModel.from_xml_string(f"<mujoco><worldbody>{joints}</worldbody><actuator>{motors}</actuator></mujoco>")
    d = mj.MjData(m)
    tau_max = np.asarray(L["torque_nm"], float)
    act = SV.Actuators(m, np.arange(6), np.arange(6), tau_max, 0.001, envelope=SV.unitree_envelope(decl, keys))
    X1, X2, Y1, Y2 = SV.unitree_envelope(decl, keys)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(2000):
        d.qvel[:] = rng.uniform(-1.2, 1.2, 6) * np.asarray(L["speed_rad_s"], float)
        t = act.write(d, rng.uniform(-2, 2, 6) * tau_max)
        a = np.abs(d.qvel)
        along = np.minimum(Y1 * np.clip((X2 - a) / (X2 - X1), 0, 1), tau_max)
        lim = np.where(t * d.qvel > 0, along, np.minimum(Y2, tau_max))
        worst = max(worst, float(np.max(np.abs(t) - lim)))
    ok = worst <= 1e-9 and np.all(X2 <= np.asarray(L["speed_rad_s"], float) + 1e-9)
    R.add("4c Actuators within the motor curves and torques", "PASS" if ok else "FAIL",
          f"worst excess {worst:.1e} N m; curves end at the URDF speeds")
    wc = SV.WalkCommand(W, 0.02)
    prev, bad = np.zeros(3), 0
    for _ in range(500):
        c = wc(rng.uniform(-3, 3, 3)).copy()
        bad += int(np.any(np.abs(c) > np.asarray(W["command_max"]) + 1e-12) or
                   np.any(np.abs(c - prev) > np.asarray(W["command_accel"]) * 0.02 + 1e-12))
        prev = c
    R.add("4c WalkCommand within the range and the rate", "PASS" if not bad else "FAIL", f"{bad} violations in 500")


# ------------------------------------------------------------------------------------------------ 5 appearance
def _compare(B, lod_a, lod_b):
    """two mesh levels of the robot at 3 random poses: (max body offset, max vertex-box offset, max centroid offset,
    {mesh: box offset > 1 cm}, number of mesh geoms)"""
    mj = _mj()
    kw = {"fixed_base": True} if _legged(B) else {}
    ms = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", lod=lod_a, **kw))
    mf = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", lod=lod_b, **kw))
    ds, df = mj.MjData(ms), mj.MjData(mf)
    rs, rf = B.rig(ms, "", hand="paddle"), B.rig(mf, "", hand="paddle")
    rng = np.random.default_rng(11)
    worst_body, worst_c, worst_box, dev = 0.0, 0.0, 0.0, {}
    n = sum(ms.geom_type[g] == mj.mjtGeom.mjGEOM_MESH for g in range(ms.ngeom))
    for _ in range(3):
        u = _random_u(rs, rng)
        rs.place(ds, u); rf.place(df, u)
        mj.mj_kinematics(ms, ds); mj.mj_kinematics(mf, df)
        worst_body = max(worst_body, float(np.max(np.abs(ds.xpos - df.xpos))))
        for g in range(ms.ngeom):
            if ms.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                continue
            vs, vf = _world_verts(ms, ds, g), _world_verts(mf, df, g)
            worst_c = max(worst_c, float(np.linalg.norm(vs.mean(0) - vf.mean(0))))
            e = float(np.max(np.abs(np.r_[vs.min(0) - vf.min(0), vs.max(0) - vf.max(0)])))
            worst_box = max(worst_box, e)
            if e > 0.01:
                k = ms.mesh(ms.geom_dataid[g]).name
                dev[k] = max(dev.get(k, 0.0), e)
    return worst_body, worst_box, worst_c, dev, n


def _dev_text(dev, worst_box, worst_c, what):
    if not dev:
        return f"vertex boxes within {worst_box * 1000:.1f} mm, centroids within {worst_c * 1000:.1f} mm"
    return f"geoms whose {what} differ by more than 1 cm: " + ", ".join(
        f"{k} {v * 1000:.0f} mm" for k, v in sorted(dev.items(), key=lambda x: -x[1]))


def check_appearance(B, R: Report):
    """5 the players' level (meshes_25k/, in git) and the fine meshes (meshes_fine/, not in git) on the simulation's
    kinematics"""
    import json
    d = HERE / B.name
    lv = d / B.decl["body"]["meshes"].get("25k", "meshes_25k")
    probs = []
    if not (lv / "lod.json").exists():
        probs.append("no meshes_25k/lod.json (python -m bodies.lod build)")
    else:
        man = json.loads((lv / "lod.json").read_text())
        probs += [f"{f}: sha256 differs from lod.json" for f, e in man["files"].items()
                  if not (lv / f).exists() or assembly._sha(lv / f) != e["sha256"]]
    R.add("5 players' level of detail committed (25k)", "FAIL" if probs else "PASS",
          "; ".join(probs) or f"{man['triangles']} triangles in {len(man['files'])} files (from {man['source_triangles']})")
    if probs:
        return
    wb, box, c, dev, n = _compare(B, "standard", "25k")
    R.add("5 appearance meshes (25k) = simulation kinematics", "PASS" if wb < 1e-9 else "FAIL",
          f"{n} mesh geoms on identical bodies (max body offset {wb:.0e} m at 3 random poses)")
    R.add("5 25k vs simulation mesh geometry (recorded)", "WARN" if dev else "PASS",
          _dev_text(dev, box, c, "players' (25k) and simulation meshes"))
    fine = d / B.decl["body"]["meshes"]["fine"]
    if not fine.is_dir() or not any(fine.iterdir()):
        R.add("5 fine meshes", "SKIP", f"{fine.relative_to(HERE.parent)} absent (not in git; tools/make_model.py)")
        return
    wb, box, c, dev, n = _compare(B, "25k", "fine")
    R.add("5 25k vs fine meshes (decimation error)", "PASS" if wb < 1e-9 and not dev else "WARN",
          _dev_text(dev, box, c, "25k and fine meshes"))
    wb, box, c, dev, n = _compare(B, "standard", "fine")
    R.add("5 fine vs simulation mesh geometry (recorded)", "WARN" if dev else "PASS",
          _dev_text(dev, box, c, "fine and simulation meshes"))


def _world_verts(m, d, g):
    """the world positions of the vertices a mesh geom's triangles use (an unreferenced vertex is not drawn)"""
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    used = np.unique(m.mesh_face[fa:fa + fn].reshape(-1))
    v = m.mesh_vert[a:a + n][used[used < n]]
    return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T


# ------------------------------------------------------------------------------------------------ 6 replay
def _run(B, seconds=3.0, walker=None):
    mj = _mj()
    if _legged(B):
        root = ET.fromstring(WORLD)
        B.mount(root, "", (0.0, 0.0, 0.0), 0.0, hand="paddle")
        m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mj.MjData(m)
    else:
        m, d = _robot_only(B)
    r = B.rig(m, "", hand="paddle", **_walker(walker))
    r.place(d, r.rest); mj.mj_forward(m, d)
    rng = np.random.default_rng(42)
    u = r.rest.copy()
    h = hashlib.sha256()
    n = int(seconds / r.dt)
    goal = u
    for k in range(n):
        if k % 50 == 0:
            goal = _random_u(r, rng, margin=0.2)
            goal[0:3] = (0.5, 0.2, 0.3) if _legged(B) else (r.rest[0:3] if r.servo.base is None
                                                            else rng.uniform(-0.3, 0.3, 3))
        nxt = u + np.clip(goal - u, -r.STEP, r.STEP)
        _step(r, m, d, u, nxt)
        u = nxt
        h.update(d.qpos.tobytes()); h.update(d.qvel.tobytes())
    return h.hexdigest()


def check_replay(B, R: Report):
    a, b = _run(B), _run(B)
    R.add("6 deterministic replay", "PASS" if a == b else "FAIL", f"3 s, state hash {a[:16]} twice")


# ------------------------------------------------------------------------------------------------ 8 control libraries
def _arbiter(B, walker=None):
    """A legged robot (a table-tennis robot: paddle on the right) placed standing, on the arbiter: (m, d, rig, arbiter,
    run); run(n) runs n control steps and returns the lowest pelvis height."""
    from bodies.control.arbiter import Arbiter
    mj = _mj()
    root = ET.fromstring(WORLD)
    B.mount(root, "", (0.0, 0.0, 0.0), 0.0, hand="paddle")
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    r = B.rig(m, "", hand="paddle", **_walker(walker))
    r.place(d, r.rest); mj.mj_forward(m, d)
    arb = Arbiter(r)
    arb.reset(d)

    def run(n):
        zmin = 9.0
        for _ in range(n):
            arb.begin_step(d)
            for j in range(1, r.substeps + 1):
                arb.apply(d, j)
                mj.mj_step(m, d)
            zmin = min(zmin, float(d.xpos[r.base_body][2]))
        return zmin
    return m, d, r, arb, run


def check_control(B, R: Report):
    """8 the optional control libraries run on the robot (not an admission requirement: reported)."""
    mj = _mj()
    if not _legged(B):
        from bodies.control.wholebody import WholeBodyPlanner
        P = WholeBodyPlanner(B, hand="paddle")
        rng = np.random.default_rng(2)
        solved = 0
        for _ in range(4):
            u = _random_u(P.rig, rng, spread=0.4)
            pos, Rm = P.fk(u)
            best = min((P.ik(pos, Rm[:, 0], s0) for s0 in (P.rest, u + 0.05)), key=lambda x: x[1])
            solved += best[1] < 2e-3
        R.add("8 wholebody planner IK on the robot", "PASS" if solved else "WARN", f"{solved}/4 tool poses solved")
        return
    from bodies.control import kinematics as KIN
    m, d, r, arb, steps = _arbiter(B)
    zmin = [9.0]

    def run(n):
        zmin[0] = min(zmin[0], steps(n))

    probs = []
    arb.walk(d, (0.5, 0.0, 0.0)); run(150)
    A = arb.GROUPS["arms"]
    q0 = arb.target()[A]
    rows = np.array([q0 + (k + 1) / 100 * np.r_[0.3, np.zeros(len(A) - 1)] for k in range(100)])
    why = arb.plan(d, "arms", rows)
    if why:
        probs.append(f"arm plan refused: {why}")
    if arb.plan(d, "arms", rows) is None:
        probs.append("a second plan within 20 ms was accepted")
    run(2)
    bad = arb.target()[A] + rows - rows[0]; bad[50:, 0] += 1.0
    if arb.plan(d, "arms", bad) is None:
        probs.append("a plan with a jump beyond the speed limit was accepted")
    run(60)
    WA = np.r_[arb.GROUPS["waist"], A]                    # table tennis: legs walk, the agent plans waist + arms
    q1 = arb.target()[WA]
    swing = np.array([q1 + 0.2 * math.sin(math.pi * (k + 1) / 80) * (np.arange(len(WA)) == len(arb.GROUPS["waist"]))
                      for k in range(80)])
    why = arb.plan(d, ["waist", "arms"], swing)
    if why:
        probs.append(f"waist + arms plan refused: {why}")
    run(100)
    if arb.owner["legs"] != "walk":
        probs.append(f"legs left the walking policy during the waist + arms plan ({arb.owner['legs']})")
    arb.walk(d, (0.0, 0.0, 0.0)); run(50)
    Lg = arb.GROUPS["legs"]
    hold = np.repeat(d.qpos[arb.qadr[Lg]][None], 30, axis=0)
    why = arb.plan(d, "legs", hold)
    if why:
        probs.append(f"legs plan refused: {why}")
    run(30)
    arb.release(d, "legs")
    run(250)
    ev = [e[2] for e in arb.events]
    up = zmin[0] > 0.45 * B.decl["body"]["height_m"]
    if not up:
        probs.append(f"fell (pelvis {zmin[0]:.2f} m)")
    if "walk_reinit" not in ev:
        probs.append("the legs were not handed back to the walking policy")
    R.add("8 arbiter: walk, arm plan, legs plan, handover", "FAIL" if probs else "PASS",
          "; ".join(probs) or f"events {sorted(set(ev))}, pelvis >= {zmin[0]:.2f} m, {arb.switches()} owner switches")
    with tempfile.TemporaryDirectory() as t:
        xml = Path(t) / "robot.xml"
        KIN.export_robot_only(xml, meshdir=str(HERE / B.name / B.decl["body"]["meshes"]["standard"]), robot=B.name)
        K = KIN.Kinematics(xml, robot=B.name)
        jf = r.fqadr
        mj.mj_kinematics(m, d)                           # the site positions of the present state
        K.set_state(d.qpos[jf:jf + 3], d.qpos[jf + 3:jf + 7], d.qpos[arb.qadr])
        e = float(np.linalg.norm(K.site_pose("left_foot")[0] - d.site_xpos[arb.foot_site["left"]]))
    R.add("8 kinematics library = simulation", "PASS" if e < 1e-9 else "FAIL", f"left foot offset {e:.1e} m")


# ------------------------------------------------------------------------------------------------ 7 assembly
def check_assembly(name, R: Report):
    is_arm = name in ARMS or name in LEGS                   # limits-only entries: no attachment, no controllers
    hand = None if is_arm else "paddle"
    ctl = [] if is_arm else (["wholebody"] if not tomllib.loads((HERE / name / "body.toml").read_text())["body"].get(
        "kind") == "legged" else ["arbiter", "kinematics"])
    a = assembly.assemble(name, hand, controllers=ctl)
    b = assembly.assemble(name, hand, controllers=ctl)
    probs = []
    if a.hash != b.hash:
        probs.append("two assemblies differ")
    if not is_arm:
        c = assembly.assemble(name, "racket", controllers=ctl)
        e = assembly.assemble(name, hand, controllers=[])
        if c.hash == a.hash or e.hash == a.hash:
            probs.append("the hash ignores the attachment or the controllers")
    with tempfile.TemporaryDirectory() as t:                 # a copy as tools/sportlib.copy_body makes it
        dst = Path(t) / "bodies"
        dst.mkdir()
        for f in ("__init__.py", "sources.py", "assembly.py", *assembly.SHARED):   # SHARED: the hashed shared files
            shutil.copy2(HERE / f, dst / f)
        shutil.copytree(HERE / "hands", dst / "hands", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(HERE / name, dst / name, ignore=shutil.ignore_patterns(
            "meshes_fine", "tools", "README.md", "__pycache__", "motion.pt", ".gitignore"))
        if ctl:
            (dst / "control").mkdir()
            for c_ in ctl:
                shutil.copy2(HERE / "control" / f"{c_}.py", dst / "control" / f"{c_}.py")
        a.write(dst / assembly.FILE_NAME)
        same, cid = assembly.verify(Path(t))
        if not same:
            probs.append("a task's copy hashes differently")
    R.add("7 assembled configuration hash stable", "FAIL" if probs else "PASS", "; ".join(probs) or a.id)
    return a.id


# ------------------------------------------------------------------------------------------------ 9 self-collision
def _pair_depth(m, d, pairs):
    """the deepest penetration (m, >= 0) among the model's pairs at d's positions"""
    mj = _mj()
    worst = 0.0
    for i in pairs:
        x = mj.mj_geomDistance(m, d, int(m.pair_geom1[i]), int(m.pair_geom2[i]), 0.01, None)
        worst = max(worst, -x)
    return worst


def _drive(B, m, u0, u1, seconds_hold=1.0, probe=None):
    """the hand arm from u0 to u1 at its speed limits (a linear ramp), then held; probe(d) every control step"""
    mj = _mj()
    d = mj.MjData(m)
    r = B.rig(m, "", hand="paddle")
    r.place(d, u0)
    mj.mj_forward(m, d)
    span = np.abs(np.asarray(u1) - np.asarray(u0))[3:] / r.VEL[3:]
    n = int(math.ceil(float(np.max(span)) / r.dt * 1.25)) + 1
    rows = [u0 + (u1 - u0) * min(1.0, k / n) for k in range(n + 1)] + [u1] * int(seconds_hold / r.dt)
    worst = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        _step(r, m, d, a, b)
        if probe is not None:
            worst = max(worst, probe(d))
    return r, d, worst


def check_selfcollision(B, R: Report):
    from bodies import selfcollision as SC
    mj = _mj()
    fails, warns, rep = SC.check_body(B.name)
    rp = rep.get("paddle", {})
    R.add("9 self-collision declared, clear at rest", "FAIL" if fails else "PASS", "; ".join(fails) or
          f"{rp.get('pairs')} pairs with the paddle, rest clearance {rp.get('rest_closest', [0, 0])[1] * 100:.1f} cm, "
          f"{rp.get('random_poses_with_self_contact')}/{rp.get('random_poses')} random legal poses in self-contact")
    if warns:
        R.add("9 self-collision with draft hands", "WARN", "; ".join(warns[:2]) + (" ..." if len(warns) > 2 else ""))
    kw = {"fixed_base": True} if _legged(B) else {}
    on = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", **kw))
    off = mj.MjModel.from_xml_string(B.robot_only_xml(hand="paddle", self_collision=False, **kw))
    pairs = [i for i in range(on.npair) if SC.is_tag(mj.mj_id2name(on, mj.mjtObj.mjOBJ_PAIR, i), "selfcol")]
    d = mj.MjData(on)
    r = B.rig(on, "", hand="paddle")
    r.place(d, r.rest)
    q0 = d.qpos.copy()
    sample = SC.rig_sampler(r, on, q0)
    rng = np.random.default_rng(9)
    best, target = 0.0, None
    for _ in range(400):                      # a legal target pose with the hand arm deep inside the body
        q = sample(rng)
        other = getattr(r, "other_qadr", r.qadr[-7:])   # a one-armed robot has none
        q[other] = q0[other]                   # the other arm at rest: only the hand arm (u) moves
        d.qpos[:] = q
        mj.mj_kinematics(on, d)
        x = _pair_depth(on, d, pairs)
        if x > best:
            best, target = x, r.u_of(d)
        if best > 0.08:
            break
    u0 = np.array(r.rest, float)
    target[0:3] = u0[0:3]
    dq = mj.MjData(on)

    def depth_on_model(dd):                   # the penetration the same positions have in the self-collision model
        dq.qpos[:] = dd.qpos
        mj.mj_kinematics(on, dq)
        return _pair_depth(on, dq, pairs)

    _, _, pen_off = _drive(B, off, u0, target, probe=depth_on_model)
    r_on, _, pen_on = _drive(B, on, u0, target, probe=depth_on_model)
    rec = r_on.self_collisions()
    ok = pen_on < 0.015 and pen_off >= 0.03 and rec["events"] >= 1 and rec["peak_force_n"] > 0
    R.add("9 self-collision blocks the arm", "PASS" if ok else "FAIL",
          f"target {best * 100:.1f} cm inside the body: off {pen_off * 100:.1f} cm, on {pen_on * 100:.2f} cm; "
          f"recorded {rec['events']} events, peak {rec['peak_force_n']:.0f} N "
          f"(10 ms mean {rec['peak_force_10ms_n']:.0f} N), "
          f"warnings {rec['warnings']}, "
          f"worst {rec['worst_pairs'][:1]}")


# ------------------------------------------------------------------------------------------------ main
def run(name) -> Report:
    R = Report(name)
    t = time.time()
    print(f"{name}:", flush=True)
    check_declared(name, R)
    if name in ARMS:
        check_arm_limiter(name, R)
    elif name in LEGS:
        check_leg_output(name, R)
    else:
        B = load_body(name)
        for fn in (check_scene, check_servo, check_limits, check_appearance, check_replay, check_control,
                   check_selfcollision):
            try:
                fn(B, R)
            except Exception as e:  # noqa: BLE001
                R.add(fn.__name__, "FAIL", f"{type(e).__name__}: {e}")
    check_assembly(name, R)
    R.add("time", "INFO", f"{time.time() - t:.0f} s")
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("robots", nargs="*", default=list(BODIES) + list(ARMS) + list(LEGS))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    reps = [run(n) for n in a.robots]
    out = {R.robot: dict(ok=R.ok(), checks=R.rows) for R in reps}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    print("conformance:", ", ".join(f"{k} {'PASS' if v['ok'] else 'FAIL'}" for k, v in out.items()))
    return 0 if all(v["ok"] for v in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
