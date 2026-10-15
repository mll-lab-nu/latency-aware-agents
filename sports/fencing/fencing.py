"""Core of the robot epee fencing service: the world (MuJoCo), the fencers' motion controllers and the referee.

FIE Technical Rules (t.) and Material Rules (m.); see the task README for the article list and the deviations.

World (metres, seconds): piste surface z = 0, long axis x, centre line x = 0, on-guard lines x = +-2, rear limits
x = +-7 (t.18-t.19). Fencer A stands at x < 0 facing +x, fencer B at x > 0 facing -x (A turned by 180 degrees about
z: both right-handed, exactly symmetric). Each seat sees everything in its OWN frame: x' = s x, y' = s y, z' = z with
s = +1 for A and -1 for B, so in its own frame every fencer stands at x' < 0 and faces +x'; its rear limit is
x' = -7.

Fencer: a cart on a rail along the piste (footwork: the rail coordinate, own frame; the rail is a Festo EGC-TB-185
toothed-belt axis run at RAIL_VMAX, RAIL_AMAX and RAIL_JMAX, its motor force at most RAIL_FMAX), a 0.90 m pedestal,
a Franka Panda arm held to the real Panda's limits (bodies/franka_panda/body.toml: joint velocity, acceleration,
jerk, torque and torque rate; bodies/limits.py) and an epee (grip, 13.5 cm bell guard, 0.90 m blade, the point). The
arm's servo runs every 1 ms physics step: a jerk-limited setpoint generator (JerkLimiter) follows the controller's
reference, the motor torque is the inertia feed-forward of the setpoint's acceleration + PD on the setpoint +
gravity/Coriolis compensation, then limited in rate (1000 N m/s) and size (87 / 12 N m). Every contact between the
two fencers (blades, guards, arms, bodies) is simulated by MuJoCo (soft contacts, 1 ms steps).

Commands (one per control step of 10 ms; a command replaces the one under way):
  arm   the point's position (in the cart frame: u forward, v left, z up, origin on the floor under the cart's centre)
        and the blade's direction at T seconds; executed by IK (point position + blade direction) and a time-optimal
        straight-line joint move under the velocity, acceleration and jerk limits (an S-curve at PLAN_MARGIN of the
        limits, stretched to T if T is longer).
  rail  a goal x' for the cart and a speed cap (<= RAIL_VMAX); a time-optimal go-to under the rail's velocity,
        acceleration and jerk limits (rail_goto), sampled at every 1 ms physics step for the rail's servo.

Touch (t.90-t.93, m.19, Material Rules Appendix B, epee): the point registers when the force pressing it along the
blade (the contact normal force's component along the blade axis) exceeds 7.35 N (750 g) on the opponent's body
(any part of the robot: cart, pedestal, arm links, t.91), not on the guard, the blade, the grip or the floor, for at
least 2 ms (3 consecutive 1 ms samples). The first touch locks out the other side DOUBLE_MS later; a touch by the
other within DOUBLE_MS is a double (both lamps, t.92).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import mujoco as mj

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent                      # sim/ in a task (bodies/ next to sport/); sports/ in a checkout
sys.path.insert(0, str(_ROOT if (_ROOT / "bodies").is_dir() else _ROOT.parent))
from bodies import limits as AL  # noqa: E402
from bodies import selfcollision as SC  # noqa: E402  (the arm's self-collision record, bodies/README.md §4.1)
SCENE = Path(os.environ.get("RCB_FENCING_SCENE", HERE / "scene" / "fencing_scene.xml"))

CONTROL_DT, SUBSTEPS, PHYS_DT = 0.01, 10, 0.001
SEATS = ("A", "B")
SIGN = {"A": 1.0, "B": -1.0}
# ---- piste (t.18, t.19)
PISTE_HALF, ON_GUARD, RAIL_LIM, CART_HALF = 7.0, 2.0, 7.6, 0.25
# ---- fencer
# The arm is a Franka Panda held to the manufacturer's joint limits (bodies/franka_panda/body.toml, from Franka's
# "Control Interface Specification and Robot Limits"): velocity 2.175 / 2.61 rad/s, acceleration 15 / 7.5 / 10 / 12.5
# / 15 / 20 / 20 rad/s^2, jerk 7500 / 3750 / 5000 / 6250 / 7500 / 10000 / 10000 rad/s^3, torque 87 / 12 N m, torque
# rate 1000 N m/s. (Revision fencing-2 used 1.5 x the Panda's velocity and acceleration limits and no jerk or torque-
# rate limits; user decision 2026-10-07: every arm is as capable as the real robot, no more.)
ARM = AL.load("franka_panda")
ARM_VEL, ARM_ACC, ARM_JERK = ARM.vel, ARM.acc, ARM.jerk
PLAN_MARGIN = 0.95             # planned moves use 95 % of the velocity, acceleration and jerk limits: the setpoint
                               # generator's headroom to follow them without lag (it never exceeds 100 %)
# The rail (fencing-4): the cart runs on a Festo EGC-TB-185 toothed-belt axis, whose catalogue limits are 5 m/s,
# 50 m/s^2 and a 2500 N feed force (Festo, "Toothed belt axis EGC-TB", operating instructions 8162896, 2023-05). The
# sport runs it inside that envelope: 2 m/s, 6 m/s^2 (as since fencing-1), the motor's force at most 2500 N (the
# scene's rail motor, fencing-3: 3000 N) and a jerk limit of 150 m/s^3, an ESTIMATE (Festo publishes no jerk limit; a
# servo drive runs the axis on jerk-limited profiles; the value of the B9 feasibility study,
# olympiad_review/b9_dynamic_sports_zh.md §2). Up to fencing-3 the rail was not modelled on a real product and had
# no jerk limit (its go-to stepped the acceleration, 8-11 m/s^2 at the stop).
RAIL_VMAX, RAIL_AMAX, RAIL_JMAX = 2.0, 6.0, 150.0
RAIL_FMAX = 2500.0             # N: the rail motor's force limit (scene/gen_scene.py, ctrlrange)
RAIL_PRODUCT = dict(model="Festo EGC-TB-185 toothed-belt axis", vmax=5.0, amax=50.0, force_n=2500.0,
                    source="Festo, Toothed belt axis EGC-TB, operating instructions 8162896 (2023-05)",
                    jerk="150 m/s^3: an estimate (Festo publishes none)")
RAIL_FF = 1.0                  # the cart's measured acceleration fed forward into the arm torques (M_arm,rail x a_cart)
RESYNC = 0.02                  # rad: a joint pushed farther than this from its setpoint (a hit, a bind) makes the
                               # setpoint follow the arm (at most RESYNC behind, within the velocity limit): the servo,
                               # its torque rate limited, does not wind up
KP = np.array([6e4, 2400.0, 2400.0, 2400.0, 2400.0, 1000.0, 600.0, 200.0])
KD = np.array([4e3, 100.0, 100.0, 100.0, 100.0, 30.0, 20.0, 6.0])
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
PED_H = 0.90
TRUNK = ("cart_g", "pedestal", "link0_c", "link1_c", "link2_c")     # a fencer's trunk (corps-a-corps, t.25)
# ---- apparatus (m.19.2, Appendix B)
F_TOUCH = 7.35                 # N: 750 g on the point
T_TOUCH_SAMPLES = 3            # >= 2 ms: three consecutive 1 ms samples above F_TOUCH
DOUBLE_MS = 45                 # lockout: a touch by the other within 45 ms is a double (40-50 ms, Appendix B)
# ---- the on-guard pose: the point (cart frame) and blade direction
GUARD_TIP = np.array([1.40, -0.05, 1.10])
GUARD_DIR = np.array([1.0, 0.0, -0.05])
GUARD_DIR = GUARD_DIR / np.linalg.norm(GUARD_DIR)
Q_GUARD0 = np.array([0.943, 0.75, -1.497, -2.498, 1.345, 0.838, -1.279])   # on-guard arm (refined by IK at import)
IK_OFFSETS = np.array([[0, 0.4, 0, 0.5, 0, 0, 0], [0, -0.4, 0, -0.3, 0, 0, 0], [0.5, 0, 0, 0, 0, 0, 0],
                       [-0.5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.6, 0.4, 0], [0, 0, 0, 0, -0.6, -0.4, 0]], float)
IK_OK = 2e-3

REVISION = "fencing-6"   # 6: the arms' setpoint generator brakes within the jerk in force (bodies/limits.JerkLimiter)


def _rail_rest(v, a, amax, jmax):
    """how far the rail still goes (towards +) from velocity v and acceleration a if it brakes to rest (v = a = 0) as
    hard as the acceleration and jerk limits allow; -inf if it is moving away (v, a <= 0)"""
    if v <= 0.0 and a <= 0.0:
        return 0.0 if v == 0.0 and a == 0.0 else -math.inf
    q = jmax * v + 0.5 * a * a
    if q < 0.0:
        return -math.inf
    ar = -math.sqrt(q)                   # the deceleration at which to ramp back (no constant-deceleration phase)
    if ar > a:                           # already braking harder than needed: ease off now, v reaches 0 first
        t = (-a - math.sqrt(max(0.0, a * a - 2.0 * jmax * v))) / jmax
        return v * t + 0.5 * a * t * t + jmax * t ** 3 / 6.0
    if ar >= -amax:
        t1 = (a - ar) / jmax
        v1 = v + a * t1 - 0.5 * jmax * t1 * t1
        t3 = -ar / jmax
        return (v * t1 + 0.5 * a * t1 * t1 - jmax * t1 ** 3 / 6.0) + \
            (v1 * t3 + 0.5 * ar * t3 * t3 + jmax * t3 ** 3 / 6.0)
    t1 = (a + amax) / jmax               # ramp to -amax, hold, ramp back
    v1 = v + a * t1 - 0.5 * jmax * t1 * t1
    t2 = (v1 - 0.5 * amax * amax / jmax) / amax
    v2 = v1 - amax * t2
    t3 = amax / jmax
    return (v * t1 + 0.5 * a * t1 * t1 - jmax * t1 ** 3 / 6.0) + (v1 * t2 - 0.5 * amax * t2 * t2) + \
        (v2 * t3 - 0.5 * amax * t3 * t3 + jmax * t3 ** 3 / 6.0)


def rail_goto(x, v, a, goal, vmax, amax=RAIL_AMAX, jmax=RAIL_JMAX, dt=CONTROL_DT):
    """One control step of the cart's go-to generator on its rail: the largest constant jerk for the step (towards the
    goal) after which the rail can still brake to rest at the goal, its speed never above vmax, |a| <= amax, |jerk| <=
    jmax (bisection; time-optimal up to the step). Returns (x, v, a, jerk) at the end of the step; the step's motion
    is the cubic x + v t + a t^2 / 2 + jerk t^3 / 6."""
    e = goal - x
    if abs(e) < 1e-9 and abs(v) < 1e-7 and abs(a) < 1e-5:
        return goal, 0.0, 0.0, 0.0                                  # at rest at the goal
    # the last few micrometres: the deadbeat jerks that put the rail at rest exactly on the goal in three steps, if
    # they are within the jerk limit (the discrete steps cannot follow the braking curve's switching times exactly)
    db = (-(-e) / dt ** 3 - 2.0 * v / dt ** 2 - 11.0 * a / (6.0 * dt), 2.0 * (-e) / dt ** 3 + 3.0 * v / dt ** 2 +
          7.0 * a / (6.0 * dt), -(-e) / dt ** 3 - v / dt ** 2 - a / (3.0 * dt))
    if max(abs(db[0]), abs(db[1]), abs(db[2])) <= jmax:
        j = db[0]
        return x + v * dt + 0.5 * a * dt * dt + j * dt ** 3 / 6.0, v + a * dt + 0.5 * j * dt * dt, a + j * dt, j
    sg = (1.0 if e > 0.0 else -1.0) if e != 0.0 else (-1.0 if v > 0.0 else 1.0)
    ep, vp, ap = sg * e, sg * v, sg * a                             # the goal ahead (+)
    jlo, jhi = max(-jmax, (-amax - ap) / dt), min(jmax, (amax - ap) / dt)

    def ok(j):
        a1, v1 = ap + j * dt, vp + ap * dt + 0.5 * j * dt * dt
        peak = v1 + (0.5 * a1 * a1 / jmax if a1 > 0.0 else 0.0)     # the speed it reaches before a can be zero
        if ap > 0.0 > a1:
            peak = max(peak, vp - 0.5 * ap * ap / j)                 # a crosses zero inside the step
        if peak > vmax + 1e-9:
            return False
        return vp * dt + 0.5 * ap * dt * dt + j * dt ** 3 / 6.0 + _rail_rest(v1, a1, amax, jmax) <= ep + 1e-9

    if ok(jhi):
        j = jhi
    elif not ok(jlo):
        j = jlo                                                     # brake as hard as the limits allow
    else:
        lo, hi = jlo, jhi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        j = lo
    return (x + sg * (vp * dt + 0.5 * ap * dt * dt + j * dt ** 3 / 6.0), sg * (vp + ap * dt + 0.5 * j * dt * dt),
            sg * (ap + j * dt), sg * j)


def rail_rows_why(xs, dt=CONTROL_DT, tol=1.001):
    """rail positions one control step apart (a low-level traj's rail column, joined to the rail's last targets)
    against the rail's limits by differences: speed, acceleration, jerk. None or the reason to refuse them"""
    x = np.asarray(xs, float)
    for n, (lim, what, unit) in enumerate(((RAIL_VMAX, "speed", "m/s"), (RAIL_AMAX, "acceleration", "m/s^2"),
                                           (RAIL_JMAX, "jerk", "m/s^3")), 1):
        if len(x) <= n:
            break
        d = np.abs(np.diff(x, n)) / dt ** n
        if np.any(d > lim * tol + 1e-9):
            return f"rail: {d.max():.3f} {unit} between rows exceeds its {what} limit {lim} {unit}"
    return None


def arm_ref(q_end, n=SUBSTEPS):
    """a resting arm reference for one control step: (q, qd) of n substeps at q_end"""
    q = np.repeat(np.asarray(q_end, float)[None], n, axis=0)
    return q, np.zeros_like(q)


class World:
    """Both fencers in MuJoCo, their PD control and the touch sensing (no rules: see Referee)."""

    def __init__(self):
        m = self.m = mj.MjModel.from_xml_path(str(SCENE))
        self.d = mj.MjData(m)
        self.kin = mj.MjData(m)
        self.selfcol = SC.monitors(m, {"A": "A_selfcol", "B": "B_selfcol"}, "franka_panda")   # recorded only
        nid = lambda t, n: mj.mj_name2id(m, t, n)
        J = mj.mjtObj.mjOBJ_JOINT
        self.qa, self.va, self.lo, self.hi, self.act = {}, {}, {}, {}, {}
        self.tip_site, self.tip_geom, self.target_geoms, self.body_geoms, self.blade_body = {}, {}, {}, {}, {}
        self.trunk_geoms = {}
        for s in SEATS:
            jid = [nid(J, f"{s}_rail")] + [nid(J, f"{s}_{j}") for j in ARM_JOINTS]
            self.qa[s] = np.array([m.jnt_qposadr[j] for j in jid])
            self.va[s] = np.array([m.jnt_dofadr[j] for j in jid])
            self.lo[s] = m.jnt_range[jid, 0].copy(); self.hi[s] = m.jnt_range[jid, 1].copy()
            self.act[s] = np.array([nid(mj.mjtObj.mjOBJ_ACTUATOR, f"{s}_m_rail")] +
                                   [nid(mj.mjtObj.mjOBJ_ACTUATOR, f"{s}_m{i}") for i in range(1, 8)])
            self.tip_site[s] = nid(mj.mjtObj.mjOBJ_SITE, f"{s}_tip_s")
            self.tip_geom[s] = nid(mj.mjtObj.mjOBJ_GEOM, f"{s}_tip")
            weapon = {nid(mj.mjtObj.mjOBJ_GEOM, f"{s}_{n}") for n in ("grip", "guard", "blade", "tip", "blade_vis",
                                                                   "tip_vis")}
            mine = {g for g in range(m.ngeom) if m.names[m.name_geomadr[g]:].split(b"\0", 1)[0].decode().startswith(f"{s}_")}
            self.target_geoms[s] = frozenset(mine - weapon)     # t.91: the whole body; not the weapon (earthed)
            self.body_geoms[s] = frozenset(mine - weapon)
            self.trunk_geoms[s] = frozenset(nid(mj.mjtObj.mjOBJ_GEOM, f"{s}_{n}") for n in TRUNK)
            self.blade_body[s] = nid(mj.mjtObj.mjOBJ_BODY, f"{s}_epee")
        self.floor = nid(mj.mjtObj.mjOBJ_GEOM, "floor")
        self.jl = {s: AL.JerkLimiter(ARM, PHYS_DT) for s in SEATS}            # the arm's 1 kHz setpoint generator
        self.trl = {s: AL.TorqueRateLimiter(ARM, PHYS_DT) for s in SEATS}     # the arm's torque rate and size limits
        self._M = np.zeros((m.nv, m.nv))
        self.reset({s: self.on_guard_q(-ON_GUARD) for s in SEATS})

    # ---- frames
    @staticmethod
    def to_world_x(s, xo):
        return SIGN[s] * xo

    def cart_x(self, s):
        """the cart's position in the seat's own frame"""
        return SIGN[s] * float(self.d.qpos[self.qa[s][0]])

    def on_guard_q(self, x_own):
        return np.r_[0.0, self._guard_arm()].astype(float), x_own

    def _guard_arm(self):
        if not hasattr(self, "_q_guard"):
            self._q_guard = None
            q, r = self.ik_arm(GUARD_TIP, GUARD_DIR, Q_GUARD0)
            assert r < 1e-3, r
            self._q_guard = q
        return self._q_guard

    # ---- state
    def reset(self, poses):
        """poses: seat -> (q8 with rail entry ignored, own-frame cart x)"""
        m, d = self.m, self.d
        mj.mj_resetData(m, d)
        self.target = {}
        for s in SEATS:
            q8, xo = poses[s]
            q = np.array(q8, float); q[0] = SIGN[s] * xo
            d.qpos[self.qa[s]] = q
            self.target[s] = q.copy()
        mj.mj_forward(m, d)
        for s in SEATS:                  # the servo starts at rest, holding the arm against gravity
            self.jl[s].reset(d.qpos[self.qa[s][1:]])
            self.trl[s].reset(d.qfrc_bias[self.va[s][1:]])
        self.k = 0
        self.press = {s: 0 for s in SEATS}
        self.contact_log = []

    def place(self, xs):
        """Referee placement (t.22): both fencers on guard, carts at own-frame xs[s], arms in the on-guard pose."""
        self.reset({s: (np.r_[0.0, self._guard_arm()], xs[s]) for s in SEATS})

    def as_ref(self, s, target):
        """a plain 8-vector target (rail world x, 7 arm joints) as a control step's reference: the arm interpolated
        linearly from the previous target, its velocity the step's difference"""
        t = np.asarray(target, float)
        a0 = self.target[s][1:]
        f = (np.arange(1, SUBSTEPS + 1) / SUBSTEPS)[:, None]
        qd = np.repeat(((t[1:] - a0) / CONTROL_DT)[None], SUBSTEPS, 0)
        return dict(x=float(t[0]), q=a0 + f * (t[1:] - a0), qd=qd)

    def step(self, targets):
        """One 10 ms control step. targets[s]: the seat's reference, dict(x = the rail's world x at the end of the step,
        q, qd = the arm's joint reference (positions, velocities) at the 10 substeps), or a plain 8-vector (as_ref).
        The arm's servo turns the reference into the setpoint (JerkLimiter) and the torque (feed-forward + PD + bias,
        TorqueRateLimiter); the rail's motor force is the inertia feed-forward of the reference's acceleration + PD on
        the reference + bias, the reference being the go-to's jerk-limited cubic sampled at the substeps (xs, xd, xdd)
        if it has them, else the linearly interpolated rail target (a traj row). Returns the sensing record of
        the step: per seat the touch time (ms index within the step, 0-9) or None, corps-a-corps."""
        m, d = self.m, self.d
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        ref = {s: (targets[s] if isinstance(targets[s], dict) else self.as_ref(s, targets[s])) for s in SEATS}
        prev = {s: float(self.target[s][0]) for s in SEATS}
        new = {s: float(ref[s]["x"]) for s in SEATS}
        xd = {s: (new[s] - prev[s]) / CONTROL_DT for s in SEATS}
        out = {"touch": {s: None for s in SEATS}, "corps": False, "tip_force": {s: 0.0 for s in SEATS}}
        sp, Ma = {}, {}
        for j in range(1, SUBSTEPS + 1):
            mj.mj_fullM(m, self._M, d.qM)                    # the joint-space inertia (of the last forward pass)
            for s in SEATS:
                va = self.va[s][1:]
                Ma[s] = self._M[np.ix_(va, va)]
                sp[s] = self.jl[s].step(ref[s]["q"][j - 1], ref[s]["qd"][j - 1], AL.jerk_caps(Ma[s], ARM))
            for s in SEATS:
                qa, va, ac = self.qa[s], self.va[s], self.act[s]
                if "xs" in ref[s]:
                    x_des, xd_des, xdd_des = ref[s]["xs"][j - 1], ref[s]["xd"][j - 1], ref[s]["xdd"][j - 1]
                else:
                    x_des, xd_des, xdd_des = prev[s] + (new[s] - prev[s]) * (j / SUBSTEPS), xd[s], 0.0
                f_rail = (self._M[va[0], va[0]] * xdd_des + KP[0] * (x_des - d.qpos[qa[0]])
                          + KD[0] * (xd_des - d.qvel[va[0]]) + d.qfrc_bias[va[0]])
                d.ctrl[ac[0]] = min(hi[ac[0]], max(lo[ac[0]], f_rail))
                q_sp, qd_sp, qdd_sp = sp[s]
                e = q_sp - d.qpos[qa[1:]]
                tau = (Ma[s] @ qdd_sp + self._M[va[1:], va[0]] * d.qacc[va[0]] * RAIL_FF + KP[1:] * e
                       + KD[1:] * (qd_sp - d.qvel[va[1:]]) + d.qfrc_bias[va[1:]])
                d.ctrl[ac[1:]] = self.trl[s](tau)
                if RESYNC and np.max(np.abs(e)) > RESYNC:      # the servo cannot hold the setpoint: it follows the arm
                    jl = self.jl[s]
                    big = np.abs(e) > RESYNC
                    jl.q[big] = d.qpos[qa[1:]][big] + np.sign(e[big]) * RESYNC
                    jl.qd[big] = np.clip(d.qvel[va[1:]], -jl.V, jl.V)[big]
                    jl.qdd[big] = 0.0
            mj.mj_step(m, d)
            for mon in self.selfcol.values():
                mon.update(d)
            f, corps = self._sense()
            for s in SEATS:
                out["tip_force"][s] = max(out["tip_force"][s], f[s])
                if f[s] >= F_TOUCH:
                    self.press[s] += 1
                    if self.press[s] == T_TOUCH_SAMPLES and out["touch"][s] is None:
                        out["touch"][s] = j - 1
                else:
                    self.press[s] = 0
            out["corps"] = out["corps"] or corps
        for s in SEATS:
            self.target[s] = np.r_[new[s], ref[s]["q"][-1]]
        self.k += 1
        return out

    def _sense(self):
        """axial force on each point from the opponent's body; corps-a-corps (the two bodies in contact)"""
        m, d = self.m, self.d
        f = {"A": 0.0, "B": 0.0}
        corps = False
        c6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for s, o in (("A", "B"), ("B", "A")):
                tg = self.tip_geom[s]
                if (g1 == tg and g2 in self.target_geoms[o]) or (g2 == tg and g1 in self.target_geoms[o]):
                    mj.mj_contactForce(m, d, i, c6)
                    n = np.asarray(c.frame[:3])
                    axis = d.xmat[self.blade_body[s]].reshape(3, 3)[:, 2]
                    f[s] += abs(float(c6[0])) * abs(float(n @ axis))
            # corps-a-corps (t.25): the bodies in contact, a trunk (cart, pedestal, shoulder) involved; arms touching
            # arms is fencing at close quarters (t.24) and goes on
            for a, b in ((g1, g2), (g2, g1)):
                if (a in self.trunk_geoms["A"] and b in self.body_geoms["B"]) or \
                   (a in self.trunk_geoms["B"] and b in self.body_geoms["A"]):
                    corps = True
        return f, corps

    # ---- kinematics on scratch data (never stepped)
    def _kin(self, s, q7):
        self.kin.qpos[:] = self.d.qpos
        self.kin.qpos[self.qa["A"][1:]] = q7       # IK always on fencer A's arm with its cart at x = 0
        self.kin.qpos[self.qa["A"][0]] = 0.0
        mj.mj_kinematics(self.m, self.kin)
        mj.mj_comPos(self.m, self.kin)

    def arm_inertia(self, q7):
        """the arm's joint-space inertia matrix (7 x 7, with the epee) at arm joints q7 (scratch data)"""
        self._kin("A", q7)
        mj.mj_crb(self.m, self.kin)
        M = np.zeros((self.m.nv, self.m.nv))
        mj.mj_fullM(self.m, M, self.kin.qM)
        va = self.va["A"][1:]
        return M[np.ix_(va, va)]

    def jerk_caps(self, *poses):
        """the jerk limits the torque-rate limit allows along a move through these arm poses (bodies/limits.py)"""
        return np.min([AL.jerk_caps(self.arm_inertia(q), ARM) for q in poses], axis=0)

    def fk_arm(self, q7):
        """point position and blade direction in the cart frame for arm joints q7 (identical for both fencers)"""
        self._kin("A", q7)
        p = self.kin.site_xpos[self.tip_site["A"]].copy()
        R = self.kin.xmat[self.blade_body["A"]].reshape(3, 3)
        return p, R[:, 2].copy()

    def jac_arm(self, q7):
        self._kin("A", q7)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mj.mj_jacSite(self.m, self.kin, jp, jr, self.tip_site["A"])
        return jp[:, self.va["A"][1:]], jr[:, self.va["A"][1:]]

    def ik_arm(self, pos, direction, q_init, iters=150, tol=1e-4):
        """Levenberg-Marquardt IK: the point at pos, the blade along direction (roll about the blade free).
        Returns (q, residual: position error + 0.5 |blade x direction|)."""
        q = np.asarray(q_init, float).copy()
        dt = np.asarray(direction, float); dt = dt / np.linalg.norm(dt)
        pos = np.asarray(pos, float)
        lo, hi = self.lo["A"][1:] + 1e-3, self.hi["A"][1:] - 1e-3
        lam = 1e-2

        def E(q):
            p, a = self.fk_arm(q)
            return np.r_[pos - p, 0.5 * np.cross(a, dt)], a
        e, a = E(q); c = float(e @ e)
        for _ in range(iters):
            if c < tol * tol:
                break
            jp, jr = self.jac_arm(q)
            da = np.cross(jr.T, a).T
            J = np.vstack([jp, 0.5 * np.cross(dt, da.T).T])
            dq = np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ e)
            qn = np.clip(q + dq, lo, hi)
            en, an = E(qn); cn = float(en @ en)
            if cn < c:
                q, e, a, c = qn, en, an, cn; lam = max(1e-5, lam * 0.5)
            else:
                lam *= 4.0
                if lam > 1e6:
                    break
        return q, math.sqrt(c)

    def ik_multi(self, pos, direction, q_now):
        """deterministic multi-start IK: from the current joints, the on-guard pose and fixed offsets of it; of the
        solutions within IK_OK the one nearest in joint time (max |dq| / joint speed limit), else the best"""
        starts = [np.asarray(q_now, float), self._guard_arm()] + [self._guard_arm() + o for o in IK_OFFSETS]
        best = None
        for i, s0 in enumerate(starts):
            q, r = self.ik_arm(pos, direction, s0)
            t = float(np.max(np.abs(q - q_now) / ARM_VEL))
            key = (0, t) if r < IK_OK else (1, r)
            if best is None or key < best[0]:
                best = (key, q, r)
            if i == 0 and r < IK_OK and t < 0.02:
                break
        return best[1], best[2]

    def state(self, s):
        """the seat's own robot: cart x (own frame), velocity, arm joints, point and guard in the own frame"""
        d = self.d
        sg = SIGN[s]
        tip = d.site_xpos[self.tip_site[s]].copy()
        ax = d.xmat[self.blade_body[s]].reshape(3, 3)[:, 2].copy()
        return dict(x=sg * float(d.qpos[self.qa[s][0]]), v=sg * float(d.qvel[self.va[s][0]]),
                    q=d.qpos[self.qa[s][1:]].copy(),
                    tip=np.array([sg * tip[0], sg * tip[1], tip[2]]), axis=np.array([sg * ax[0], sg * ax[1], ax[2]]))

    def state_sha(self):
        d = self.d
        h = hashlib.sha256()
        for a in (d.qpos, d.qvel, d.qacc_warmstart, d.ctrl, np.array([d.time])):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()


class Controller:
    """One fencer's motion controller: turns arm and rail commands into each control step's reference (the arm's
    joint positions and velocities at the 10 substeps, the rail target) within the limits."""

    def __init__(self, w: World, s: str):
        self.w, self.s = w, s
        self.reset()

    def reset(self):
        w, s = self.w, self.s
        self.q_t = w.target[s][1:].copy()          # the arm's reference at the end of the last control step
        self.qd_t = np.zeros(7)
        self.line = None                           # the arm move under way (bodies/limits.Line) and its clock
        self.t_line = 0.0
        self.x_t = SIGN[s] * float(w.target[s][0])  # own-frame rail target, its velocity and acceleration
        self.v_t = self.a_t = 0.0
        self.x_hist = [self.x_t] * 3               # the last three rail targets (a traj's join is checked against them)
        self.rail_goal, self.rail_vmax = self.x_t, RAIL_VMAX

    @property
    def moving(self) -> bool:
        return self.line is not None

    def remaining_s(self) -> float:
        return 0.0 if self.line is None else max(0.0, self.line.T - self.t_line)

    def arm(self, T, tip, direction):
        """the point at `tip` (cart frame) with the blade along `direction` T s from now; returns info"""
        q_goal, res = self.w.ik_multi(np.asarray(tip, float), np.asarray(direction, float), self.q_t)
        info = self.install(q_goal, T)
        info["ik_residual"] = round(float(res), 4)
        return info

    def install(self, q_goal, T):
        """a straight joint move from the current arm reference to q_goal, arriving T s from now or as soon as the
        joint velocity, acceleration and jerk limits allow (time-optimal S-curve at PLAN_MARGIN of the limits; a move
        under way is replaced: the setpoint generator brakes it within the limits)"""
        q_goal = np.asarray(q_goal, float)
        caps = self.w.jerk_caps(self.q_t, 0.5 * (self.q_t + q_goal), q_goal)
        self.line = AL.Line(self.q_t, q_goal, ARM.with_jerk(caps), T, PLAN_MARGIN)
        self.t_line = 0.0
        n = max(1, int(math.ceil(self.line.T / CONTROL_DT - 1e-9)))
        return dict(arrive_s=round(n * CONTROL_DT, 3), limited_by_joints=bool(self.line.t_min > T + 1e-9))

    def hold_arm(self):
        """stop the arm where its reference is now"""
        self.line = None

    def rail(self, goal, vmax=RAIL_VMAX):
        self.rail_goal = float(min(RAIL_LIM, max(-RAIL_LIM, goal)))
        self.rail_vmax = float(min(RAIL_VMAX, max(0.05, vmax)))

    def _rail_step(self):
        """the go-to's next control step: the rail target at its end and the samples (world frame) at the substeps"""
        x0, v0, a0 = self.x_t, self.v_t, self.a_t
        self.x_t, self.v_t, self.a_t, jerk = rail_goto(x0, v0, a0, self.rail_goal, self.rail_vmax, RAIL_AMAX, RAIL_JMAX)
        t = PHYS_DT * np.arange(1, SUBSTEPS + 1)
        xs = x0 + v0 * t + 0.5 * a0 * t * t + jerk * t ** 3 / 6.0
        xd = v0 + a0 * t + 0.5 * jerk * t * t
        xdd = a0 + jerk * t
        xs[-1] = self.x_t                          # the goal exactly when it arrives
        self._push_x()
        return SIGN[self.s] * xs, SIGN[self.s] * xd, SIGN[self.s] * xdd

    def _push_x(self):
        self.x_hist = self.x_hist[1:] + [self.x_t]

    def next_target(self):
        """the next control step's reference: dict(x = rail world x, q, qd = arm reference at the 10 substeps)"""
        if self.line is not None:
            ts = self.t_line + PHYS_DT * np.arange(1, SUBSTEPS + 1)
            qs = [self.line.at(t) for t in ts]
            q, qd = np.array([x[0] for x in qs]), np.array([x[1] for x in qs])
            self.t_line += CONTROL_DT
            if self.t_line >= self.line.T - 1e-9:
                q[-1], qd[-1] = self.line.q1, 0.0
                self.line = None
        else:
            q, qd = arm_ref(self.q_t)
        self.q_t, self.qd_t = q[-1].copy(), qd[-1].copy()
        xs, xd, xdd = self._rail_step()
        return dict(x=SIGN[self.s] * self.x_t, xs=xs, xd=xd, xdd=xdd, q=q, qd=qd)

    def row_target(self, row, nxt=None):
        """a low-level traj row (own-frame rail x, 7 arm joints) as the next control step's reference: the arm on a
        cubic Hermite segment from the current reference to the row (its velocity at the row: the central difference
        to the next row, zero at the last row), the rail at the row (its velocity and acceleration: the differences of
        the rail targets; after the last row the go-to brings the rail to rest at it within the rail's limits)"""
        row = np.asarray(row, float)
        v1 = np.zeros(7) if nxt is None else (np.asarray(nxt, float)[1:] - self.q_t) / (2 * CONTROL_DT)
        q, qd = AL.hermite(self.q_t, self.qd_t, row[1:], v1, CONTROL_DT, SUBSTEPS)
        self.line = None
        self.q_t, self.qd_t = row[1:].copy(), v1.copy()
        v = (float(row[0]) - self.x_t) / CONTROL_DT
        self.x_t, self.v_t, self.a_t = float(row[0]), v, (v - self.v_t) / CONTROL_DT
        self.rail_goal = self.x_t
        self._push_x()
        return dict(x=SIGN[self.s] * self.x_t, q=q, qd=qd)


class Phrase:
    """The referee for one phrase (from 'Fence' to 'Halt'): touches with the lockout, rear limit (t.34),
    corps-a-corps (t.25). Returns a halt record."""

    def __init__(self, w: World):
        self.w = w
        self.first = None          # (seat, ms)
        self.touches = {}          # seat -> ms
        self.halt = None

    def update(self, rec, k):
        """rec: World.step output for control step k; returns the halt dict or None"""
        w = self.w
        for s in SEATS:
            j = rec["touch"][s]
            if j is not None and s not in self.touches:
                ms = k * SUBSTEPS + j
                if self.first is None:
                    self.first = (s, ms); self.touches[s] = ms
                elif ms - self.first[1] <= DOUBLE_MS:
                    self.touches[s] = ms
        if self.first is not None:
            if len(self.touches) == 2 or (k + 1) * SUBSTEPS - 1 - self.first[1] >= DOUBLE_MS:
                return self._halt("touch", k)
            return None
        for s in SEATS:
            if w.cart_x(s) + CART_HALF < -PISTE_HALF:
                return self._halt("rear_limit", k, seat=s)
        if rec["corps"]:
            return self._halt("corps_a_corps", k)
        return None

    def _halt(self, why, k, seat=None):
        h = dict(why=why, k=k, touches=dict(self.touches))
        if seat:
            h["seat"] = seat
        self.halt = h
        return h
