"""The air-hockey pilot's skill planner: turns the high-level skills `move` and `strike` into plans of joint targets
(one row of the 9 joint targets per 10 ms control step) within the real KUKA iiwa 14 limits. Built ONLY on the
low-level toolkit (toolkit/lib/ahkin.py: forward kinematics, Jacobians, IK; the joint limits), so the same file is
both the platform's implementation of the high-level API (world.py, in the trusted simulator) and the core of our
low-level reference implementation (programs/ref_low_skill.py, which runs in the player's container).

  move(q, qd, xy)                  -> rows: the mallet to (x, y), arriving at rest (a joint-space Hermite move from
                                      the current target state, as fast as the limits allow)
  strike(q, qd, t_now, req)        -> {rows, contact_t, speed, ...}: req = {t, x, y, dx, dy, speed}: at game time t the
                                      puck centre is at (x, y); the mallet hits it moving along the unit direction
                                      (dx, dy) at `speed` m/s. Plan: approach the wind-up point W behind the contact
                                      point C = (x, y) - CONTACT (dx, dy), settle there (DWELL), then drive along the
                                      line with a constant acceleration so that the mallet passes C at time t with the
                                      speed, follow through and stop. The drive's acceleration and, if needed, the speed
                                      are chosen so that every joint stays within its velocity and acceleration limits.

Seat A's frame (the table frame, the own half x < 0). Rows are targets for the ends of the next control steps:
row i is the target at t_now + (i + 1) DT."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _p in (_here / "toolkit" / "lib", _here.parent / "lib", Path("/opt/robot/lib")):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ahkin  # noqa: E402

DT = 0.01
PUCK_R, MALLET_R = .03165, .04815
CONTACT = PUCK_R + MALLET_R
X_NEAR, X_FAR, Y_MAX, REACH, BASE_X = .05, .92, .47, 1.45, 1.51
DWELL = 0.08            # s at rest at the wind-up point before the drive (the arm settles; B9: strike from rest)
LAG = 0.006             # s: the servo's setpoint generator lags the targets; the plan runs this much ahead
MARGIN_V, MARGIN_A = 0.97, 0.80     # planning margins on the servo's velocity and acceleration limits
DRIVE_ACCELS = (7.0, 5.5, 4.5, 3.5, 2.6, 1.9)    # m/s^2 candidates for the drive (the largest feasible is used)
SPEED_STEPS = (1.0, .88, .76, .64, .52, .4)
SPEED_MIN, SPEED_MAX = 0.3, 2.5
MAX_ROWS = 300
CLEAR = 0.015           # m: the approach keeps the mallet this far beyond contact with the puck (near P)


def local(xy):
    return np.array([-float(xy[0]), float(xy[1])])


def clip(xy, m=0.0):
    """the nearest point of seat A's workspace shrunk by m (box, then radially towards the base)"""
    u, v = local(xy)
    u, v = min(max(u, X_NEAR + m), X_FAR - m), min(max(v, -Y_MAX + m), Y_MAX - m)
    dd = math.hypot(BASE_X - u, v)
    if dd > REACH - m:
        u, v = BASE_X - (BASE_X - u) * (REACH - m) / dd, v * (REACH - m) / dd
    return np.array([-u, v])


def inside(xy, m=0.0):
    """seat A's mallet workspace (own half X_NEAR <= -x <= X_FAR, |y| <= Y_MAX, within REACH of the base), shrunk by m"""
    u, v = local(xy)
    return X_NEAR + m <= u <= X_FAR - m and abs(v) <= Y_MAX - m and math.hypot(BASE_X - u, v) <= REACH - m


class Planner:
    def __init__(self, kin: ahkin.Kin | None = None):
        self.kin = kin or ahkin.Kin()
        self.V = ahkin.VEL * MARGIN_V
        self.A = ahkin.ACC * MARGIN_A

    # ------------------------------------------------------------------ checks
    def rows_ok(self, rows, q0, qd0):
        """the rows (with the current target q0 moving at qd0 before them) within the planning margins"""
        a = np.vstack([q0[None], rows])
        v = np.diff(a, axis=0) / DT
        if np.any(np.abs(v) > self.V):
            return False
        acc = np.diff(np.vstack([qd0[None], v]), axis=0) / DT
        return not np.any(np.abs(acc) > self.A)

    # ------------------------------------------------------------------ joint moves
    def approach(self, q0, qd0, q1, t_min=0.0):
        """rows that take the mallet from the current target state (q0 moving at qd0) to where q1 puts it, at rest:
        a cubic Hermite path of the mallet centre on the table (from its current velocity), IK row by row (warm
        started, so the arm stays on one branch; the mallet flat and within the workspace), the shortest duration
        (at least t_min) whose rows keep every joint within the velocity and acceleration limits"""
        q0, qd0, q1 = (np.asarray(x, float) for x in (q0, qd0, q1))
        p0 = self.kin.fk(q0)[0][:2]
        p1 = clip(self.kin.fk(q1)[0][:2], .004)
        v0 = (self.kin.jac(q0)[0] @ qd0)[:2]
        D = float(np.linalg.norm(p1 - p0))
        T = max(t_min, .05, 1.4 * D / 1.0, math.sqrt(6.0 * D / 6.0), float(np.linalg.norm(v0)) / 4.0)
        rows = None
        for _ in range(14):
            n = max(1, int(math.ceil(T / DT - 1e-9)))
            s = (np.arange(1, n + 1) / n)[:, None]
            h00, h10, h01 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2
            path = h00 * p0 + h10 * (n * DT) * v0 + h01 * p1
            rows = np.empty((n, ahkin.NQ))
            q = q0
            for i, pt in enumerate(path):
                q = self.kin.ik(clip(pt, .004), q, 20)[0]
                rows[i] = q
            if self.rows_ok(rows, q0, qd0):
                return rows
            T *= 1.2
        return rows

    def move(self, q, qd, xy):
        """rows that bring the mallet to xy (clipped into the workspace), at rest"""
        q = np.asarray(q, float)
        u, v = local(xy)
        u, v = min(max(u, X_NEAR), X_FAR), min(max(v, -Y_MAX), Y_MAX)
        dd = math.hypot(BASE_X - u, v)
        if dd > REACH:
            u, v = BASE_X - (BASE_X - u) * REACH / dd, v * REACH / dd
        q1, _ = self.kin.ik([-u, v], q, 80)
        return self.approach(q, qd, q1)

    def clearance(self, rows, P):
        """the smallest distance between the mallet centre along the rows and the puck centre P"""
        return min(float(np.linalg.norm(self.kin.fk(r)[0][:2] - P)) for r in rows[::2]) if len(rows) else 9.0

    def clear_approach(self, q, qd, qW, W, P, d):
        """the approach to the wind-up point, around the puck (assumed near P) if the direct move would touch it: via a
        point beside the puck on the mallet's side (as the reference player goes round it)"""
        app = self.approach(q, qd, qW)
        if self.clearance(app, P) >= CONTACT + CLEAR:
            return app
        m = self.kin.fk(q)[0][:2]
        n = np.array([-d[1], d[0]])
        s0 = 1.0 if float(n @ (m - P)) >= 0 else -1.0
        best = app
        for s in (s0, -s0):
            for V in (P + n * s * .17, P - d * .12 + n * s * .15, P - d * .2 + n * s * .1):
                if not inside(V, .01):
                    continue
                qV, res = self.kin.ik(V, q, 80)
                if res > .003:
                    continue
                a1 = self.approach(q, qd, qV)
                a2 = self.approach(qV, np.zeros(ahkin.NQ), qW)
                rows = np.vstack([a1, a2])
                if self.clearance(rows, P) >= CONTACT + CLEAR:
                    return rows
        return best

    # ------------------------------------------------------------------ the strike
    def speed_cap(self, qC, d):
        """the largest mallet speed along d at the configuration qC that keeps every joint within the planning
        velocity limits (the mallet's orientation held)"""
        Jp, Jr = self.kin.jac(qC)
        J = np.vstack([Jp, Jr])
        qd_unit = np.linalg.pinv(J) @ np.r_[d[0], d[1], 0, 0, 0, 0]
        return float(np.min(self.V / np.maximum(np.abs(qd_unit), 1e-9)))

    def line(self, q_start, P0, d, s_vals):
        """IK along the line P0 + s d (warm-started); None if the IK fails (residual > 3 mm)"""
        out = np.empty((len(s_vals), ahkin.NQ))
        q = q_start
        for i, s in enumerate(s_vals):
            q, res = self.kin.ik(P0 + s * d, q, 30)
            if res > .003:
                return None
            out[i] = q
        return out

    def strike(self, q, qd, t_now, req):
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        P = np.array([float(req["x"]), float(req["y"])])
        d = np.array([float(req["dx"]), float(req["dy"])])
        d = d / max(1e-9, float(np.linalg.norm(d)))
        v_req = min(max(float(req["speed"]), SPEED_MIN), SPEED_MAX)
        t_c = float(req["t"])
        C = P - CONTACT * d
        out = {"contact_point": C.tolist(), "requested_speed": v_req, "feasible": False}
        if not inside(C, .005):
            out["why"] = "the contact point (the mallet centre at contact) lies outside the workspace"
            return out
        qC, res = self.kin.ik(C, q, 100)
        if res > .003:
            out["why"] = "no IK solution at the contact point"
            return out
        v_cap = self.speed_cap(qC, d)
        best = None
        tried = set()
        for f in SPEED_STEPS:
            v = round(min(v_req, v_cap) * f, 4)
            if v in tried:
                continue
            tried.add(v)
            for a in DRIVE_ACCELS:
                L = v * v / (2 * a)
                W = C - L * d
                if not inside(W, .005):
                    continue
                # follow-through: decelerate at a (more if the workspace ends sooner)
                Lf = v * v / (2 * a)
                af = a
                k = 0
                while not inside(C + Lf * d, .005) and k < 30:
                    Lf *= .85
                    k += 1
                if not inside(C + Lf * d, .005):
                    Lf = 0.0
                af = v * v / (2 * max(Lf, 1e-3))
                if af > 3 * a + 6:
                    continue
                Td, Tf = v / a, v / af
                taus = np.arange(int(math.ceil((Td + Tf) / DT)) + 2) * DT
                u = np.clip(taus - Td, 0.0, Tf)
                s = np.where(taus <= Td, .5 * a * taus**2, L + v * u - .5 * af * u * u)
                qW, resW = self.kin.ik(W, qC, 100)
                if resW > .003:
                    continue
                Q = self.line(qW, W, d, s)
                if Q is None:
                    continue
                # velocity / acceleration of the drive rows (sampled at DT, from rest at W)
                if not self.rows_ok(Q[1:], Q[0], np.zeros(ahkin.NQ)):
                    continue
                best = dict(v=v, a=a, af=af, L=L, Lf=Lf, Td=Td, W=W, qW=qW, s=s, taus=taus)
                break
            if best is not None:
                break
        if best is None:
            out["why"] = "no drive within the joint limits on this line (the wind-up point outside the workspace?)"
            return out
        v, a, af, L, Td = best["v"], best["a"], best["af"], best["L"], best["Td"]
        # the approach to W, then the dwell, then the drive starting at t_s
        app = self.clear_approach(q, qd, best["qW"], best["W"], P, d)
        t_app_end = t_now + len(app) * DT
        t_s = t_c - Td - LAG
        late = max(0.0, t_app_end + DWELL - t_s)
        t_s += late
        # rows at t_now + (i + 1) DT
        n_total = int(math.ceil((t_s + best["taus"][-1] - t_now) / DT)) + 1
        rows = np.empty((n_total, ahkin.NQ))
        rows[:len(app)] = app[:n_total]
        Lf, Tf = best["Lf"], v / af
        qprev = app[-1] if len(app) else best["qW"]
        for i in range(len(app), n_total):
            t = t_now + (i + 1) * DT
            tau = t - t_s
            if tau <= 0:
                rows[i] = qprev
                continue
            if tau <= Td:
                s = .5 * a * tau * tau
            else:
                u = min(tau - Td, Tf)
                s = L + v * u - .5 * af * u * u
            qprev, _ = self.kin.ik(best["W"] + s * d, qprev, 30)
            rows[i] = qprev
        rows = rows[:MAX_ROWS]
        out.update(rows=rows, feasible=late <= 0.0 and v >= v_req * .999, contact_t=round(t_s + Td + LAG, 4),
                   late_s=round(late, 4), speed=round(v, 4), speed_cap=round(v_cap, 4), drive_accel=a,
                   windup_point=best["W"].tolist(), windup_m=round(L, 4), follow_m=round(Lf, 4),
                   why="ok" if late <= 0 and v >= v_req * .999 else
                   ("late: not enough time to wind up" if late > 0 else "speed reduced to the joint limits"))
        return out
