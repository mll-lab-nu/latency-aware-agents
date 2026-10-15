"""The robot library's whole-body planner (optional control library, bodies/README.md): a tool state at a contact
time -> base motion + torso/waist/arm targets. It plans with the speed limits; the robot's servo (bodies/servo.py)
enforces the acceleration and jerk limits on the way, and the rehearsals see that servo. It works on
any body whose robot_only_xml + rig give the wheeled humanoid's u layout (base x, y, yaw + 9 joints): the wheeled
humanoid (base, lift, waist, arm) and the legged G1's planar copy (pelvis x, y, yaw at its walking speed, waist pitch,
waist yaw, arm).

A sport's high-level action ("the paddle's pose and velocity at contact T s from now") becomes 100 Hz targets of the
body's controlled vector u (body.py), within the body's limits:

  1. stance    candidate base positions put the contact point at a forehand, middle or backhand offset from the base
               (clipped to the sport's stance box); the base tracker (time-optimal under the speed and acceleration
               limits, body.toml [limits.base]) says where the base really is at the contact step;
  2. arm IK    damped least squares on lift, waist and the 7 arm joints with the base there (tool position + face
               normal; the rotation about the normal is free; a null-space pull towards the ready pose); the
               candidate with an exact solution and the shortest joint travel time wins;
  3. velocity  the tool velocity minus the base's own velocity at contact, distributed over the 9 joints by the
               speed-weighted pseudo-inverse (scaled down if a joint would exceed 90 % of its limit);
  4. swing     a cubic Hermite from the targets in force to (q, qd) at the contact step, the shortest one within
               97 % of every joint's speed limit, then a 0.25 s follow-through; every row clipped to the per-step
               speed limits and the ranges; the base keeps tracking its stance;
  5. rehearse  up to `iters` robot-only rollouts of the targets on a twin (the same servo, no world) correct the goal
               pose and velocity for the servo's lag (the Franka tasks' planner did the same).
Everything is in the robot's mount frame (x forward, y left, z up, origin on the floor under the mount); the caller
converts. Deterministic: no randomness here (a sport adds its own seeded execution scatter before calling).
"""
from __future__ import annotations

import math

import numpy as np

JOINTS9 = slice(3, 12)                          # lift, waist, arm_0..6 in u
FOLLOW_STEPS = 25


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def hermite(q0, v0, q1, v1, T, n):
    t = np.arange(1, n + 1) * (T / n)
    s = t / T
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00[:, None] * q0 + h10[:, None] * T * v0 + h01[:, None] * q1 + h11[:, None] * T * v1


def clip_norm(v, lim):
    n = float(np.linalg.norm(v))
    return v if n <= lim or n == 0.0 else v * (lim / n)


def base_track(x, v, goal, vmax, amax, dt):
    """One step of the base tracker (2-D or 1-D): the velocity for the next step towards `goal` (stopping there), the
    change limited to amax * dt (norm) and the speed to vmax. Deterministic, never exceeds either limit."""
    d = np.asarray(goal, float) - x
    dist = float(np.linalg.norm(d))
    if dist < 1e-12:
        v_des = np.zeros_like(d)
    else:
        sp = min(vmax, math.sqrt(2.0 * amax * dist), dist / dt)
        v_des = d * (sp / dist)
    return v + clip_norm(v_des - v, amax * dt)


def track_rows(x0, v0, goal, n, vmax, amax, dt, fence=None):
    """n steps of the tracker from position x0 moving at v0: (positions (n, k), velocities (n, k))."""
    x, v = np.array(x0, float), np.array(v0, float)
    P, V = [], []
    for _ in range(n):
        v_new = base_track(x, v, goal, vmax, amax, dt)
        v = v_new if fence is None else fence(x, v, v_new)
        x = x + v * dt
        P.append(x.copy()); V.append(v.copy())
    return np.array(P).reshape(n, -1), np.array(V).reshape(n, -1)


class WholeBodyPlanner:
    def __init__(self, body, hand="paddle", arm="right", rest=None, swing="short", speed_scale=None, range9=None):
        """body: a body of bodies/ (its robot_only_xml: the wheeled humanoid, or the legged one's planar copy); rest:
        the sport's ready pose (u[3:]; None = the body's own); swing: "short" = the shortest swing within the speed
        limits, started as late as possible (the Franka tasks' way), "full" = the swing uses all the time to contact
        (gentler accelerations: a standing legged robot turns against a fast swing); speed_scale: per joint of the 9
        (u[3:12]) a fraction of its speed limit the plans use (None = all of it); range9: {index in the 9: (lo, hi)}
        narrower joint ranges for the plans (e.g. the waist's turn)."""
        self.swing = swing
        import mujoco as mj
        self.mj, self.B = mj, body
        self.m = mj.MjModel.from_xml_string(body.robot_only_xml(hand=hand, arm=arm))
        self.kin = mj.MjData(self.m)
        self.twin = mj.MjData(self.m)
        self.rig = body.rig(self.m, "", arm, hand, rest)
        r = self.rig
        self.dt = r.dt
        self.VEL9 = r.VEL[JOINTS9] * (1.0 if speed_scale is None else np.asarray(speed_scale, float))
        self.LO9, self.HI9 = r.LO[JOINTS9] + 1e-4, r.HI[JOINTS9] - 1e-4
        for i, (lo, hi) in (range9 or {}).items():
            self.LO9[int(i)], self.HI9[int(i)] = max(self.LO9[int(i)], lo), min(self.HI9[int(i)], hi)
        self.rest = r.rest.copy()
        u = self.rest.copy()
        p, _ = self.fk(u)
        self.rest_tool = p                    # the tool at the ready pose (mount frame, base at the origin)
        self.W9 = np.diag(self.VEL9 ** 2)
        self.guards = self._make_guards(arm)

    def _make_guards(self, arm):
        """Guard points of the forearm, wrist and hand: both ends of every collision capsule (with its radius) and the
        centre of every box (radius = its largest half size) below the body's guard root (the forearm's link)."""
        mj, m = self.mj, self.m
        root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, self.B.guard_root(arm))
        out = []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            b = m.geom_bodyid[g]
            while b not in (0, root):
                b = m.body_parentid[b]
            if b != root:
                continue
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE:
                r, h = float(m.geom_size[g, 0]), float(m.geom_size[g, 1])
                out += [(g, np.array([0.0, 0.0, h]), r), (g, np.array([0.0, 0.0, -h]), r)]
            elif m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX:
                out.append((g, np.zeros(3), float(np.max(m.geom_size[g]))))
        return out

    # ------------------------------------------------------------------ kinematics (mount frame)
    def _set(self, u):
        mj, d = self.mj, self.kin
        d.qpos[self.rig.qadr] = self.rig.joint_targets(u)
        mj.mj_kinematics(self.m, d)
        mj.mj_comPos(self.m, d)

    def fk(self, u):
        self._set(u)
        return self.kin.site_xpos[self.rig.tool_site].copy(), self.kin.site_xmat[self.rig.tool_site].reshape(3, 3).copy()

    def jac(self, u):
        """(6, 12): d(tool position; tool rotation vector) / du."""
        self._set(u)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        self.mj.mj_jacSite(self.m, self.kin, jp, jr, self.rig.tool_site)
        J = np.vstack([jp, jr])[:, self.rig.vadr]
        return J @ self.rig.M

    def _guard_points(self):
        """World positions (n, 3) of the guard points (capsule ends, box centres) at the kinematic state."""
        d = self.kin
        out = np.empty((len(self.guards), 3))
        for i, (g, off, _) in enumerate(self.guards):
            out[i] = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ off
        return out

    def guard_violations(self, u, keep_out):
        """[(guard index, depth below the box top)] for guard points (with their radius) inside a keep-out box."""
        self._set(u)
        P = self._guard_points()
        out = []
        for gi, (g, off, r) in enumerate(self.guards):
            x = P[gi]
            for lo, hi in keep_out:
                if np.all(x >= lo - r) and np.all(x <= hi + r):
                    out.append((gi, float(hi[2] + r - x[2]), x))
        return out

    def ik(self, pos, normal, u0, iters=150, tol=1e-4, pull=0.05, keep_out=(), axis_pref=None):
        """Lift, waist and arm (the base stays at u0's) so that the tool centre is at pos with face normal `normal`
        (the tool's local +x), with the elbow, wrist and hand out of the keep-out boxes [(lo, hi)] (pushed above the
        box top), and optionally a tool axis kept near a world direction: axis_pref = (tool axis index, world direction,
        min cos), e.g. (2, down, 0.3): the handle points down (the wrist above the blade). Returns (u, residual = position
        error + normal error in rad + the deepest guard violation; the axis preference is soft and not counted)."""
        u = np.array(u0, float)
        n_t = np.asarray(normal, float) / np.linalg.norm(normal)
        err = math.inf
        keep_out = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in keep_out]
        for _ in range(iters):
            p, R = self.fk(u)
            n = R[:, 0]
            e = np.r_[pos - p, _cross(n, n_t)]
            err = float(np.linalg.norm(e[:3]) + math.asin(min(1.0, float(np.linalg.norm(e[3:])))))
            if n @ n_t < 0:
                e[3:] = _cross(n, n_t) + 0.5 * _cross(n, R[:, 1])
                err += math.pi / 2
            viol = self.guard_violations(u, keep_out) if keep_out else []
            err += sum(d for _, d, _ in viol)
            off = False
            if axis_pref is not None:
                ai, wd, cmin = axis_pref
                ax = R[:, ai]
                c = float(ax @ wd)
                off = c < cmin - 1e-3
            if err < tol and not off:
                break
            J = self.jac(u)[:, JOINTS9]
            if off:                                   # soft: one row while the tool axis is off the preferred cone
                J = np.vstack([J, 0.3 * (_cross(ax, wd) @ J[3:6])[None, :]])
                e = np.r_[e, 0.3 * (cmin - c)]
            if viol:                                  # one extra row per violation: lift that guard point
                rows, es = [], []
                for gi, depth, x in viol:
                    jp = np.zeros((3, self.m.nv))
                    self.mj.mj_jac(self.m, self.kin, jp, None, x, int(self.m.geom_bodyid[self.guards[gi][0]]))
                    rows.append((jp[2:3, self.rig.vadr] @ self.rig.M)[:, JOINTS9])
                    es.append(depth + 0.005)
                J = np.vstack([J, *rows])
                e = np.r_[e, es]
            JJ = J @ J.T + 1e-3 * np.eye(len(e))
            dq = J.T @ np.linalg.solve(JJ, e)
            null = np.eye(9) - J.T @ np.linalg.solve(JJ, J)
            dq += null @ (pull * (self.rest[JOINTS9] - u[JOINTS9]))
            u[JOINTS9] = np.clip(u[JOINTS9] + dq, self.LO9, self.HI9)
        return u, err

    # ------------------------------------------------------------------ the plan
    def plan(self, u_s, v_s, qpos, qvel, n_avail, pos, normal, vel, box, iters=3, keep_out=(), prefer=None,
             axis_pref=None, seeds=()):
        """u_s: the targets in force (12, mount frame), v_s: their velocity (12, per second); qpos / qvel: the robot's
        joint state (the 21 joints of Rig.joints, as Rig reads them) for the rehearsal; n_avail: control steps until
        contact (the contact happens at the end of row n_avail - 1); pos / normal / vel: the tool at contact; box:
        {"lo": [x, y, yaw], "hi": [...]} the stance limits of the base; keep_out: boxes [(lo, hi)] the elbow, wrist
        and hand stay above at contact (e.g. a table); prefer: a base goal tried first and kept when it works (the goal of
        the previous plan for the same contact: no dithering between stances); seeds: extra IK seeds (u) tried first
        (e.g. the previous plan's contact pose: no flipping between strokes). Returns (rows (list of u), info)."""
        r, dt = self.rig, self.dt
        pos, vel = np.asarray(pos, float), np.asarray(vel, float)
        normal = np.asarray(normal, float) / np.linalg.norm(normal)
        n_avail = int(n_avail)
        info = {"n_swing_steps_available": n_avail}
        if n_avail < 1:
            info["stroke"] = "too late: the contact time has already passed"
            return [], info
        lo, hi = np.asarray(box["lo"], float), np.asarray(box["hi"], float)
        fence = self.fence(box)
        n_total = n_avail + FOLLOW_STEPS
        # 1-3: stance candidates x IK seeds; each gets its contact velocity and its swing, checked against keep_out
        best = None
        turned = self.rest.copy()
        turned[4] -= 0.7                              # a seed turned to the backhand side
        goals = [] if prefer is None else [np.clip(np.asarray(prefer, float), lo[:2], hi[:2])]
        for off_y in (self.rest_tool[1], self.rest_tool[1] * 0.4, -self.rest_tool[1] * 0.35):
            goals.append(np.clip(pos[:2] - np.array([self.rest_tool[0], off_y]), lo[:2], hi[:2]))
        for goal in goals:
            P, V = track_rows(u_s[0:2], v_s[0:2], goal, n_total, self.B.base_speed, self.B.base_accel, dt, fence)
            v_base = np.r_[V[n_avail - 1], 0.0]
            for seed in (*seeds, u_s, self.rest, turned):
                u0 = u_s.copy()
                u0[0:2] = P[n_avail - 1]
                u0[JOINTS9] = seed[JOINTS9]
                uh, err = self.ik(pos, normal, u0, keep_out=keep_out, axis_pref=axis_pref)
                J9 = self.jac(uh)[:3, JOINTS9]
                need = vel - v_base
                qd = self.contact_qd(J9, need, uh[JOINTS9])
                scale = max(1.0, float(np.max(np.abs(qd) / self.VEL9)) / 0.9)
                rows = self._rows(u_s, v_s, P, uh[JOINTS9], qd / scale, n_avail)
                path = self.path_violation(rows, keep_out) if keep_out else 0.0
                travel = float(np.max(np.abs(uh[JOINTS9] - u_s[JOINTS9]) / self.VEL9))
                short = float(np.linalg.norm(P[n_avail - 1] - goal)) > 0.02
                key = (0 if err < 2e-3 else 1, 0 if path < 1e-3 else 1, int(short), travel if err < 2e-3 else err)
                if best is None or key < best[0]:
                    best = (key, uh, err, P, V, goal, qd / scale, scale, need, v_base, path, self._n_sw)
                if key[:3] == (0, 0, 0) and travel < 0.5 * n_avail * dt:
                    break
            if best[0][:3] == (0, 0, 0) and (best[0][3] < 0.5 * n_avail * dt or prefer is not None):
                break
        _, u_h, res, P, V, goal, qd_h, scale, need, v_base, path, _ = best
        info["ik_residual"] = round(float(res), 5)
        info["base_goal"] = [round(float(x), 4) for x in goal]
        self.last_contact = u_h.copy()
        info["speed_scale"] = round(scale, 3)
        info["path_clear"] = bool(path < 1e-3)
        # 4-5: swing + rehearsals
        q_goal, qd_goal = u_h[JOINTS9].copy(), qd_h.copy()
        best = None
        for it in range(max(1, iters)):
            rows = self._rows(u_s, v_s, P, q_goal, qd_goal, n_avail)
            if iters <= 0:
                best = (0.0, rows, None, None, None)
                break
            p, n_act, v_act = self._rehearse(rows, u_s, u_s - v_s * dt, qpos, qvel, n_avail)
            ang = float(np.arccos(np.clip(n_act @ normal, -1, 1)))
            loss = float(np.linalg.norm(p - pos) + 0.1 * ang + 0.01 * np.linalg.norm(v_act - (v_base + need / scale)))
            if best is None or loss < best[0]:
                best = (loss, rows, p, n_act, v_act)
            if np.linalg.norm(p - pos) < 5e-4 and ang < math.radians(0.1) or it == iters - 1:
                break
            ug = u_h.copy(); ug[JOINTS9] = q_goal
            J = self.jac(ug)[:, JOINTS9]
            e = np.r_[pos - p, _cross(n_act, normal)]
            q_goal = np.clip(q_goal + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.LO9, self.HI9)
            Jp = J[:3]
            qd_goal = qd_goal + self.W9 @ Jp.T @ np.linalg.solve(Jp @ self.W9 @ Jp.T + 1e-9 * np.eye(3),
                                                                 (v_base + need / scale) - v_act) * 0.7
            qd_goal = qd_goal / max(1.0, float(np.max(np.abs(qd_goal) / self.VEL9)))
        _, rows, p, n_act, v_act = best
        notes = []
        if p is not None:
            info["blade_error_m"] = round(float(np.linalg.norm(p - pos)), 4)
            info["normal_error_deg"] = round(float(np.degrees(np.arccos(np.clip(n_act @ normal, -1, 1)))), 3)
            info["paddle_vel_delivered"] = [round(float(x), 3) for x in v_act]
            if info["blade_error_m"] > .03 or info["normal_error_deg"] > 5:
                notes.append("command not achievable accurately: see blade_error_m / normal_error_deg")
        if res > 2e-3:
            notes.append(f"pose not reachable exactly (IK residual {res:.3f})")
        if scale > 1.0:
            notes.append(f"paddle speed limited by the joint speed limits (x{1 / scale:.2f})")
        if not info["path_clear"]:
            notes.append("the swing passes close to the table (keep-out)")
        if np.linalg.norm(P[n_avail - 1] - goal) > 0.01:
            notes.append("the base cannot reach its stance in time (speed / acceleration limits)")
        info["stroke"] = "; ".join(notes) or "ok"
        info["n_swing_steps"] = self._n_sw
        return rows, info

    def contact_qd(self, J9, need, q9, margin=0.15):
        """The joint velocities at contact for the tool velocity `need`: the speed-weighted pseudo-inverse, without the
        joints that are within `margin` rad of a range limit and would have to arrive moving away from it (that needs
        the joint beyond its limit just before the contact: the clipped swing would arrive with no speed at all)."""
        w = self.VEL9 ** 2
        for _ in range(4):
            W = np.diag(w)
            qd = W @ J9.T @ np.linalg.solve(J9 @ W @ J9.T + 1e-9 * np.eye(3), need)
            blocked = ((q9 > self.HI9 - margin) & (qd < 0)) | ((q9 < self.LO9 + margin) & (qd > 0))
            blocked &= w > 0
            if not blocked.any():
                break
            w = np.where(blocked, 0.0, w)
        return qd

    def path_violation(self, rows, keep_out, every=3):
        """The deepest guard-point violation of keep_out along the rows (kinematic, sampled every few rows)."""
        boxes = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in keep_out]
        worst = 0.0
        for i in range(0, len(rows), every):
            for _, depth, _ in self.guard_violations(rows[i], boxes):
                worst = max(worst, depth)
        return worst

    def _rows(self, u_s, v_s, P, q_goal, qd_goal, n_avail):
        dt, r = self.dt, self.rig
        q_s, vq_s = u_s[JOINTS9], v_s[JOINTS9]
        n_sw = n_avail
        if np.max(np.abs(vq_s)) < 1e-6 and self.swing == "short":
            for n in range(25, n_avail + 1):
                sw = hermite(q_s, np.zeros(9), q_goal, qd_goal, n * dt, n)
                if np.max(np.abs(np.diff(np.vstack([q_s, sw]), axis=0)) / (self.VEL9 * dt)) <= 0.97:
                    n_sw = n
                    break
        self._n_sw = n_sw = int(min(n_sw, n_avail))
        pre = np.tile(q_s, (n_avail - n_sw, 1))
        swing = hermite(q_s, vq_s if n_sw == n_avail else np.zeros(9), q_goal, qd_goal, n_sw * dt, n_sw)
        follow = hermite(q_goal, qd_goal, q_goal + qd_goal * 0.5 * FOLLOW_STEPS * dt, np.zeros(9),
                         FOLLOW_STEPS * dt, FOLLOW_STEPS)
        Q = np.vstack([pre, swing, follow])
        lim = np.minimum(r.STEP[JOINTS9], self.VEL9 * dt) * 0.999
        rows, prev = [], q_s
        for i, q in enumerate(Q):
            q = np.clip(q, self.LO9, self.HI9)
            q = prev + np.clip(q - prev, -lim, lim)
            u = np.empty(12)
            u[0:2] = P[i]
            u[2] = u_s[2]
            u[JOINTS9] = q
            rows.append(u); prev = q
        return rows

    def _rehearse(self, rows, u_s, u_p, qpos, qvel, n):
        mj, m, d, r = self.mj, self.m, self.twin, self.rig
        mj.mj_resetData(m, d)
        d.qpos[r.qadr] = qpos
        d.qvel[r.vadr] = qvel
        mj.mj_forward(m, d)
        r.sync(d)                                  # the twin's servo setpoints from the state (bodies/servo.py)
        prev, prev2 = np.array(u_s, float), np.array(u_p, float)
        for j in range(n):
            for s in range(1, r.substeps + 1):
                r.apply(d, prev, rows[j], s, prev2)
                mj.mj_step(m, d)
            prev2, prev = prev, rows[j]
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mj.mj_jacSite(m, d, jp, jr, r.tool_site)
        p, R = r.tool(d)
        return p, R[:, 0], jp @ d.qvel

    def fence(self, box):
        """The stance limits as a velocity filter (fence_xy)."""
        lo, hi = np.asarray(box["lo"], float)[:2], np.asarray(box["hi"], float)[:2]
        a, dt = self.B.base_accel / math.sqrt(2.0), self.dt
        return lambda x, v_old, v_new: fence_xy(x, v_old, v_new, lo, hi, a, dt)


def fence_xy(x, v_old, v_new, lo, hi, a_axis, dt):
    """Stance limits: an axis whose stopping distance (at a_axis) reaches the box edge brakes at a_axis; the other
    axis then changes by at most a_axis * dt, so the base acceleration stays within sqrt(2) a_axis. The base may pass
    the edge by less than a_axis dt^2 (0.1 mm)."""
    out, braked = np.array(v_new, float), []
    for i in range(len(out)):
        vo = float(v_old[i])
        if vo == 0.0:
            continue
        room = (hi[i] - x[i]) if vo > 0 else (x[i] - lo[i])
        sp = abs(vo)
        if room <= sp * dt + sp * sp / (2 * a_axis):
            out[i] = math.copysign(max(0.0, sp - a_axis * dt), vo)
            braked.append(i)
    if braked:
        for j in range(len(out)):
            if j not in braked:
                out[j] = v_old[j] + float(np.clip(v_new[j] - v_old[j], -a_axis * dt, a_axis * dt))
    return out
